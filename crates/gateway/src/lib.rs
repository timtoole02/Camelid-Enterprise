//! Transparent HTTP forwarding for Camelid Enterprise inference replicas.
//!
//! The gateway treats request and response bodies as opaque streams. It does
//! not retry, inspect, buffer, or rewrite inference traffic; the replica remains
//! the authority for output and attribution.

use axum::body::Body;
use axum::extract::{Request, State};
use axum::http::header::{CONNECTION, CONTENT_TYPE, HOST};
use axum::http::uri::{Authority, Scheme};
use axum::http::{HeaderMap, HeaderName, HeaderValue, StatusCode, Uri};
use axum::response::{IntoResponse, Response};
use axum::{Json, Router};
use hyper_util::client::legacy::connect::HttpConnector;
use hyper_util::client::legacy::Client;
use hyper_util::rt::TokioExecutor;
use std::fmt;
use std::time::Duration;

const CONNECT_TIMEOUT: Duration = Duration::from_secs(5);

#[derive(Clone, Debug)]
pub struct UpstreamOrigin {
    scheme: Scheme,
    authority: Authority,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InvalidUpstream(String);

impl fmt::Display for InvalidUpstream {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl std::error::Error for InvalidUpstream {}

impl UpstreamOrigin {
    pub fn parse(value: &str) -> Result<Self, InvalidUpstream> {
        let uri: Uri = value
            .parse()
            .map_err(|_| InvalidUpstream("upstream must be a valid HTTP origin".into()))?;
        let scheme = uri
            .scheme()
            .cloned()
            .ok_or_else(|| InvalidUpstream("upstream must include http://".into()))?;
        if scheme != Scheme::HTTP {
            return Err(InvalidUpstream(
                "upstream must use http:// in this release".into(),
            ));
        }
        let authority = uri
            .authority()
            .cloned()
            .ok_or_else(|| InvalidUpstream("upstream must include a host".into()))?;
        if uri.path() != "/" || uri.query().is_some() {
            return Err(InvalidUpstream(
                "upstream must be an origin without a path or query".into(),
            ));
        }
        Ok(Self { scheme, authority })
    }

    fn request_uri(&self, incoming: &Uri) -> Result<Uri, axum::http::uri::InvalidUriParts> {
        let mut parts = axum::http::uri::Parts::default();
        parts.scheme = Some(self.scheme.clone());
        parts.authority = Some(self.authority.clone());
        parts.path_and_query = incoming.path_and_query().cloned();
        Uri::from_parts(parts)
    }
}

#[derive(Clone)]
struct GatewayState {
    upstream: UpstreamOrigin,
    client: Client<HttpConnector, Body>,
}

pub fn router(upstream: UpstreamOrigin) -> Router {
    let mut connector = HttpConnector::new();
    connector.enforce_http(true);
    connector.set_connect_timeout(Some(CONNECT_TIMEOUT));
    let mut client_builder = Client::builder(TokioExecutor::new());
    client_builder.retry_canceled_requests(false);
    let client = client_builder.build(connector);
    Router::new()
        .fallback(proxy)
        .with_state(GatewayState { upstream, client })
}

async fn proxy(State(state): State<GatewayState>, mut request: Request) -> Response {
    let upstream_uri = match state.upstream.request_uri(request.uri()) {
        Ok(uri) => uri,
        Err(error) => {
            tracing::error!(%error, "could not construct upstream request URI");
            return gateway_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "gateway could not construct the upstream request",
            );
        }
    };

    *request.uri_mut() = upstream_uri;
    remove_hop_by_hop(request.headers_mut());
    let host = HeaderValue::from_str(state.upstream.authority.as_str())
        .expect("a valid URI authority is a valid Host header");
    request.headers_mut().insert(HOST, host);

    match state.client.request(request).await {
        Ok(response) => {
            let (mut parts, body) = response.into_parts();
            remove_hop_by_hop(&mut parts.headers);
            Response::from_parts(parts, Body::new(body))
        }
        Err(error) => {
            tracing::warn!(%error, "upstream replica request failed");
            gateway_error(StatusCode::BAD_GATEWAY, "upstream replica is unavailable")
        }
    }
}

fn gateway_error(status: StatusCode, message: &'static str) -> Response {
    let mut response = (
        status,
        Json(serde_json::json!({
            "error": {
                "message": message,
                "type": "gateway_error"
            }
        })),
    )
        .into_response();
    response
        .headers_mut()
        .insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
    response
}

fn remove_hop_by_hop(headers: &mut HeaderMap) {
    let connection_headers: Vec<HeaderName> = headers
        .get_all(CONNECTION)
        .iter()
        .flat_map(|value| value.as_bytes().split(|byte| *byte == b','))
        .filter_map(|name| HeaderName::from_bytes(trim_optional_whitespace(name)).ok())
        .collect();

    for name in connection_headers {
        headers.remove(name);
    }
    for name in [
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "proxy-connection",
        "te",
        "trailer",
        "transfer-encoding",
        "upgrade",
    ] {
        headers.remove(name);
    }
}

fn trim_optional_whitespace(mut value: &[u8]) -> &[u8] {
    while matches!(value.first(), Some(b' ' | b'\t')) {
        value = &value[1..];
    }
    while matches!(value.last(), Some(b' ' | b'\t')) {
        value = &value[..value.len() - 1];
    }
    value
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::http::header::HeaderName;

    #[test]
    fn upstream_must_be_a_plain_http_origin() {
        assert!(UpstreamOrigin::parse("http://127.0.0.1:8181").is_ok());
        assert!(UpstreamOrigin::parse("http://replica.default.svc").is_ok());
        assert_eq!(
            UpstreamOrigin::parse("https://example.test")
                .unwrap_err()
                .to_string(),
            "upstream must use http:// in this release"
        );
        assert_eq!(
            UpstreamOrigin::parse("http://example.test/v1")
                .unwrap_err()
                .to_string(),
            "upstream must be an origin without a path or query"
        );
    }

    #[test]
    fn removes_standard_and_connection_nominated_headers() {
        let mut headers = HeaderMap::new();
        headers.insert(
            CONNECTION,
            HeaderValue::from_static("keep-alive, x-private"),
        );
        headers.insert("keep-alive", HeaderValue::from_static("timeout=5"));
        headers.insert("x-private", HeaderValue::from_static("remove-me"));
        headers.insert("x-camelid-lane", HeaderValue::from_static("deterministic"));

        remove_hop_by_hop(&mut headers);

        assert!(!headers.contains_key(CONNECTION));
        assert!(!headers.contains_key("keep-alive"));
        assert!(!headers.contains_key("x-private"));
        assert_eq!(
            headers[HeaderName::from_static("x-camelid-lane")],
            "deterministic"
        );
    }

    #[test]
    fn removes_valid_connection_tokens_even_when_another_token_is_non_utf8() {
        let mut headers = HeaderMap::new();
        headers.insert(
            CONNECTION,
            HeaderValue::from_bytes(b"x-private,\xff").unwrap(),
        );
        headers.insert("x-private", HeaderValue::from_static("remove-me"));

        remove_hop_by_hop(&mut headers);

        assert!(!headers.contains_key("x-private"));
    }
}
