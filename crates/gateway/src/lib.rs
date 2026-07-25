//! Transparent HTTP forwarding for Camelid Enterprise inference replicas.
//!
//! The gateway treats request and response bodies as opaque streams. It does
//! not retry, inspect, buffer, or rewrite inference traffic; the replica remains
//! the authority for output and attribution.

use axum::body::Body;
use axum::extract::{Request, State};
use axum::http::header::{AUTHORIZATION, CONNECTION, CONTENT_TYPE, HOST, WWW_AUTHENTICATE};
use axum::http::uri::{Authority, Scheme};
use axum::http::{HeaderMap, HeaderName, HeaderValue, StatusCode, Uri};
use axum::response::{IntoResponse, Response};
use axum::{Json, Router};
use hyper_util::client::legacy::connect::HttpConnector;
use hyper_util::client::legacy::Client;
use hyper_util::rt::TokioExecutor;
use identity::{IdentityError, PrincipalId, SqliteIdentityStore, TokenStore};
use std::fmt;
use std::sync::Arc;
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
    auth: GatewayAuth,
}

/// Whether the gateway requires every request to carry a bearer token that
/// resolves to a known principal before it is forwarded upstream.
///
/// [`GatewayAuth::Disabled`] preserves the gateway's original transparent
/// behavior unchanged: nothing about existing deployments breaks until an
/// operator opts in by supplying an identity database.
#[derive(Clone)]
pub enum GatewayAuth {
    Disabled,
    RequireToken(Arc<SqliteIdentityStore>),
}

// Merge-order constraint for the pending admission-control rebase:
//
// This branch is cut from `main` before admission control (a bounded
// in-flight semaphore, tracked in the separate gateway-hardening PR) lands.
// When this branch rebases onto that work, authentication here must run
// *before* the admission permit is acquired: otherwise an unauthenticated
// flood consumes permits (and the SQLite lookup time behind each one) before
// ever being rejected, starving legitimate authenticated traffic. Any local
// health-check route that work introduces (e.g. `/healthz`) must also stay
// exempt from this auth check, the way it stays exempt from admission.

pub fn router(upstream: UpstreamOrigin, auth: GatewayAuth) -> Router {
    let mut connector = HttpConnector::new();
    connector.enforce_http(true);
    connector.set_connect_timeout(Some(CONNECT_TIMEOUT));
    let mut client_builder = Client::builder(TokioExecutor::new());
    client_builder.retry_canceled_requests(false);
    let client = client_builder.build(connector);
    Router::new().fallback(proxy).with_state(GatewayState {
        upstream,
        client,
        auth,
    })
}

async fn proxy(State(state): State<GatewayState>, mut request: Request) -> Response {
    if let GatewayAuth::RequireToken(store) = &state.auth {
        match authenticate(store, request.headers()).await {
            Ok(principal) => {
                request.extensions_mut().insert(principal);
            }
            Err(response) => return response,
        }
    }

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

/// Resolves the request's bearer token to a principal, or returns the exact
/// response the gateway should send back without forwarding anything upstream.
async fn authenticate(
    store: &Arc<SqliteIdentityStore>,
    headers: &HeaderMap,
) -> Result<PrincipalId, Response> {
    let Some(token) = bearer_token(headers) else {
        return Err(unauthorized("missing bearer token"));
    };
    let store = Arc::clone(store);
    let token = token.to_string();
    let resolved = tokio::task::spawn_blocking(move || store.resolve(&token))
        .await
        .expect("identity resolution task panicked");

    resolved.map_err(|error| match error {
        IdentityError::InvalidToken => unauthorized("invalid bearer token"),
        IdentityError::Storage(message) => {
            tracing::error!(%message, "identity store error while authenticating request");
            gateway_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "gateway could not verify the bearer token",
            )
        }
    })
}

fn bearer_token(headers: &HeaderMap) -> Option<&str> {
    let value = headers.get(AUTHORIZATION)?.to_str().ok()?;
    // RFC 7235 defines the auth-scheme token ("Bearer") as case-insensitive;
    // some clients and SDKs send `bearer` or `BEARER`. Match accordingly and
    // tolerate any amount of whitespace between the scheme and the token.
    let (scheme, rest) = value.split_once(char::is_whitespace)?;
    if !scheme.eq_ignore_ascii_case("bearer") {
        return None;
    }
    let token = rest.trim();
    (!token.is_empty()).then_some(token)
}

fn unauthorized(message: &'static str) -> Response {
    let mut response = (
        StatusCode::UNAUTHORIZED,
        Json(serde_json::json!({
            "error": {
                "message": message,
                "type": "unauthorized"
            }
        })),
    )
        .into_response();
    response
        .headers_mut()
        .insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
    response
        .headers_mut()
        .insert(WWW_AUTHENTICATE, HeaderValue::from_static("Bearer"));
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

    #[test]
    fn bearer_token_matches_the_scheme_case_insensitively() {
        for scheme in ["Bearer", "bearer", "BEARER", "BeArEr"] {
            let mut headers = HeaderMap::new();
            headers.insert(
                AUTHORIZATION,
                HeaderValue::from_str(&format!("{scheme} secret-token")).unwrap(),
            );
            assert_eq!(
                bearer_token(&headers),
                Some("secret-token"),
                "scheme {scheme}"
            );
        }
    }

    #[test]
    fn bearer_token_tolerates_extra_whitespace() {
        let mut headers = HeaderMap::new();
        headers.insert(
            AUTHORIZATION,
            HeaderValue::from_static("Bearer    secret-token   "),
        );
        assert_eq!(bearer_token(&headers), Some("secret-token"));
    }

    #[test]
    fn bearer_token_rejects_other_schemes_and_missing_headers() {
        let mut headers = HeaderMap::new();
        assert_eq!(bearer_token(&headers), None);

        headers.insert(
            AUTHORIZATION,
            HeaderValue::from_static("Basic dXNlcjpwYXNz"),
        );
        assert_eq!(bearer_token(&headers), None);

        headers.insert(AUTHORIZATION, HeaderValue::from_static("Bearer"));
        assert_eq!(bearer_token(&headers), None);

        headers.insert(AUTHORIZATION, HeaderValue::from_static("Bearer   "));
        assert_eq!(bearer_token(&headers), None);
    }
}
