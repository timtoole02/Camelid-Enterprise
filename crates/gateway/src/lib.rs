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
use axum::routing::{get, post};
use axum::{Json, Router};
use http_body::{Body as HttpBody, Frame, SizeHint};
use hyper_util::client::legacy::connect::HttpConnector;
use hyper_util::client::legacy::Client;
use hyper_util::rt::{TokioExecutor, TokioTimer};
use identity::{IdentityError, PrincipalId, SqliteIdentityStore, TokenStore};
use pin_project_lite::pin_project;
use std::fmt;
use std::io::Write;
use std::num::NonZeroUsize;
use std::path::PathBuf;
use std::pin::Pin;
use std::sync::{Arc, Mutex, OnceLock};
use std::task::{Context, Poll};
use std::time::Duration;
use tokio::sync::{OwnedSemaphorePermit, Semaphore};
use tower_http::cors::CorsLayer;

const CONNECT_TIMEOUT: Duration = Duration::from_secs(5);
const POOL_IDLE_TIMEOUT: Duration = Duration::from_secs(30);
const MAX_IDLE_PER_HOST: usize = 32;
/// Retry-After is jittered across this inclusive range (seconds) so that
/// clients rejected at the same instant do not retry in lockstep and
/// re-create the exact saturation spike that rejected them.
const RETRY_AFTER_JITTER_SECONDS: (u8, u8) = (1, 3);
pub const DEFAULT_MAX_IN_FLIGHT: NonZeroUsize = NonZeroUsize::new(256).unwrap();
/// Every accepted client connection is force-closed after this long,
/// regardless of activity. Nothing else in this module bounds how long a
/// single HTTP exchange may run: the admission permit in [`PermitBody`] is
/// held for the full request+response lifetime, so a client that drips a
/// request body one byte at a time, or opens a response stream and never
/// reads it, would otherwise pin a permit (and a TCP socket) indefinitely.
/// This is a coarse backstop, not an idle timeout: legitimate long-running
/// generations must complete within this bound. Size it to the slowest real
/// completion the deployment expects, with margin.
pub const DEFAULT_MAX_CONNECTION_DURATION: Duration = Duration::from_secs(300);
/// Gateway-authoritative request correlation id header. The gateway sets this
/// on every forwarded request, overwriting any value a client sent, and the
/// replica records it in its serving receipt. A gateway audit record and the
/// replica receipt that served the same request can then be joined on this id.
/// The id is opaque: it carries no identity, so the replica stays
/// identity-blind.
const REQUEST_ID_HEADER: &str = "x-camelid-request-id";

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
    admission: Arc<Semaphore>,
    auth: GatewayAuth,
    /// Optional append-only JSONL audit sink. When set, the gateway records one
    /// line per request it handles — `{ts, request_id, principal, method, path,
    /// status}` — including requests it rejects for authentication or
    /// admission. `None` disables auditing entirely. CORS preflight and
    /// `/healthz` are answered before this path and are never audited.
    audit: Option<Arc<PathBuf>>,
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

pub fn router(upstream: UpstreamOrigin) -> Router {
    router_with_max_in_flight(upstream, DEFAULT_MAX_IN_FLIGHT)
}

pub fn router_with_max_in_flight(upstream: UpstreamOrigin, max_in_flight: NonZeroUsize) -> Router {
    router_with_options(upstream, max_in_flight, GatewayAuth::Disabled, None)
}

// This route table is a hand-maintained mirror of the pinned engine's public
// `/v1` surface (see `crates/replica-contract` for the authoritative,
// tested inventory once it lands). If the pinned engine adds, removes, or
// renames a public route, this list must be updated by hand: nothing today
// verifies the gateway's allowlist and the replica's actual contract agree.
pub fn router_with_options(
    upstream: UpstreamOrigin,
    max_in_flight: NonZeroUsize,
    auth: GatewayAuth,
    audit: Option<Arc<PathBuf>>,
) -> Router {
    let mut connector = HttpConnector::new();
    connector.enforce_http(true);
    connector.set_connect_timeout(Some(CONNECT_TIMEOUT));
    let mut client_builder = Client::builder(TokioExecutor::new());
    client_builder.retry_canceled_requests(false);
    client_builder.pool_timer(TokioTimer::new());
    client_builder.pool_idle_timeout(POOL_IDLE_TIMEOUT);
    client_builder.pool_max_idle_per_host(MAX_IDLE_PER_HOST);
    let client = client_builder.build(connector);
    Router::new()
        .route("/healthz", get(gateway_health))
        .route("/v1/health", get(proxy))
        .route("/v1/models", get(proxy))
        .route("/v1/models/:model", get(proxy))
        .route("/v1/completions", post(proxy))
        .route("/v1/chat/completions", post(proxy))
        .route("/v1/embeddings", post(proxy))
        .route("/v1/responses", post(proxy))
        .route("/v1/messages", post(proxy))
        .route("/v1/rerank", post(proxy))
        .route("/v1/reranking", post(proxy))
        // CORS preflight (`OPTIONS` with `Access-Control-Request-Method`) is
        // answered locally by this layer, before the request ever reaches a
        // route handler: it does not consume an admission permit, run
        // authentication, or contact the replica. Real cross-origin,
        // non-safelisted requests (for example `POST` with
        // `Content-Type: application/json`) get a complete preflight
        // response, not just `Access-Control-Allow-Origin`.
        .layer(CorsLayer::permissive())
        .with_state(GatewayState {
            upstream,
            client,
            admission: Arc::new(Semaphore::new(max_in_flight.get())),
            auth,
            audit,
        })
}

async fn gateway_health() -> StatusCode {
    StatusCode::NO_CONTENT
}

async fn proxy(State(state): State<GatewayState>, mut request: Request) -> Response {
    // Minted before anything can reject the request, so every audited outcome
    // (including authentication and admission rejections) carries a correlation
    // id. This is gateway-authoritative: any inbound
    // `x-camelid-request-id` is overwritten below, never trusted, exactly as
    // the untrusted forwarding headers are stripped.
    let request_id = mint_request_id();
    let method = request.method().to_string();
    // Path only, never the query string: the audit log records which route was
    // called, not caller-supplied query parameters.
    let path = request.uri().path().to_string();

    // Authentication runs before admission is checked. Otherwise an
    // unauthenticated flood would consume in-flight permits (and the SQLite
    // lookup time behind each one) meant for legitimate authenticated
    // traffic before ever being rejected; a request that fails auth must
    // never take a permit at all.
    let principal = match &state.auth {
        GatewayAuth::RequireToken(store) => match authenticate(store, request.headers()).await {
            Ok(principal) => Some(principal),
            Err(response) => return audited(&state, &request_id, None, &method, &path, response),
        },
        GatewayAuth::Disabled => None,
    };

    let permit = match Arc::clone(&state.admission).try_acquire_owned() {
        Ok(permit) => permit,
        Err(_) => {
            let mut response = gateway_error(
                StatusCode::SERVICE_UNAVAILABLE,
                "gateway concurrency limit reached",
            );
            let (low, high) = RETRY_AFTER_JITTER_SECONDS;
            let retry_after = fastrand::u8(low..=high);
            response.headers_mut().insert(
                "retry-after",
                HeaderValue::from_str(&retry_after.to_string())
                    .expect("a small decimal integer is a valid header value"),
            );
            return audited(&state, &request_id, principal.as_ref(), &method, &path, response);
        }
    };
    let upstream_uri = match state.upstream.request_uri(request.uri()) {
        Ok(uri) => uri,
        Err(error) => {
            tracing::error!(%error, "could not construct upstream request URI");
            return audited(
                &state,
                &request_id,
                principal.as_ref(),
                &method,
                &path,
                gateway_error(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "gateway could not construct the upstream request",
                ),
            );
        }
    };

    *request.uri_mut() = upstream_uri;
    remove_hop_by_hop(request.headers_mut());
    remove_untrusted_forwarding_headers(request.headers_mut());
    let host = HeaderValue::from_str(state.upstream.authority.as_str())
        .expect("a valid URI authority is a valid Host header");
    request.headers_mut().insert(HOST, host);
    // Stamp the gateway-authoritative correlation id after the untrusted
    // inbound headers have been stripped, so the replica receives exactly the
    // id this gateway audits. `insert` replaces any client-supplied value.
    request.headers_mut().insert(
        REQUEST_ID_HEADER,
        HeaderValue::from_str(&request_id).expect("a req_ hex id is a valid header value"),
    );

    let response = match state.client.request(request).await {
        Ok(response) => {
            let (mut parts, body) = response.into_parts();
            remove_hop_by_hop(&mut parts.headers);
            Response::from_parts(parts, Body::new(PermitBody::new(body, permit)))
        }
        Err(error) => {
            tracing::warn!(%error, "upstream replica request failed");
            gateway_error(StatusCode::BAD_GATEWAY, "upstream replica is unavailable")
        }
    };
    audited(&state, &request_id, principal.as_ref(), &method, &path, response)
}

/// Records the audit line for a handled request, when auditing is enabled, and
/// returns the response unchanged so it can wrap every return path in
/// [`proxy`].
///
/// The recorded `status` is the response *head* status, known as soon as the
/// upstream response head arrives; the streaming body is never buffered or
/// touched. A response whose head is `200` but whose stream then errors or is
/// cut short mid-generation is still recorded as `200`. This makes the audit
/// log a request-**initiation** and correlation record — not a
/// stream-completion or metering substrate: it cannot, on its own, distinguish
/// a full generation from one truncated after three tokens. Metering must not
/// be built on head status alone; that needs stream-completion accounting the
/// gateway does not do here.
fn audited(
    state: &GatewayState,
    request_id: &str,
    principal: Option<&PrincipalId>,
    method: &str,
    path: &str,
    response: Response,
) -> Response {
    if let Some(log) = &state.audit {
        write_audit_record(log, request_id, principal, method, path, response.status());
    }
    response
}

/// Serializes whole-record appends to the audit log within this process,
/// mirroring the replica's receipt writer so both append-only JSONL logs share
/// one safety model. `O_APPEND` makes a single `write` atomic, but `write_all`
/// can issue more than one `write` on a short write, and the audit line and the
/// receipt line are meant to be joined: a torn line loses the join at exactly
/// the concurrency burst where auditing matters most. Taking the whole-record
/// append under this lock removes that risk.
///
/// The gateway and the replica are separate processes writing separate files,
/// so this lock is deliberately per-process and independent of the replica's
/// receipt lock; there is no cross-process contention to coordinate.
fn audit_write_lock() -> &'static Mutex<()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
}

/// Appends one JSONL audit line: `{ts, request_id, principal, method, path,
/// status}`. `principal` is the resolved principal's opaque id, or JSON `null`
/// when the request was unauthenticated (auth disabled) or rejected before a
/// principal was established. `request_id` is the same id stamped on the
/// forwarded request, so this line joins to the replica's serving receipt.
///
/// Best-effort and off the request's async context: a failed write must never
/// fail the request. The write is not awaited, so audit lines still queued in
/// `spawn_blocking` when the process exits (for example on SIGTERM) are
/// dropped, matching the best-effort, no-durability contract the receipt writer
/// carries. The whole record (JSON plus newline) is written under
/// `audit_write_lock` in a single `write_all` so concurrent appends cannot
/// interleave into a corrupt line.
fn write_audit_record(
    log: &Arc<PathBuf>,
    request_id: &str,
    principal: Option<&PrincipalId>,
    method: &str,
    path: &str,
    status: StatusCode,
) {
    let line = serde_json::json!({
        "ts": std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs_f64())
            .unwrap_or(0.0),
        "request_id": request_id,
        "principal": principal.map(PrincipalId::as_str),
        "method": method,
        "path": path,
        "status": status.as_u16(),
    });
    let log = Arc::clone(log);
    tokio::task::spawn_blocking(move || {
        // Recover a poisoned lock instead of dropping every later audit line
        // for the rest of the process's life. Each holder appends exactly one
        // complete, newline-terminated record, so a panic while holding the
        // lock cannot leave a partial line behind; treating a poisoned guard as
        // usable is safe and keeps the audit trail alive after an unrelated
        // one-off failure.
        let _guard = audit_write_lock()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if let Ok(mut file) = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&*log)
        {
            let mut record = line.to_string();
            record.push('\n');
            let _ = file.write_all(record.as_bytes());
        }
    });
}

/// Mints an opaque, gateway-authoritative request correlation id. 128 bits of
/// randomness make a collision between two records probabilistically negligible
/// (a birthday bound around 2^64 mints), though not guaranteed unique, which is
/// worth recording if this id is ever promoted to a billing key. The id is
/// neither a secret nor a capability, so a fast, non-cryptographic source is
/// sufficient; `fastrand`'s per-thread PRNG supplies the value only, never an
/// ordering source.
fn mint_request_id() -> String {
    format!("req_{:016x}{:016x}", fastrand::u64(..), fastrand::u64(..))
}

pin_project! {
    struct PermitBody<B> {
        #[pin]
        inner: B,
        permit: Option<OwnedSemaphorePermit>,
    }
}

impl<B> PermitBody<B> {
    fn new(inner: B, permit: OwnedSemaphorePermit) -> Self
    where
        B: HttpBody,
    {
        let permit = (!inner.is_end_stream()).then_some(permit);
        Self { inner, permit }
    }
}

impl<B> HttpBody for PermitBody<B>
where
    B: HttpBody,
{
    type Data = B::Data;
    type Error = B::Error;

    fn poll_frame(
        self: Pin<&mut Self>,
        context: &mut Context<'_>,
    ) -> Poll<Option<Result<Frame<Self::Data>, Self::Error>>> {
        let mut this = self.project();
        let poll = this.inner.as_mut().poll_frame(context);
        if matches!(&poll, Poll::Ready(None) | Poll::Ready(Some(Err(_))))
            || this.inner.as_ref().is_end_stream()
        {
            this.permit.take();
        }
        poll
    }

    fn is_end_stream(&self) -> bool {
        self.inner.is_end_stream()
    }

    fn size_hint(&self) -> SizeHint {
        self.inner.size_hint()
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

fn remove_untrusted_forwarding_headers(headers: &mut HeaderMap) {
    for name in [
        "forwarded",
        "x-forwarded-for",
        "x-forwarded-host",
        "x-forwarded-port",
        "x-forwarded-proto",
        "x-real-ip",
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

/// Serves `router` on `listener` until `shutdown` resolves, then finishes
/// in-flight connections gracefully before returning.
///
/// Every accepted connection is force-closed after `max_connection_duration`
/// (see [`DEFAULT_MAX_CONNECTION_DURATION`]) regardless of activity, which is
/// what bounds how long a stalled client can pin an admission permit. This is
/// a per-connection wall-clock cap, not an idle timer: it fires even while a
/// connection is making steady, legitimate progress, so it must be sized to
/// comfortably exceed the slowest real generation this deployment serves.
pub async fn serve(
    listener: tokio::net::TcpListener,
    router: Router,
    max_connection_duration: Duration,
    shutdown: impl std::future::Future<Output = ()> + Send + 'static,
) -> std::io::Result<()> {
    let graceful = hyper_util::server::graceful::GracefulShutdown::new();
    let mut shutdown = std::pin::pin!(shutdown);
    loop {
        tokio::select! {
            accepted = listener.accept() => {
                let (stream, _peer_addr) = match accepted {
                    Ok(pair) => pair,
                    Err(error) => {
                        tracing::warn!(%error, "failed to accept a gateway connection");
                        continue;
                    }
                };
                let io = hyper_util::rt::TokioIo::new(stream);
                let service = hyper_util::service::TowerToHyperService::new(router.clone());
                let connection = hyper::server::conn::http1::Builder::new().serve_connection(io, service);
                let watched = graceful.watch(connection);
                tokio::spawn(async move {
                    if tokio::time::timeout(max_connection_duration, watched)
                        .await
                        .is_err()
                    {
                        tracing::warn!(
                            seconds = max_connection_duration.as_secs(),
                            "gateway connection exceeded the maximum duration and was closed",
                        );
                    }
                });
            }
            _ = &mut shutdown => {
                break;
            }
        }
    }
    drop(listener);
    graceful.shutdown().await;
    Ok(())
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
