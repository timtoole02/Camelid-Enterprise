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
use axum::routing::{get, MethodRouter};
use axum::{Extension, Json, Router};
use bytes::Buf;
use http_body::{Body as HttpBody, Frame, SizeHint};
use hyper_util::client::legacy::connect::HttpConnector;
use hyper_util::client::legacy::Client;
use hyper_util::rt::{TokioExecutor, TokioTimer};
use identity::{AuthenticatedContext, IdentityError, OrganizationId, SqliteIdentityStore};
use pin_project_lite::pin_project;
use replica_contract::HttpMethod;
use std::collections::HashMap;
use std::fmt;
use std::io::{self, Write};
use std::num::{NonZeroU32, NonZeroU64, NonZeroUsize};
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicU8, Ordering};
use std::sync::mpsc::{sync_channel, Receiver, SyncSender, TrySendError};
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll};
use std::time::Duration;
use std::time::Instant;
use tokio::sync::{OwnedSemaphorePermit, Semaphore};
use tower_http::cors::CorsLayer;

const CONNECT_TIMEOUT: Duration = Duration::from_secs(5);
const POOL_IDLE_TIMEOUT: Duration = Duration::from_secs(30);
const MAX_IDLE_PER_HOST: usize = 32;
const MAX_PENDING_LOG_RECORDS: usize = 1_024;
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
const CONNECTION_ACTIVE: u8 = 0;
const CONNECTION_TIMED_OUT: u8 = 1;

/// Shared by every request on one HTTP connection. It is set before the
/// server's wall-clock timeout drops the connection future, so a response-body
/// wrapper can distinguish that gateway policy from an otherwise unexplained
/// incomplete stream.
struct ConnectionTermination {
    reason: AtomicU8,
}

impl ConnectionTermination {
    fn gateway_timeout(&self) {
        self.reason.store(CONNECTION_TIMED_OUT, Ordering::Release);
    }

    fn incomplete_outcome(&self) -> &'static str {
        match self.reason.load(Ordering::Acquire) {
            CONNECTION_TIMED_OUT => "gateway_timeout",
            CONNECTION_ACTIVE => "incomplete",
            _ => "incomplete",
        }
    }
}

impl Default for ConnectionTermination {
    fn default() -> Self {
        Self {
            reason: AtomicU8::new(CONNECTION_ACTIVE),
        }
    }
}

/// A bounded, single-writer append-only JSONL sink. Opening the destination is
/// part of startup, so a missing parent or unwritable path fails closed before
/// the gateway accepts traffic. Runtime writes never block a request task: a
/// full queue drops the record and reports a rate-limited warning instead of
/// growing an unbounded backlog on Tokio's blocking pool.
pub struct GatewayLog {
    destination: PathBuf,
    sender: SyncSender<String>,
    dropped_records: AtomicU64,
}

impl GatewayLog {
    pub fn open(path: impl AsRef<Path>) -> io::Result<Arc<Self>> {
        let file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path.as_ref())?;
        let destination = std::fs::canonicalize(path.as_ref())?;
        let (sender, receiver) = sync_channel(MAX_PENDING_LOG_RECORDS);
        let writer_destination = destination.clone();
        std::thread::Builder::new()
            .name("camelid-gateway-jsonl".into())
            .spawn(move || write_jsonl_records(file, receiver, writer_destination))?;
        Ok(Arc::new(Self {
            destination,
            sender,
            dropped_records: AtomicU64::new(0),
        }))
    }

    pub fn path(&self) -> &Path {
        &self.destination
    }

    pub fn has_same_destination(&self, other: &Self) -> bool {
        self.destination == other.destination
    }

    fn write(&self, line: String) {
        match self.sender.try_send(line) {
            Ok(()) => {}
            Err(TrySendError::Full(_)) => self.record_drop("writer queue is full"),
            Err(TrySendError::Disconnected(_)) => self.record_drop("writer thread stopped"),
        }
    }

    fn record_drop(&self, reason: &'static str) {
        let dropped = self.dropped_records.fetch_add(1, Ordering::Relaxed) + 1;
        if dropped == 1 || dropped.is_power_of_two() {
            tracing::warn!(
                path = %self.destination.display(),
                dropped_records = dropped,
                reason,
                "gateway JSONL record dropped"
            );
        }
    }
}

fn write_jsonl_records(mut file: std::fs::File, records: Receiver<String>, destination: PathBuf) {
    let mut failed_writes = 0_u64;
    for line in records {
        if let Err(error) = file.write_all(line.as_bytes()) {
            failed_writes += 1;
            if failed_writes == 1 || failed_writes.is_power_of_two() {
                tracing::warn!(
                    path = %destination.display(),
                    failed_writes,
                    %error,
                    "gateway JSONL writer could not append a record"
                );
            }
        } else {
            failed_writes = 0;
        }
    }
}

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
    /// line per request it handles — `{ts, request_id, principal, organization,
    /// method, path, status}` — including requests it rejects for authentication or
    /// admission. `None` disables auditing entirely. CORS preflight and
    /// `/healthz` are answered before this path and are never audited.
    audit: Option<Arc<GatewayLog>>,
}

/// Per-organization request-rate quota, independent of admission control.
/// Admission ([`DEFAULT_MAX_IN_FLIGHT`]) bounds how much concurrent work the
/// gateway does in total; [`OrgQuota`] bounds how much of that shared budget
/// one organization may consume in a fixed window, so one oversubscribed or
/// misbehaving tenant cannot starve every other tenant's share of it.
///
/// State is in-memory and per-process: it resets on restart and is not
/// shared across gateway replicas behind the same Service. That is
/// consistent with this gateway's other non-durable state (the request audit
/// log): a fixed-window approximation is enough to stop one tenant from
/// monopolizing shared capacity, it is not a durable metering or billing
/// substrate. The quota is charged after successful authentication but before
/// admission and forwarding, so a request that later receives a gateway `503`
/// or an upstream `502` still counts: this is an anti-starvation control for
/// gateway work, not a record of successful inference.
///
/// Authentication necessarily resolves the bearer token in SQLite before the
/// gateway knows which organization to charge. Consequently, this quota does
/// not bound per-request identity-store work from an over-budget valid token.
/// A token cache would change token-revocation semantics, so it needs its own
/// bounded, invalidation-aware design rather than an implicit fast path here.
pub struct OrgQuota {
    limit: NonZeroU32,
    window: Duration,
    windows: Mutex<HashMap<String, OrgWindow>>,
}

struct OrgWindow {
    started_at: Instant,
    count: u32,
}

impl OrgQuota {
    /// Allows at most `limit` requests per organization in every fixed window
    /// of `window_seconds`. A nonzero type makes a configuration that resets
    /// on every request impossible.
    pub fn new(limit: NonZeroU32, window_seconds: NonZeroU64) -> Self {
        Self {
            limit,
            window: Duration::from_secs(window_seconds.get()),
            windows: Mutex::new(HashMap::new()),
        }
    }

    /// Admits one request for `organization`, incrementing its window count,
    /// or rejects it with the remaining time until that organization's
    /// window resets.
    ///
    /// Fixed-window, not sliding or token-bucket: a burst can land up to
    /// `limit` requests at the end of one window and another `limit`
    /// immediately after it resets, so the true worst case across a window
    /// boundary is under `2 * limit` requests in a short span, not exactly
    /// `limit`. That trade-off is intentional: admission is a single hash-map
    /// lookup and increment under one process-wide lock, with no per-request
    /// timer queue. At the gateway's default 256 in-flight requests this
    /// short critical section is deliberate; revisit sharding only if real
    /// traffic proves it contended.
    fn admit(&self, organization: &OrganizationId) -> Result<(), Duration> {
        let now = Instant::now();
        let mut windows = self
            .windows
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if let Some(entry) = windows.get_mut(organization.as_str()) {
            return self.admit_window(entry, now);
        }

        // A key is created only for a previously unseen organization. Sweep
        // expired windows on that slow path so deleted or long-idle tenants do
        // not remain resident forever, without adding work to steady-state
        // authenticated requests.
        windows.retain(|_, entry| now.duration_since(entry.started_at) < self.window);
        let entry = windows
            .entry(organization.as_str().to_string())
            .or_insert(OrgWindow {
                started_at: now,
                count: 0,
            });
        self.admit_window(entry, now)
    }

    fn admit_window(&self, entry: &mut OrgWindow, now: Instant) -> Result<(), Duration> {
        if now.duration_since(entry.started_at) >= self.window {
            entry.started_at = now;
            entry.count = 0;
        }
        if entry.count >= self.limit.get() {
            let elapsed = now.duration_since(entry.started_at);
            return Err(self.window.saturating_sub(elapsed));
        }
        entry.count += 1;
        Ok(())
    }
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
    RequireToken {
        store: Arc<SqliteIdentityStore>,
        /// A quota exists only inside the authenticated mode because it needs
        /// the authenticated organization as its key. `None` retains the
        /// original authenticated-but-unlimited behavior.
        quota: Option<Arc<OrgQuota>>,
        /// Optional append-only JSONL terminal transport-accounting sink. It
        /// belongs to the authenticated mode so every usage record carries a
        /// resolved principal and organization; it cannot be configured for
        /// anonymous traffic that a later aggregator could not attribute.
        usage: Option<Arc<GatewayLog>>,
    },
}

pub fn router(upstream: UpstreamOrigin) -> Router {
    router_with_max_in_flight(upstream, DEFAULT_MAX_IN_FLIGHT)
}

pub fn router_with_max_in_flight(upstream: UpstreamOrigin, max_in_flight: NonZeroUsize) -> Router {
    router_with_options(upstream, max_in_flight, GatewayAuth::Disabled, None)
}

// The forwarded route surface is derived from `replica_contract::PUBLIC_ROUTES`,
// the single machine-readable source of truth for the replica's public HTTP
// contract, so the gateway's allowlist cannot silently drift from it. Adding,
// removing, or renaming a public route in the contract crate changes exactly
// what this gateway forwards, with no hand-maintained mirror to keep in sync.
// Private replica routes (model, runtime, workspace, legacy, diagnostics) are
// absent from `PUBLIC_ROUTES` by construction, so they are never forwarded.
pub fn router_with_options(
    upstream: UpstreamOrigin,
    max_in_flight: NonZeroUsize,
    auth: GatewayAuth,
    audit: Option<Arc<GatewayLog>>,
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
    // `/healthz` is the gateway's own liveness probe, not part of the replica
    // contract, so it is registered explicitly and answered locally.
    let mut router = Router::new().route("/healthz", get(gateway_health));
    for spec in replica_contract::contractual_routes() {
        router = router.route(spec.path, proxy_methods(spec.methods));
    }
    router
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

/// Builds the [`MethodRouter`] for one contractual route: every HTTP method the
/// contract declares for it is forwarded to [`proxy`], and any other method
/// falls through to axum's automatic `405 Method Not Allowed`. Matching
/// [`HttpMethod`] exhaustively means a new method variant added to the contract
/// crate stops this crate from compiling until the gateway is taught to forward
/// it, rather than silently dropping a newly-contractual method.
fn proxy_methods(methods: &[HttpMethod]) -> MethodRouter<GatewayState> {
    let mut method_router = MethodRouter::new();
    for method in methods {
        method_router = match method {
            HttpMethod::Get => method_router.get(proxy),
            HttpMethod::Post => method_router.post(proxy),
            HttpMethod::Delete => method_router.delete(proxy),
        };
    }
    method_router
}

async fn proxy(
    State(state): State<GatewayState>,
    connection_termination: Option<Extension<Arc<ConnectionTermination>>>,
    mut request: Request,
) -> Response {
    let connection_termination = connection_termination.map(|Extension(marker)| marker);
    let started_ts = unix_timestamp();
    let started_at = Instant::now();
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
    let (identity, usage) = match &state.auth {
        GatewayAuth::RequireToken {
            store,
            quota,
            usage,
        } => {
            let identity = match authenticate(store, request.headers()).await {
                Ok(context) => context,
                Err(response) => {
                    return audited(
                        &state,
                        AuditedRequest {
                            request_id: &request_id,
                            identity: None,
                            method: &method,
                            path: &path,
                            usage: None,
                            request_metrics: None,
                            connection_termination: connection_termination.clone(),
                            started_ts,
                            started_at,
                        },
                        response,
                    )
                }
            };

            // Quota is checked after authentication (it needs a resolved
            // organization to charge) but before the admission permit is
            // acquired. A request the gateway is about to reject must never
            // first take a permit meant for traffic that will be forwarded.
            if let Some(quota) = quota {
                if let Err(retry_after) = quota.admit(identity.organization_id()) {
                    let response = quota_exceeded(retry_after);
                    return audited(
                        &state,
                        AuditedRequest {
                            request_id: &request_id,
                            identity: Some(&identity),
                            method: &method,
                            path: &path,
                            usage: None,
                            request_metrics: None,
                            connection_termination: connection_termination.clone(),
                            started_ts,
                            started_at,
                        },
                        response,
                    );
                }
            }
            (Some(identity), usage.clone())
        }
        GatewayAuth::Disabled => (None, None),
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
            return audited(
                &state,
                AuditedRequest {
                    request_id: &request_id,
                    identity: identity.as_ref(),
                    method: &method,
                    path: &path,
                    usage: usage.clone(),
                    request_metrics: None,
                    connection_termination: connection_termination.clone(),
                    started_ts,
                    started_at,
                },
                response,
            );
        }
    };
    let upstream_uri = match state.upstream.request_uri(request.uri()) {
        Ok(uri) => uri,
        Err(error) => {
            tracing::error!(%error, "could not construct upstream request URI");
            return audited(
                &state,
                AuditedRequest {
                    request_id: &request_id,
                    identity: identity.as_ref(),
                    method: &method,
                    path: &path,
                    usage: usage.clone(),
                    request_metrics: None,
                    connection_termination: connection_termination.clone(),
                    started_ts,
                    started_at,
                },
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

    // A byte count is meaningful only when the gateway actually attempts to
    // forward the request and observes its body reach EOF. Admission and URI
    // rejections intentionally leave this `None`: draining a rejected body
    // would consume unbounded work solely to manufacture an accounting value.
    let request_metrics = usage
        .as_ref()
        .map(|_| Arc::new(RequestBodyMetrics::default()));
    if let Some(metrics) = &request_metrics {
        let (parts, body) = request.into_parts();
        request = Request::from_parts(
            parts,
            Body::new(CountingBody::new(body, Arc::clone(metrics))),
        );
    }

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
    audited(
        &state,
        AuditedRequest {
            request_id: &request_id,
            identity: identity.as_ref(),
            method: &method,
            path: &path,
            usage,
            request_metrics,
            connection_termination,
            started_ts,
            started_at,
        },
        response,
    )
}

/// Records the audit line for a handled request, when auditing is enabled, and
/// wraps its response in terminal usage accounting when configured. It is the
/// single funnel for every return path in [`proxy`].
///
/// The recorded `status` is the response *head* status, known as soon as the
/// upstream response head arrives; the streaming body is never buffered or
/// touched. A response whose head is `200` but whose stream then errors or is
/// cut short mid-generation is still recorded as `200`. This makes the audit
/// log a request-**initiation** and correlation record — not a
/// stream-completion or metering substrate: it cannot, on its own, distinguish
/// a full generation from one truncated after three tokens. The optional usage
/// sink below supplies that separate terminal transport-accounting record.
struct AuditedRequest<'a> {
    request_id: &'a str,
    identity: Option<&'a AuthenticatedContext>,
    method: &'a str,
    path: &'a str,
    usage: Option<Arc<GatewayLog>>,
    request_metrics: Option<Arc<RequestBodyMetrics>>,
    connection_termination: Option<Arc<ConnectionTermination>>,
    started_ts: f64,
    started_at: Instant,
}

fn audited(state: &GatewayState, request: AuditedRequest<'_>, response: Response) -> Response {
    if let Some(log) = &state.audit {
        write_audit_record(
            log,
            request.request_id,
            request.identity,
            request.method,
            request.path,
            response.status(),
        );
    }
    let Some((log, identity)) = request.usage.zip(request.identity) else {
        return response;
    };
    let (parts, body) = response.into_parts();
    let tracker = UsageTracker::new(UsageRecord {
        log,
        request_id: request.request_id.to_string(),
        principal: identity.principal_id().as_str().to_string(),
        organization: identity.organization_id().as_str().to_string(),
        method: request.method.to_string(),
        path: request.path.to_string(),
        response_head_status: parts.status.as_u16(),
        request_metrics: request.request_metrics,
        response_bytes: 0,
        started_ts: request.started_ts,
        started_at: request.started_at,
        connection_termination: request.connection_termination,
    });
    Response::from_parts(parts, Body::new(UsageBody::new(body, tracker)))
}

/// Appends one JSONL audit line: `{ts, request_id, principal, organization,
/// method, path, status}`. `principal` and `organization` are opaque gateway-
/// local identity fields, or JSON `null` when authentication is disabled or
/// rejected before identity was established. They are never forwarded to a
/// replica. `request_id` is the only correlation value sent upstream.
///
/// Best-effort and off the request's async context: a failed write must never
/// fail the request. The dedicated bounded writer queue can lose records on
/// process exit, when full, or after a write failure; [`GatewayLog`] emits
/// rate-limited warnings for each condition. Its single writer serializes
/// complete JSONL records so concurrent requests cannot interleave lines.
fn write_audit_record(
    log: &GatewayLog,
    request_id: &str,
    identity: Option<&AuthenticatedContext>,
    method: &str,
    path: &str,
    status: StatusCode,
) {
    let line = serde_json::json!({
        "ts": unix_timestamp(),
        "request_id": request_id,
        "principal": identity.map(|context| context.principal_id().as_str()),
        "organization": identity.map(|context| context.organization_id().as_str()),
        "method": method,
        "path": path,
        "status": status.as_u16(),
    });
    log.write(jsonl_line(line));
}

struct UsageRecord {
    log: Arc<GatewayLog>,
    request_id: String,
    principal: String,
    organization: String,
    method: String,
    path: String,
    response_head_status: u16,
    request_metrics: Option<Arc<RequestBodyMetrics>>,
    response_bytes: u64,
    started_ts: f64,
    started_at: Instant,
    connection_termination: Option<Arc<ConnectionTermination>>,
}

/// Owns one terminal usage record until a response body finishes, errors, or is
/// dropped before its terminal frame. A dropped body has no single observable
/// cause: it can be a client disconnect, the gateway connection deadline, or a
/// transport cancellation. Its `Drop` implementation therefore records the
/// fact as `incomplete` rather than attributing fault to a peer.
struct UsageTracker {
    record: Option<UsageRecord>,
}

impl UsageTracker {
    fn new(record: UsageRecord) -> Self {
        Self {
            record: Some(record),
        }
    }

    fn add_response_bytes(&mut self, bytes: u64) {
        if let Some(record) = &mut self.record {
            record.response_bytes = record.response_bytes.saturating_add(bytes);
        }
    }

    fn finish(&mut self, outcome: &'static str) {
        if let Some(record) = self.record.take() {
            write_usage_record(record, outcome);
        }
    }
}

impl Drop for UsageTracker {
    fn drop(&mut self) {
        let outcome = self
            .record
            .as_ref()
            .and_then(|record| record.connection_termination.as_ref())
            .map_or("incomplete", |marker| marker.incomplete_outcome());
        self.finish(outcome);
    }
}

/// Appends one terminal JSONL usage record:
/// `{ts, started_ts, duration_ms, request_id, principal, organization, method,
/// path, response_head_status, request_bytes, response_bytes, stream_outcome}`.
/// `request_bytes` is `null` unless the gateway observed the forwarded request
/// body reach EOF. Byte counts are raw payload bytes the gateway read or
/// forwarded, never tokenizer counts, so this record must not be used as an
/// inference token-billing source.
fn write_usage_record(record: UsageRecord, outcome: &'static str) {
    let line = serde_json::json!({
        "ts": unix_timestamp(),
        "started_ts": record.started_ts,
        "duration_ms": record.started_at.elapsed().as_millis().min(u128::from(u64::MAX)) as u64,
        "request_id": record.request_id,
        "principal": record.principal,
        "organization": record.organization,
        "method": record.method,
        "path": record.path,
        "response_head_status": record.response_head_status,
        "request_bytes": record
            .request_metrics
            .as_deref()
            .and_then(RequestBodyMetrics::completed_bytes),
        "response_bytes": record.response_bytes,
        "stream_outcome": outcome,
    });
    record.log.write(jsonl_line(line));
}

fn unix_timestamp() -> f64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| duration.as_secs_f64())
        .unwrap_or(0.0)
}

fn jsonl_line(value: serde_json::Value) -> String {
    let mut line = value.to_string();
    line.push('\n');
    line
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
    struct CountingBody<B> {
        #[pin]
        inner: B,
        metrics: Arc<RequestBodyMetrics>,
    }
}

#[derive(Default)]
struct RequestBodyMetrics {
    bytes: AtomicU64,
    completed: AtomicBool,
}

impl RequestBodyMetrics {
    fn completed_bytes(&self) -> Option<u64> {
        self.completed
            .load(Ordering::Acquire)
            .then(|| self.bytes.load(Ordering::Acquire))
    }

    fn complete(&self) {
        self.completed.store(true, Ordering::Release);
    }
}

impl<B> CountingBody<B>
where
    B: HttpBody,
{
    fn new(inner: B, metrics: Arc<RequestBodyMetrics>) -> Self {
        if inner.is_end_stream() {
            metrics.complete();
        }
        Self { inner, metrics }
    }
}

impl<B> HttpBody for CountingBody<B>
where
    B: HttpBody,
    B::Data: Buf,
{
    type Data = B::Data;
    type Error = B::Error;

    fn poll_frame(
        self: Pin<&mut Self>,
        context: &mut Context<'_>,
    ) -> Poll<Option<Result<Frame<Self::Data>, Self::Error>>> {
        let mut this = self.project();
        let poll = this.inner.as_mut().poll_frame(context);
        if let Poll::Ready(Some(Ok(frame))) = &poll {
            if let Some(data) = frame.data_ref() {
                this.metrics
                    .bytes
                    .fetch_add(data.remaining() as u64, Ordering::Relaxed);
            }
        }
        if matches!(&poll, Poll::Ready(None)) || this.inner.as_ref().is_end_stream() {
            this.metrics.complete();
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

pin_project! {
    struct UsageBody<B> {
        #[pin]
        inner: B,
        tracker: UsageTracker,
    }
}

impl<B> UsageBody<B>
where
    B: HttpBody,
{
    fn new(inner: B, mut tracker: UsageTracker) -> Self {
        if inner.is_end_stream() {
            tracker.finish("completed");
        }
        Self { inner, tracker }
    }
}

impl<B> HttpBody for UsageBody<B>
where
    B: HttpBody,
    B::Data: Buf,
{
    type Data = B::Data;
    type Error = B::Error;

    fn poll_frame(
        self: Pin<&mut Self>,
        context: &mut Context<'_>,
    ) -> Poll<Option<Result<Frame<Self::Data>, Self::Error>>> {
        let mut this = self.project();
        let poll = this.inner.as_mut().poll_frame(context);
        match &poll {
            Poll::Ready(Some(Ok(frame))) => {
                if let Some(data) = frame.data_ref() {
                    this.tracker.add_response_bytes(data.remaining() as u64);
                }
            }
            Poll::Ready(Some(Err(_))) => this.tracker.finish("body_error"),
            Poll::Ready(None) => this.tracker.finish("completed"),
            Poll::Pending => {}
        }
        if this.inner.as_ref().is_end_stream() {
            this.tracker.finish("completed");
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

/// Resolves the request's bearer token to its principal and organization, or
/// returns the exact response the gateway should send without forwarding.
async fn authenticate(
    store: &Arc<SqliteIdentityStore>,
    headers: &HeaderMap,
) -> Result<AuthenticatedContext, Response> {
    let Some(token) = bearer_token(headers) else {
        return Err(unauthorized("missing bearer token"));
    };
    let store = Arc::clone(store);
    let token = token.to_string();
    let resolved = tokio::task::spawn_blocking(move || store.resolve_context(&token))
        .await
        .expect("identity resolution task panicked");

    resolved.map_err(|error| match error {
        IdentityError::InvalidToken => unauthorized("invalid bearer token"),
        // Distinct from the above so a client that is behaving correctly can
        // tell "present a new credential" from "your credential is wrong".
        // Both are 401: an expired token confers nothing, and the difference
        // is a reason, not a policy.
        IdentityError::ExpiredToken => unauthorized("expired bearer token"),
        error => {
            tracing::error!(%error, "identity store error while authenticating request");
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

/// A typed `429` for a request whose organization has exhausted its request
/// quota for the current window. `retry_after` is the gateway's own remaining
/// time until that organization's window resets; it is rounded up to whole
/// seconds and floored at one, since `Retry-After` cannot express sub-second
/// or zero delays meaningfully.
fn quota_exceeded(retry_after: Duration) -> Response {
    let mut response = (
        StatusCode::TOO_MANY_REQUESTS,
        Json(serde_json::json!({
            "error": {
                "message": "organization request quota exceeded",
                "type": "quota_exceeded"
            }
        })),
    )
        .into_response();
    response
        .headers_mut()
        .insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
    let retry_after_seconds = retry_after_seconds(retry_after);
    response.headers_mut().insert(
        "retry-after",
        HeaderValue::from_str(&retry_after_seconds.to_string())
            .expect("a small decimal integer is a valid header value"),
    );
    response
}

fn retry_after_seconds(retry_after: Duration) -> u64 {
    retry_after
        .as_secs()
        .saturating_add(u64::from(retry_after.subsec_nanos() > 0))
        .max(1)
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
                let termination = Arc::new(ConnectionTermination::default());
                let service = hyper_util::service::TowerToHyperService::new(
                    router
                        .clone()
                        .layer(Extension(Arc::clone(&termination))),
                );
                let connection = hyper::server::conn::http1::Builder::new().serve_connection(io, service);
                let watched = graceful.watch(connection);
                tokio::spawn(async move {
                    tokio::pin!(watched);
                    tokio::select! {
                        _ = &mut watched => {}
                        _ = tokio::time::sleep(max_connection_duration) => {
                            // Set the cause before dropping `watched`; its cancellation
                            // drops active response wrappers synchronously.
                            termination.gateway_timeout();
                            tracing::warn!(
                                seconds = max_connection_duration.as_secs(),
                                "gateway connection exceeded the maximum duration and was closed",
                            );
                        }
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

    #[test]
    fn org_quota_admits_up_to_the_limit_then_rejects() {
        let quota = OrgQuota::new(NonZeroU32::new(2).unwrap(), NonZeroU64::new(60).unwrap());
        let organization = OrganizationId::new("org_acme".to_string());
        assert!(quota.admit(&organization).is_ok());
        assert!(quota.admit(&organization).is_ok());
        let retry_after = quota.admit(&organization).unwrap_err();
        assert!(retry_after > Duration::ZERO && retry_after <= Duration::from_secs(60));
    }

    #[test]
    fn org_quota_tracks_organizations_independently() {
        let quota = OrgQuota::new(NonZeroU32::new(1).unwrap(), NonZeroU64::new(60).unwrap());
        let acme = OrganizationId::new("org_acme".to_string());
        let globex = OrganizationId::new("org_globex".to_string());
        assert!(quota.admit(&acme).is_ok());
        assert!(quota.admit(&acme).is_err());
        // A different organization has its own, unaffected budget.
        assert!(quota.admit(&globex).is_ok());
    }

    #[test]
    fn org_quota_resets_once_the_window_elapses() {
        let quota = OrgQuota::new(NonZeroU32::new(1).unwrap(), NonZeroU64::new(1).unwrap());
        let organization = OrganizationId::new("org_acme".to_string());
        assert!(quota.admit(&organization).is_ok());
        assert!(quota.admit(&organization).is_err());
        std::thread::sleep(Duration::from_millis(1_100));
        assert!(quota.admit(&organization).is_ok());
    }

    #[test]
    fn retry_after_seconds_rounds_up_and_never_returns_zero() {
        assert_eq!(retry_after_seconds(Duration::ZERO), 1);
        assert_eq!(retry_after_seconds(Duration::from_millis(1)), 1);
        assert_eq!(retry_after_seconds(Duration::from_secs(1)), 1);
        assert_eq!(retry_after_seconds(Duration::new(59, 999_999_999)), 60);
    }
}
