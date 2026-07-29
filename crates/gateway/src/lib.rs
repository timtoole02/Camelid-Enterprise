//! Transparent HTTP forwarding for Camelid Enterprise inference replicas.
//!
//! The gateway treats request and response bodies as opaque streams. It does
//! not retry, inspect, buffer, or rewrite inference traffic; the replica remains
//! the authority for output and attribution.

use axum::body::{to_bytes, Body};
use axum::extract::{Path as AxumPath, Request, State};
use axum::http::header::{
    AUTHORIZATION, CONNECTION, CONTENT_LENGTH, CONTENT_TYPE, HOST, WWW_AUTHENTICATE,
};
use axum::http::uri::{Authority, Scheme};
use axum::http::{HeaderMap, HeaderName, HeaderValue, Method, StatusCode, Uri};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, MethodRouter};
use axum::{Extension, Json, Router};
use bytes::Buf;
use http_body::{Body as HttpBody, Frame, SizeHint};
use http_body_util::{BodyExt as _, LengthLimitError, Limited};
use hyper_util::client::legacy::connect::HttpConnector;
use hyper_util::client::legacy::Client;
use hyper_util::rt::{TokioExecutor, TokioTimer};
use identity::{AuthenticatedContext, IdentityError, OrganizationId, SqliteIdentityStore};
use pin_project_lite::pin_project;
use replica_contract::{HttpMethod, RouteSpec};
use serde::Deserialize;
use std::borrow::Cow;
use std::collections::{BTreeMap, HashMap};
use std::fmt;
use std::io::{self, Write};
use std::num::{NonZeroU32, NonZeroU64, NonZeroUsize};
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicU8, Ordering};
use std::sync::mpsc::{sync_channel, Receiver, SyncSender, TrySendError};
use std::sync::{Arc, Mutex, MutexGuard};
use std::task::{Context, Poll};
use std::time::Duration;
use std::time::Instant;
use tokio::sync::{OwnedSemaphorePermit, Semaphore};
use tower_http::cors::CorsLayer;

const CONNECT_TIMEOUT: Duration = Duration::from_secs(5);
const POOL_IDLE_TIMEOUT: Duration = Duration::from_secs(30);
const MAX_IDLE_PER_HOST: usize = 32;
const MAX_PENDING_LOG_RECORDS: usize = 1_024;
const CATALOG_DISCOVERY_TIMEOUT: Duration = Duration::from_secs(5);
const MAX_CATALOG_DISCOVERY_BODY_BYTES: usize = 1024 * 1024;
/// The largest JSON generation request catalog mode will materialize to select
/// its model. This is an explicit bound: without it, a caller could make the
/// gateway allocate arbitrarily before it knows which replica pool to use.
pub const DEFAULT_MAX_MODEL_SELECTION_BODY_BYTES: NonZeroUsize =
    NonZeroUsize::new(2 * 1024 * 1024).unwrap();
/// Total bytes catalog mode reserves for concurrently materialized selector
/// bodies. The selector semaphore capacity is
/// `memory_budget / max_body_bytes`, so the gateway never has more bounded
/// selector bodies in memory than this declared budget allows.
pub const DEFAULT_MODEL_SELECTION_MEMORY_BUDGET_BYTES: NonZeroUsize =
    NonZeroUsize::new(32 * 1024 * 1024).unwrap();
/// How long a request may wait for selector capacity before the gateway sheds
/// it with a typed `503`.
///
/// Waiting rather than refusing on contact is the whole point. Selector
/// capacity is a *memory* reservation held only while a request body is being
/// read, which for an ordinary generation request is milliseconds. Refusing
/// immediately turns every momentary overlap between two valid requests into a
/// failed request: measured on loopback with the previous fail-fast admission,
/// sixteen concurrent 64 KiB requests from one organization lost three
/// quarters of the burst, and sixty-four requests spread across sixty-four
/// distinct organizations lost 84% of them to the global bound alone. A
/// waiter holds no body memory, so queueing briefly costs nothing the bound
/// exists to protect.
///
/// It is still bounded: past this the gateway is genuinely out of selector
/// memory for long enough that shedding load is the honest answer.
pub const DEFAULT_MODEL_SELECTION_ACQUIRE_TIMEOUT: Duration = Duration::from_secs(5);
/// How long a request that holds selector capacity may take to deliver its
/// body before the gateway gives up on it.
///
/// This is what makes waiting for capacity safe. Without it the only bound on
/// how long one slot can be occupied is [`DEFAULT_MAX_CONNECTION_DURATION`] --
/// a client that dribbles a body could hold a selector slot for five minutes,
/// so the global budget would be exhaustible by a handful of slow clients and
/// every waiter behind them would time out. A request that cannot deliver at
/// most `--max-model-selection-body-bytes` within this window is not a
/// generation request the deployment can serve anyway.
pub const DEFAULT_MODEL_SELECTION_READ_TIMEOUT: Duration = Duration::from_secs(15);
/// How long shutdown waits for each JSONL log to drain what it already
/// accepted. Deliberately short relative to a termination grace period: the
/// point is to save the queue in the ordinary case, not to let a stuck
/// filesystem hold the process open until the orchestrator kills it, which
/// would lose the records anyway and delay everything else on the way out.
pub const DEFAULT_LOG_FLUSH_DEADLINE: Duration = Duration::from_secs(5);
/// Retry-After is jittered across this inclusive range (seconds) so that
/// clients rejected at the same instant do not retry in lockstep and
/// re-create the exact saturation spike that rejected them.
const RETRY_AFTER_JITTER_SECONDS: (u8, u8) = (1, 3);
pub const DEFAULT_MAX_IN_FLIGHT: NonZeroUsize = NonZeroUsize::new(256).unwrap();
/// Every accepted client connection is force-closed after this long,
/// regardless of activity. Nothing else in this module bounds how long a
/// single HTTP exchange may run: the admission permit in `PermitBody` is
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
///
/// The queue is drained by a dedicated thread, so a record that has been
/// accepted is not yet a record that has been written. [`Self::flush_and_stop`]
/// closes that gap at shutdown; without it, everything still queued when the
/// process exits is lost, which on Kubernetes is every rolling update rather
/// than an edge case.
pub struct GatewayLog {
    destination: PathBuf,
    /// The record sender and the writer's completion signal, behind one lock.
    ///
    /// One lock rather than two on purpose: shutdown has to take both halves
    /// atomically. Split across separate mutexes, two concurrent flushes can
    /// interleave so that each takes one half, and then neither waits for the
    /// drain it just triggered.
    ///
    /// This costs one uncontended mutex acquisition per record. The bounded
    /// channel behind it already takes an internal lock on every send, so this
    /// adds a second lock of the same order to a path that is already far
    /// cheaper than the request it accompanies.
    channel: Mutex<LogChannel>,
    queued_records: AtomicU64,
    written_records: Arc<AtomicU64>,
    dropped_records: AtomicU64,
}

/// Both halves are `None` once [`GatewayLog::flush_and_stop`] has taken them.
/// Dropping the sender is what tells the writer thread to drain and exit, and
/// holding it here means shutdown does not depend on every `Arc` clone of the
/// log having been released first.
struct LogChannel {
    records: Option<SyncSender<String>>,
    /// A channel rather than a `JoinHandle` because joining a thread cannot be
    /// given a deadline, and shutdown must stay bounded.
    writer_stopped: Option<Receiver<()>>,
}

/// What [`GatewayLog::flush_and_stop`] managed to do.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LogFlush {
    /// The writer drained every accepted record and stopped.
    Drained,
    /// The deadline expired first. `pending_at_deadline` records had been
    /// accepted but not yet written when the wait gave up.
    ///
    /// Deliberately not called "lost". Giving up on the wait does not stop the
    /// writer: it keeps draining on its own, so some or all of these may still
    /// reach the file before the process exits. What the count states is how
    /// far behind the writer was at the deadline, which is an upper bound on
    /// the loss rather than a measurement of it. An exact figure is not
    /// available to a caller that has chosen not to keep waiting.
    TimedOut { pending_at_deadline: u64 },
    /// The log was already stopped. Flushing twice is not an error.
    AlreadyStopped,
}

impl GatewayLog {
    pub fn open(path: impl AsRef<Path>) -> io::Result<Arc<Self>> {
        let file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path.as_ref())?;
        let destination = std::fs::canonicalize(path.as_ref())?;
        let (sender, receiver) = sync_channel(MAX_PENDING_LOG_RECORDS);
        let (stopped_sender, stopped_receiver) = sync_channel::<()>(1);
        let writer_destination = destination.clone();
        let written_records = Arc::new(AtomicU64::new(0));
        let writer_written = Arc::clone(&written_records);
        std::thread::Builder::new()
            .name("camelid-gateway-jsonl".into())
            .spawn(move || {
                write_jsonl_records(file, receiver, writer_destination, writer_written);
                // Best-effort: if the flush already timed out and dropped the
                // receiver, nobody is waiting for this and the send fails
                // harmlessly.
                let _ = stopped_sender.send(());
            })?;
        Ok(Arc::new(Self {
            destination,
            channel: Mutex::new(LogChannel {
                records: Some(sender),
                writer_stopped: Some(stopped_receiver),
            }),
            queued_records: AtomicU64::new(0),
            written_records,
            dropped_records: AtomicU64::new(0),
        }))
    }

    pub fn path(&self) -> &Path {
        &self.destination
    }

    pub fn has_same_destination(&self, other: &Self) -> bool {
        self.destination == other.destination
    }

    /// Stops accepting records, waits up to `deadline` for the writer to drain
    /// what was already accepted, and reports the outcome.
    ///
    /// Bounded on purpose. An unbounded wait would make a stuck filesystem able
    /// to hold the process open past the orchestrator's termination grace
    /// period, at which point the process is killed anyway and the wait has
    /// bought nothing while delaying every other shutdown step. Losing a
    /// bounded number of evidence records is the better failure, provided the
    /// shortfall is reported rather than silent.
    ///
    /// Giving up does not stop the writer. It continues draining until the
    /// process exits, so [`LogFlush::TimedOut`] reports how far behind it was,
    /// not how much was ultimately lost.
    ///
    /// Safe to call more than once and from any thread; later calls return
    /// [`LogFlush::AlreadyStopped`].
    pub fn flush_and_stop(&self, deadline: Duration) -> LogFlush {
        // Both halves come out under one lock, so a concurrent flush either
        // does all of this or none of it.
        //
        // Dropping the sender closes the channel. `Receiver`'s iterator yields
        // everything already buffered before it observes the disconnect, so
        // accepted records are written rather than discarded.
        let (sender, stopped) = {
            let mut channel = Self::locked(&self.channel);
            (channel.records.take(), channel.writer_stopped.take())
        };
        let (Some(sender), Some(stopped)) = (sender, stopped) else {
            return LogFlush::AlreadyStopped;
        };
        drop(sender);
        match stopped.recv_timeout(deadline) {
            Ok(()) => LogFlush::Drained,
            Err(_) => LogFlush::TimedOut {
                pending_at_deadline: self
                    .queued_records
                    .load(Ordering::Relaxed)
                    .saturating_sub(self.written_records.load(Ordering::Relaxed)),
            },
        }
    }

    fn locked<T>(value: &Mutex<T>) -> MutexGuard<'_, T> {
        value
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    fn write(&self, line: String) {
        let refusal = {
            let channel = Self::locked(&self.channel);
            match channel.records.as_ref() {
                None => Some("log is shutting down"),
                Some(sender) => match sender.try_send(line) {
                    Ok(()) => {
                        // Counted under the same lock `flush_and_stop` takes,
                        // so no record can be accepted after a flush has
                        // decided what the totals are.
                        self.queued_records.fetch_add(1, Ordering::Relaxed);
                        None
                    }
                    Err(TrySendError::Full(_)) => Some("writer queue is full"),
                    Err(TrySendError::Disconnected(_)) => Some("writer thread stopped"),
                },
            }
        };
        // Reported outside the lock: the warning must not serialise other
        // request tasks behind a logging call.
        if let Some(reason) = refusal {
            self.record_drop(reason);
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

fn write_jsonl_records(
    mut file: std::fs::File,
    records: Receiver<String>,
    destination: PathBuf,
    written: Arc<AtomicU64>,
) {
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
        // Counted whether or not the write succeeded: this tracks records the
        // writer has finished with, which is what makes the pending count at a
        // flush timeout mean "still queued" rather than "not yet durable".
        written.fetch_add(1, Ordering::Relaxed);
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
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

    fn label(&self) -> String {
        format!("{}://{}", self.scheme, self.authority)
    }
}

/// Immutable gateway-owned mapping from a public model id to its replica pool.
///
/// The catalog is validated before the gateway starts accepting traffic. Model
/// ids are opaque application identifiers: they are not paths, hostnames, or
/// client-supplied routing targets. The only network destinations are the
/// operator-configured [`UpstreamOrigin`] values held here.
#[derive(Clone, Debug)]
pub struct ModelCatalog {
    routes: BTreeMap<String, UpstreamOrigin>,
}

/// A [`ModelCatalog`] whose exact model ids have been verified against the
/// `/v1/models` response of every configured origin. Only this type can build
/// catalog routing, preventing a caller from serving an alias the backend will
/// reject after the gateway has already accepted the request.
#[derive(Clone, Debug)]
pub struct VerifiedModelCatalog {
    catalog: ModelCatalog,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InvalidModelCatalog(String);

impl fmt::Display for InvalidModelCatalog {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl std::error::Error for InvalidModelCatalog {}

/// Limits for the request-body work catalog mode performs before it knows which
/// replica pool to use.
///
/// Four bounds, each answering a different question: how large one selector
/// body may be, how much memory all of them may occupy at once, how long a
/// request may wait for a share of that memory, and how long it may hold one.
/// The last two are what keep the first two from turning ordinary concurrency
/// into failed requests -- see [`DEFAULT_MODEL_SELECTION_ACQUIRE_TIMEOUT`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ModelSelectionLimits {
    max_body_bytes: NonZeroUsize,
    memory_budget_bytes: NonZeroUsize,
    bytes_per_selector: NonZeroUsize,
    max_concurrent: NonZeroUsize,
    acquire_timeout: Duration,
    read_timeout: Duration,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InvalidModelSelectionLimits(String);

impl fmt::Display for InvalidModelSelectionLimits {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl std::error::Error for InvalidModelSelectionLimits {}

impl ModelSelectionLimits {
    /// Creates a selector-work limit. The body limit must fit in the declared
    /// budget so one valid selector is always possible. A selector holds its
    /// bounded raw body and the decoded copy of the JSON `model` string it
    /// extracts, so every slot reserves two body limits. The integer quotient
    /// is the exact maximum number of those worst-case selector states that
    /// may exist at once.
    ///
    /// Timing bounds start at [`DEFAULT_MODEL_SELECTION_ACQUIRE_TIMEOUT`] and
    /// [`DEFAULT_MODEL_SELECTION_READ_TIMEOUT`]; see
    /// [`Self::with_acquire_timeout`] and [`Self::with_read_timeout`].
    pub fn new(
        max_body_bytes: NonZeroUsize,
        memory_budget_bytes: NonZeroUsize,
    ) -> Result<Self, InvalidModelSelectionLimits> {
        let bytes_per_selector = max_body_bytes
            .get()
            .checked_mul(2)
            .and_then(NonZeroUsize::new)
            .ok_or_else(|| {
                InvalidModelSelectionLimits(
                    "max model selection body size is too large to reserve a decoded model id"
                        .into(),
                )
            })?;
        if memory_budget_bytes.get() < bytes_per_selector.get() {
            return Err(InvalidModelSelectionLimits(
                "model selection memory budget must reserve both the request body and a decoded model id"
                    .into(),
            ));
        }
        let max_concurrent =
            NonZeroUsize::new(memory_budget_bytes.get() / bytes_per_selector.get())
                .expect("a budget at least as large as one selector reservation has one slot");
        Ok(Self {
            max_body_bytes,
            memory_budget_bytes,
            bytes_per_selector,
            max_concurrent,
            acquire_timeout: DEFAULT_MODEL_SELECTION_ACQUIRE_TIMEOUT,
            read_timeout: DEFAULT_MODEL_SELECTION_READ_TIMEOUT,
        })
    }

    /// Overrides how long a request may wait for selector capacity. Exists so
    /// tests can exercise the shed-load path without waiting out the
    /// production budget.
    pub fn with_acquire_timeout(mut self, acquire_timeout: Duration) -> Self {
        self.acquire_timeout = acquire_timeout;
        self
    }

    /// Overrides how long a request may hold selector capacity while its body
    /// arrives.
    pub fn with_read_timeout(mut self, read_timeout: Duration) -> Self {
        self.read_timeout = read_timeout;
        self
    }

    pub fn max_body_bytes(self) -> NonZeroUsize {
        self.max_body_bytes
    }

    pub fn memory_budget_bytes(self) -> NonZeroUsize {
        self.memory_budget_bytes
    }

    pub fn bytes_per_selector(self) -> NonZeroUsize {
        self.bytes_per_selector
    }

    pub fn max_concurrent(self) -> NonZeroUsize {
        self.max_concurrent
    }

    pub fn acquire_timeout(self) -> Duration {
        self.acquire_timeout
    }

    pub fn read_timeout(self) -> Duration {
        self.read_timeout
    }

    /// How many selector bodies one authenticated organization may hold at
    /// once when the operator does not say.
    ///
    /// Half the global capacity, so the invariant an operator has to remember
    /// is a sentence: *no single organization can take more than half the
    /// selector budget*, and there is always room left for somebody else. A
    /// flat bound cannot say that -- it is either too small to serve one
    /// tenant's ordinary concurrency or too large to reserve anything for a
    /// second one, and which of the two it is depends on a budget it does not
    /// know about.
    pub fn default_max_org_selections(self) -> NonZeroUsize {
        NonZeroUsize::new(self.max_concurrent.get() / 2).unwrap_or(NonZeroUsize::MIN)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CatalogBackendVerificationError(String);

impl fmt::Display for CatalogBackendVerificationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl std::error::Error for CatalogBackendVerificationError {}

impl ModelCatalog {
    /// Builds a static catalog, rejecting an empty catalog, duplicate ids, and
    /// control characters that could make a model id unsafe to log or render.
    /// A [`BTreeMap`] gives discovery a stable lexical order regardless of the
    /// order in which an operator supplied the routes.
    pub fn new(
        routes: impl IntoIterator<Item = (String, UpstreamOrigin)>,
    ) -> Result<Self, InvalidModelCatalog> {
        let mut catalog = BTreeMap::new();
        for (model_id, upstream) in routes {
            if model_id.is_empty() {
                return Err(InvalidModelCatalog(
                    "model catalog entries must have a non-empty model id".into(),
                ));
            }
            if model_id.chars().any(char::is_control) {
                return Err(InvalidModelCatalog(format!(
                    "model catalog id {model_id:?} contains a control character"
                )));
            }
            if catalog.insert(model_id.clone(), upstream).is_some() {
                return Err(InvalidModelCatalog(format!(
                    "model catalog contains duplicate model id {model_id:?}"
                )));
            }
        }
        if catalog.is_empty() {
            return Err(InvalidModelCatalog(
                "model catalog must contain at least one model route".into(),
            ));
        }
        Ok(Self { routes: catalog })
    }

    /// Proves every catalog id is an exact id advertised by its configured
    /// replica pool. Catalog mode forwards a generation body unchanged, so an
    /// alias that differs from the backend's loaded id would route correctly
    /// and then fail at the replica as `model_not_found`; refusing startup is
    /// the only honest behavior without introducing request rewriting.
    pub async fn verify_backend_model_ids(
        self,
    ) -> Result<VerifiedModelCatalog, CatalogBackendVerificationError> {
        let mut origins: Vec<(UpstreamOrigin, Vec<String>)> = Vec::new();
        for (model_id, origin) in &self.routes {
            if let Some((_, configured_ids)) = origins
                .iter_mut()
                .find(|(configured_origin, _)| configured_origin == origin)
            {
                configured_ids.push(model_id.clone());
            } else {
                origins.push((origin.clone(), vec![model_id.clone()]));
            }
        }

        let client = http_client();
        for (origin, configured_ids) in origins {
            let uri = origin
                .request_uri(&"/v1/models".parse().expect("a constant URI is valid"))
                .map_err(|error| {
                    CatalogBackendVerificationError(format!(
                        "could not build model discovery URI for {}: {error}",
                        origin.label()
                    ))
                })?;
            let request = Request::get(uri).body(Body::empty()).map_err(|error| {
                CatalogBackendVerificationError(format!(
                    "could not build model discovery request for {}: {error}",
                    origin.label()
                ))
            })?;
            let response = tokio::time::timeout(CATALOG_DISCOVERY_TIMEOUT, client.request(request))
                .await
                .map_err(|_| {
                    CatalogBackendVerificationError(format!(
                        "timed out discovering models from {}",
                        origin.label()
                    ))
                })?
                .map_err(|error| {
                    CatalogBackendVerificationError(format!(
                        "could not discover models from {}: {error}",
                        origin.label()
                    ))
                })?;
            if response.status() != StatusCode::OK {
                return Err(CatalogBackendVerificationError(format!(
                    "model discovery from {} returned HTTP {}",
                    origin.label(),
                    response.status()
                )));
            }
            let body = tokio::time::timeout(
                CATALOG_DISCOVERY_TIMEOUT,
                Limited::new(response.into_body(), MAX_CATALOG_DISCOVERY_BODY_BYTES).collect(),
            )
            .await
            .map_err(|_| {
                CatalogBackendVerificationError(format!(
                    "timed out reading model discovery from {}",
                    origin.label()
                ))
            })?
            .map_err(|error| {
                CatalogBackendVerificationError(format!(
                    "could not read model discovery from {}: {error}",
                    origin.label()
                ))
            })?
            .to_bytes();
            let discovered: BackendModelList = serde_json::from_slice(&body).map_err(|error| {
                CatalogBackendVerificationError(format!(
                    "{} returned an invalid /v1/models response: {error}",
                    origin.label()
                ))
            })?;
            for model_id in configured_ids {
                if !discovered.data.iter().any(|model| model.id == model_id) {
                    return Err(CatalogBackendVerificationError(format!(
                        "catalog model id {model_id:?} is not advertised by {}",
                        origin.label()
                    )));
                }
            }
        }
        Ok(VerifiedModelCatalog { catalog: self })
    }
}

impl VerifiedModelCatalog {
    fn upstream(&self, model_id: &str) -> Option<&UpstreamOrigin> {
        self.catalog.routes.get(model_id)
    }

    fn route(&self, model_id: &str) -> Option<(&str, &UpstreamOrigin)> {
        self.catalog
            .routes
            .get_key_value(model_id)
            .map(|(catalog_model_id, upstream)| (catalog_model_id.as_str(), upstream))
    }

    /// Number of configured public model ids.
    pub fn len(&self) -> usize {
        self.catalog.routes.len()
    }

    /// Whether the catalog contains no public model ids.
    pub fn is_empty(&self) -> bool {
        self.catalog.routes.is_empty()
    }

    fn model_ids(&self) -> impl Iterator<Item = &str> {
        self.catalog.routes.keys().map(String::as_str)
    }
}

#[derive(Deserialize)]
struct BackendModelList {
    data: Vec<BackendModel>,
}

#[derive(Deserialize)]
struct BackendModel {
    id: String,
}

#[derive(Clone)]
enum UpstreamRouting {
    /// The original transparent gateway behavior. It leaves all public route
    /// semantics, including the replica's default-model behavior, to one
    /// fixed upstream.
    Passthrough(UpstreamOrigin),
    /// A gateway-owned static model catalog. Only the two generation routes
    /// have a contractual JSON `model` selector at the pinned engine revision;
    /// every other forwarded public route is refused instead of being sent to
    /// an arbitrary catalog entry whose semantics it cannot honestly represent.
    Catalog {
        catalog: VerifiedModelCatalog,
        selection_limits: ModelSelectionLimits,
        selection_admission: Arc<Semaphore>,
        org_selection_admission: Option<Arc<OrgSelectionAdmission>>,
    },
}

#[derive(Clone)]
struct GatewayState {
    routing: UpstreamRouting,
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

/// Per-organization concurrency bound for catalog selector bodies. This is
/// separate from [`OrgQuota`]: a selector is charged to quota only after it
/// proves routable, but a tenant must not be able to keep an incomplete body
/// open and monopolize global selector memory while remaining quota-free.
///
/// A semaphore per organization rather than a counter, because a request that
/// finds its organization at its bound has to be able to *wait* for a slot.
/// The bound exists to stop one tenant from occupying the global budget, not
/// to serialize that tenant: a counter can only say yes or no on contact, and
/// saying no to work that would have been admissible a millisecond later
/// rejects ordinary concurrency rather than abuse.
struct OrgSelectionAdmission {
    limit: NonZeroUsize,
    slots: Mutex<HashMap<String, Arc<Semaphore>>>,
}

impl OrgSelectionAdmission {
    fn new(limit: NonZeroUsize) -> Self {
        Self {
            limit,
            slots: Mutex::new(HashMap::new()),
        }
    }

    /// Waits until `deadline` for one of this organization's selector slots,
    /// or `None` if the budget expires first. The returned permit covers only
    /// the bounded body-read phase; it is released before quota, inference
    /// admission, and any response stream.
    async fn acquire(
        &self,
        organization: &OrganizationId,
        deadline: tokio::time::Instant,
    ) -> Option<OwnedSemaphorePermit> {
        let slot = self.slot(organization);
        // The semaphore is never closed, so an acquisition either succeeds or
        // runs out of budget.
        tokio::time::timeout_at(deadline, slot.acquire_owned())
            .await
            .ok()?
            .ok()
    }

    fn slot(&self, organization: &OrganizationId) -> Arc<Semaphore> {
        let mut slots = GatewayLog::locked(&self.slots);
        if let Some(slot) = slots.get(organization.as_str()) {
            return Arc::clone(slot);
        }
        // A key is created only for an organization with no live slot, so this
        // sweep runs off the steady-state path, exactly as `OrgQuota` sweeps
        // its windows. A semaphore whose only remaining owner is this map has
        // no permit holder and no waiter: every one of those clones the `Arc`
        // before releasing this lock, so a strong count of one under it means
        // the entry is idle and reclaimable.
        slots.retain(|_, slot| Arc::strong_count(slot) > 1);
        Arc::clone(
            slots
                .entry(organization.as_str().to_string())
                .or_insert_with(|| Arc::new(Semaphore::new(self.limit.get()))),
        )
    }
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
    router_with_routing_options(
        UpstreamRouting::Passthrough(upstream),
        max_in_flight,
        auth,
        audit,
    )
}

/// Builds a gateway that routes supported generation requests by the `model`
/// field in their bounded JSON body. Discovery is served from `catalog`; no
/// client-provided value can choose an origin outside that immutable map.
pub fn router_with_model_catalog(
    catalog: VerifiedModelCatalog,
    max_in_flight: NonZeroUsize,
    selection_limits: ModelSelectionLimits,
    max_org_model_selections: Option<NonZeroUsize>,
    auth: GatewayAuth,
    audit: Option<Arc<GatewayLog>>,
) -> Router {
    let org_selection_admission = matches!(&auth, GatewayAuth::RequireToken { .. }).then(|| {
        Arc::new(OrgSelectionAdmission::new(
            max_org_model_selections
                .unwrap_or_else(|| selection_limits.default_max_org_selections()),
        ))
    });
    router_with_routing_options(
        UpstreamRouting::Catalog {
            catalog,
            selection_admission: Arc::new(Semaphore::new(selection_limits.max_concurrent().get())),
            selection_limits,
            org_selection_admission,
        },
        max_in_flight,
        auth,
        audit,
    )
}

fn router_with_routing_options(
    routing: UpstreamRouting,
    max_in_flight: NonZeroUsize,
    auth: GatewayAuth,
    audit: Option<Arc<GatewayLog>>,
) -> Router {
    let client = http_client();
    // `/healthz` is the gateway's own liveness probe, not part of the replica
    // contract, so it is registered explicitly and answered locally.
    let mut router = Router::new().route("/healthz", get(gateway_health));
    for spec in replica_contract::contractual_routes() {
        router = router.route(spec.path, proxy_methods(spec));
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
            routing,
            client,
            admission: Arc::new(Semaphore::new(max_in_flight.get())),
            auth,
            audit,
        })
}

fn http_client() -> Client<HttpConnector, Body> {
    let mut connector = HttpConnector::new();
    connector.enforce_http(true);
    connector.set_connect_timeout(Some(CONNECT_TIMEOUT));
    let mut client_builder = Client::builder(TokioExecutor::new());
    client_builder.retry_canceled_requests(false);
    client_builder.pool_timer(TokioTimer::new());
    client_builder.pool_idle_timeout(POOL_IDLE_TIMEOUT);
    client_builder.pool_max_idle_per_host(MAX_IDLE_PER_HOST);
    client_builder.build(connector)
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
fn proxy_methods(route: &RouteSpec) -> MethodRouter<GatewayState> {
    let mut method_router = MethodRouter::new();
    for method in route.methods {
        method_router = match method {
            HttpMethod::Get if route.path == "/v1/models/:model" => {
                method_router.get(proxy_model_detail)
            }
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
    request: Request,
) -> Response {
    proxy_request(state, connection_termination, request, None).await
}

async fn proxy_model_detail(
    State(state): State<GatewayState>,
    AxumPath(model_id): AxumPath<String>,
    connection_termination: Option<Extension<Arc<ConnectionTermination>>>,
    request: Request,
) -> Response {
    proxy_request(state, connection_termination, request, Some(model_id)).await
}

async fn proxy_request(
    state: GatewayState,
    connection_termination: Option<Extension<Arc<ConnectionTermination>>>,
    mut request: Request,
    detail_model_id: Option<String>,
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
    let (identity, quota, usage) = match &state.auth {
        GatewayAuth::RequireToken {
            store,
            quota,
            usage,
        } => {
            let identity = match authenticate(store, request.headers()).await {
                Ok(context) => context,
                Err(refusal) => {
                    return audited(
                        &state,
                        AuditedRequest {
                            request_id: &request_id,
                            identity: None,
                            reason: Some(refusal.audit_reason()),
                            method: &method,
                            path: &path,
                            model_id: None,
                            usage: None,
                            request_metrics: None,
                            connection_termination: connection_termination.clone(),
                            started_ts,
                            started_at,
                        },
                        refusal.into_response(),
                    )
                }
            };

            (Some(identity), quota.clone(), usage.clone())
        }
        GatewayAuth::Disabled => (None, None, None),
    };

    // Catalog selection happens after authentication, so an authenticated
    // deployment does not disclose its model inventory to an unauthenticated
    // caller. Anything decidable from the request head alone is refused before
    // any capacity is committed to it, then per-organization capacity is
    // acquired before the global selector-work budget: one tenant's stalled
    // body cannot take every global slot while invalid selectors remain
    // quota-free. Both permits cover only body materialization; inference
    // admission stays free until selection succeeds.
    let selector_permits = match state.routing.catalog_selection(&request) {
        None => None,
        Some(selection) => {
            let acquired = match model_selection_head_refusal(request.headers(), selection.limits) {
                Some(refusal) => Err(refusal),
                None => acquire_selection_capacity(&selection, identity.as_ref()).await,
            };
            match acquired {
                Ok(permits) => Some(permits),
                Err(refusal) => {
                    return audited(
                        &state,
                        AuditedRequest {
                            request_id: &request_id,
                            identity: identity.as_ref(),
                            reason: Some(refusal.audit_reason),
                            method: &method,
                            path: &path,
                            model_id: None,
                            usage: None,
                            request_metrics: None,
                            connection_termination: connection_termination.clone(),
                            started_ts,
                            started_at,
                        },
                        refusal.response,
                    )
                }
            }
        }
    };
    let route = match resolve_route(&state.routing, &mut request, detail_model_id.as_deref()).await
    {
        Ok(route) => route,
        Err(refusal) => {
            return audited(
                &state,
                AuditedRequest {
                    request_id: &request_id,
                    identity: identity.as_ref(),
                    reason: Some(refusal.audit_reason),
                    method: &method,
                    path: &path,
                    model_id: None,
                    usage: None,
                    request_metrics: None,
                    connection_termination: connection_termination.clone(),
                    started_ts,
                    started_at,
                },
                refusal.response,
            )
        }
    };
    // Body materialization has completed, failed, or timed out. The request
    // body is now either back in `request` for forwarding or the handler has
    // returned, so selector capacity can be released before quota/admission
    // and the response stream.
    drop(selector_permits);

    // Quota is checked after authentication (it needs a resolved organization)
    // and after a request has proved routable, but before it can take an
    // admission permit. A request the gateway rejects must never first take a
    // permit meant for traffic that can reach a replica.
    if let (Some(identity), Some(quota)) = (identity.as_ref(), quota.as_ref()) {
        if let Err(retry_after) = quota.admit(identity.organization_id()) {
            let response = quota_exceeded(retry_after);
            return audited(
                &state,
                AuditedRequest {
                    request_id: &request_id,
                    identity: Some(identity),
                    reason: Some("organization_quota_exceeded"),
                    method: &method,
                    path: &path,
                    model_id: route.model_id(),
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

    let permit = match Arc::clone(&state.admission).try_acquire_owned() {
        Ok(permit) => permit,
        Err(_) => {
            return audited(
                &state,
                AuditedRequest {
                    request_id: &request_id,
                    identity: identity.as_ref(),
                    reason: Some("gateway_overloaded"),
                    method: &method,
                    path: &path,
                    model_id: route.model_id(),
                    usage: usage.clone(),
                    request_metrics: None,
                    connection_termination: connection_termination.clone(),
                    started_ts,
                    started_at,
                },
                overloaded("gateway concurrency limit reached"),
            );
        }
    };
    let (upstream, model_id) = match route {
        RoutedRequest::Forward { upstream, model_id } => (upstream, model_id),
        RoutedRequest::Local { response, model_id } => {
            drop(permit);
            return audited(
                &state,
                AuditedRequest {
                    request_id: &request_id,
                    identity: identity.as_ref(),
                    reason: None,
                    method: &method,
                    path: &path,
                    model_id: model_id.as_deref(),
                    usage,
                    request_metrics: None,
                    connection_termination,
                    started_ts,
                    started_at,
                },
                response,
            );
        }
    };
    let upstream_uri = match upstream.request_uri(request.uri()) {
        Ok(uri) => uri,
        Err(error) => {
            tracing::error!(%error, "could not construct upstream request URI");
            return audited(
                &state,
                AuditedRequest {
                    request_id: &request_id,
                    identity: identity.as_ref(),
                    reason: Some("upstream_uri_invalid"),
                    method: &method,
                    path: &path,
                    model_id: model_id.as_deref(),
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
    let host = HeaderValue::from_str(upstream.authority.as_str())
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

    let (response, reason) = match state.client.request(request).await {
        Ok(response) => {
            let (mut parts, body) = response.into_parts();
            remove_hop_by_hop(&mut parts.headers);
            // Forwarded: the upstream's own status is the record, and the
            // gateway has no refusal of its own to name.
            (
                Response::from_parts(parts, Body::new(PermitBody::new(body, permit))),
                None,
            )
        }
        Err(error) => {
            tracing::warn!(%error, "upstream replica request failed");
            (
                gateway_error(StatusCode::BAD_GATEWAY, "upstream replica is unavailable"),
                Some("upstream_unavailable"),
            )
        }
    };
    audited(
        &state,
        AuditedRequest {
            request_id: &request_id,
            identity: identity.as_ref(),
            reason,
            method: &method,
            path: &path,
            model_id: model_id.as_deref(),
            usage,
            request_metrics,
            connection_termination,
            started_ts,
            started_at,
        },
        response,
    )
}

enum RoutedRequest {
    Forward {
        upstream: UpstreamOrigin,
        model_id: Option<String>,
    },
    Local {
        response: Response,
        model_id: Option<String>,
    },
}

impl RoutedRequest {
    fn model_id(&self) -> Option<&str> {
        match self {
            Self::Forward { model_id, .. } | Self::Local { model_id, .. } => model_id.as_deref(),
        }
    }
}

impl UpstreamRouting {
    /// The selector-work capacity a request must hold before catalog mode may
    /// read its body, or `None` when nothing about this request materializes
    /// one: transparent routing, local catalog discovery, and endpoints with
    /// no model-selection contract all read no body here.
    fn catalog_selection(&self, request: &Request) -> Option<CatalogSelection> {
        match self {
            Self::Catalog {
                selection_limits,
                selection_admission,
                org_selection_admission,
                ..
            } if is_catalog_generation_route(request) => Some(CatalogSelection {
                limits: *selection_limits,
                global: Arc::clone(selection_admission),
                per_organization: org_selection_admission.clone(),
            }),
            Self::Passthrough(_) | Self::Catalog { .. } => None,
        }
    }
}

/// The two routes whose JSON body carries a contractual `model` selector at the
/// pinned engine revision. Every place that has to agree on "is this request
/// routed by its body?" asks here, so the admission gate, the refusal gate, and
/// the router cannot drift apart.
fn is_catalog_generation_route(request: &Request) -> bool {
    request.method() == Method::POST
        && matches!(
            request.uri().path(),
            "/v1/completions" | "/v1/chat/completions"
        )
}

/// Selector-work capacity resolved from the routing mode for one request.
struct CatalogSelection {
    limits: ModelSelectionLimits,
    global: Arc<Semaphore>,
    /// `None` when the gateway is unauthenticated: without a resolved
    /// organization there is no tenant to be fair between.
    per_organization: Option<Arc<OrgSelectionAdmission>>,
}

/// Selector capacity held for exactly one request's body materialization.
///
/// Both permits release on drop, including when the request task is cancelled
/// mid-body. Neither covers quota, inference admission, or a response stream:
/// [`proxy_request`] drops this the moment selection resolves.
struct SelectionPermits {
    /// Acquired first, so a tenant already at its bound waits on its own share
    /// instead of consuming a global slot to do so.
    _per_organization: Option<OwnedSemaphorePermit>,
    _global: OwnedSemaphorePermit,
}

/// Waits for selector capacity, or reports the refusal to send instead.
///
/// One deadline covers both acquisitions rather than one each: two five-second
/// budgets in series is a ten-second budget, and the number a shed-load bound
/// promises has to be the number a request can actually experience.
async fn acquire_selection_capacity(
    selection: &CatalogSelection,
    identity: Option<&AuthenticatedContext>,
) -> Result<SelectionPermits, RoutingRefusal> {
    let deadline = tokio::time::Instant::now() + selection.limits.acquire_timeout();
    let per_organization = match (identity, selection.per_organization.as_ref()) {
        (Some(identity), Some(admission)) => {
            let permit = admission
                .acquire(identity.organization_id(), deadline)
                .await;
            if permit.is_none() {
                return Err(RoutingRefusal {
                    audit_reason: "organization_model_selection_overloaded",
                    response: overloaded("gateway organization model selection limit reached"),
                });
            }
            permit
        }
        _ => None,
    };
    // Same reasoning as the per-organization slot: closure is impossible, so
    // the only failure is running out of budget.
    let Some(Ok(global)) =
        tokio::time::timeout_at(deadline, Arc::clone(&selection.global).acquire_owned())
            .await
            .ok()
    else {
        return Err(RoutingRefusal {
            audit_reason: "gateway_model_selection_overloaded",
            response: overloaded("gateway model selection limit reached"),
        });
    };
    Ok(SelectionPermits {
        _per_organization: per_organization,
        _global: global,
    })
}

struct RoutingRefusal {
    audit_reason: &'static str,
    response: Response,
}

/// The one field catalog routing reads out of a generation request body.
///
/// Deserialized through an explicit map visitor rather than `#[derive]`.
/// serde's derived struct implementation also accepts a JSON *sequence*,
/// filling fields positionally, so `["alpha"]` would be read as
/// `{"model":"alpha"}` and routed. A replica deserializing the same bytes maps
/// position zero onto the first field of *its* request type, so gateway and
/// replica would disagree about what the request says while the gateway logged
/// a `model_id` it had inferred from a body that never named one. A selector
/// that decides where a request goes must not be looser than the format it
/// claims to read: only a JSON object carries a top-level `model` member.
struct ModelSelection {
    model: Option<String>,
}

impl<'de> Deserialize<'de> for ModelSelection {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        deserializer.deserialize_map(ModelSelectionVisitor)
    }
}

struct ModelSelectionVisitor;

impl<'de> serde::de::Visitor<'de> for ModelSelectionVisitor {
    type Value = ModelSelection;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("a JSON object with an optional string \"model\" member")
    }

    fn visit_map<A>(self, mut entries: A) -> Result<ModelSelection, A::Error>
    where
        A: serde::de::MapAccess<'de>,
    {
        let mut model = None;
        while let Some(key) = entries.next_key::<Cow<'_, str>>()? {
            if key != "model" {
                // Every other member is the replica's business. Skipping is
                // what keeps this selector from becoming a second, divergent
                // copy of the generation request schema.
                entries.next_value::<serde::de::IgnoredAny>()?;
                continue;
            }
            if model.is_some() {
                // Matches the derived behavior this replaces. A body naming
                // two models has no single answer, and picking one would make
                // the gateway's choice differ from a parser that picks the
                // other.
                return Err(serde::de::Error::duplicate_field("model"));
            }
            model = Some(entries.next_value::<Option<String>>()?);
        }
        Ok(ModelSelection {
            model: model.flatten(),
        })
    }
}

/// Refusals decidable from a generation request's head alone, before the
/// gateway commits any selector capacity to reading its body.
///
/// Both are cheap and both are unconditional, so answering them first means a
/// caller cannot spend a slot -- or the bandwidth behind it -- on a request
/// that was never going to be routed.
fn model_selection_head_refusal(
    headers: &HeaderMap,
    limits: ModelSelectionLimits,
) -> Option<RoutingRefusal> {
    if !has_json_content_type(headers) {
        return Some(unsupported_model_selection_media_type_refusal());
    }
    // A declared length over the limit is a decision the streaming limit would
    // reach anyway, after reading the whole allowance first. Hyper holds the
    // body to this declaration, so a request that understates it is truncated
    // rather than admitted.
    if declared_content_length(headers).is_some_and(|declared| declared > limits.max_body_bytes()) {
        return Some(model_selection_body_too_large_refusal());
    }
    None
}

fn declared_content_length(headers: &HeaderMap) -> Option<NonZeroUsize> {
    headers
        .get(CONTENT_LENGTH)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.trim().parse::<usize>().ok())
        .and_then(NonZeroUsize::new)
}

async fn select_catalog_upstream(
    catalog: &VerifiedModelCatalog,
    request: &mut Request,
    limits: ModelSelectionLimits,
) -> Result<RoutedRequest, RoutingRefusal> {
    if let Some(refusal) = model_selection_head_refusal(request.headers(), limits) {
        return Err(refusal);
    }

    let incoming = std::mem::replace(request, Request::new(Body::empty()));
    let (parts, body) = incoming.into_parts();
    let body = match tokio::time::timeout(
        limits.read_timeout(),
        to_bytes(body, limits.max_body_bytes().get()),
    )
    .await
    {
        Ok(Ok(body)) => body,
        Ok(Err(error)) => {
            let is_too_large = std::error::Error::source(&error)
                .is_some_and(|source| source.is::<LengthLimitError>());
            return Err(if is_too_large {
                model_selection_body_too_large_refusal()
            } else {
                RoutingRefusal {
                    audit_reason: "model_selection_body_unreadable",
                    response: gateway_api_error(
                        StatusCode::BAD_REQUEST,
                        "gateway could not read the request body",
                        "invalid_request",
                        Some("unreadable_request_body"),
                        None,
                    ),
                }
            });
        }
        Err(_) => {
            return Err(RoutingRefusal {
                audit_reason: "model_selection_body_timeout",
                response: gateway_api_error(
                    StatusCode::REQUEST_TIMEOUT,
                    "request body did not arrive within the model-selection deadline",
                    "invalid_request",
                    Some("request_body_timeout"),
                    None,
                ),
            })
        }
    };
    let selection: ModelSelection = match serde_json::from_slice(&body) {
        Ok(selection) => selection,
        Err(_) => {
            return Err(RoutingRefusal {
                audit_reason: "malformed_model_selection",
                response: gateway_api_error(
                    StatusCode::BAD_REQUEST,
                    "request body is not a JSON object",
                    "invalid_request",
                    Some("malformed_json"),
                    None,
                ),
            });
        }
    };
    let Some(model_id) = selection
        .model
        .as_deref()
        .filter(|model_id| !model_id.is_empty())
    else {
        return Err(RoutingRefusal {
            audit_reason: "missing_model",
            response: gateway_api_error(
                StatusCode::BAD_REQUEST,
                "request must specify a non-empty model",
                "invalid_request",
                Some("model_required"),
                Some("model"),
            ),
        });
    };
    let Some((catalog_model_id, upstream)) = catalog.route(model_id) else {
        return Err(unknown_model_refusal());
    };
    let routed_model_id = catalog_model_id.to_string();

    // The exact bytes, headers, and URI survive selection. The catalog changes
    // only the destination, leaving the pinned engine as the authority for all
    // request-field validation and generation semantics.
    *request = Request::from_parts(parts, Body::from(body));
    Ok(RoutedRequest::Forward {
        upstream: upstream.clone(),
        model_id: Some(routed_model_id),
    })
}

async fn resolve_route(
    routing: &UpstreamRouting,
    request: &mut Request,
    detail_model_id: Option<&str>,
) -> Result<RoutedRequest, RoutingRefusal> {
    match routing {
        UpstreamRouting::Passthrough(upstream) => Ok(RoutedRequest::Forward {
            upstream: upstream.clone(),
            model_id: None,
        }),
        UpstreamRouting::Catalog {
            catalog,
            selection_limits,
            ..
        } => {
            let is_read = matches!(request.method(), &Method::GET | &Method::HEAD);
            if is_read && request.uri().path() == "/v1/models" {
                return Ok(RoutedRequest::Local {
                    response: catalog_models_response(catalog),
                    model_id: None,
                });
            }
            if is_read {
                if let Some(model_id) = detail_model_id {
                    return match catalog.upstream(model_id) {
                        Some(_) => Ok(RoutedRequest::Local {
                            response: catalog_model_response(model_id),
                            model_id: Some(model_id.to_string()),
                        }),
                        None => Err(unknown_model_refusal()),
                    };
                }
            }
            if is_catalog_generation_route(request) {
                return select_catalog_upstream(catalog, request, *selection_limits).await;
            }
            Err(RoutingRefusal {
                audit_reason: "model_routing_unsupported",
                response: gateway_api_error(
                    StatusCode::NOT_IMPLEMENTED,
                    "this endpoint cannot be routed by a multi-model gateway",
                    "not_implemented",
                    Some("model_routing_unsupported"),
                    None,
                ),
            })
        }
    }
}

fn has_json_content_type(headers: &HeaderMap) -> bool {
    headers
        .get(CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.split(';').next())
        .is_some_and(|media_type| {
            let media_type = media_type.trim().to_ascii_lowercase();
            media_type == "application/json"
                || (media_type.starts_with("application/") && media_type.ends_with("+json"))
        })
}

fn unsupported_model_selection_media_type_refusal() -> RoutingRefusal {
    RoutingRefusal {
        audit_reason: "unsupported_model_selection_media_type",
        response: gateway_api_error(
            StatusCode::UNSUPPORTED_MEDIA_TYPE,
            "model routing requires an application/json request body",
            "invalid_request",
            Some("unsupported_media_type"),
            None,
        ),
    }
}

fn model_selection_body_too_large_refusal() -> RoutingRefusal {
    RoutingRefusal {
        audit_reason: "model_selection_body_too_large",
        response: gateway_api_error(
            StatusCode::PAYLOAD_TOO_LARGE,
            "request body exceeds the model-selection limit",
            "invalid_request",
            Some("request_too_large"),
            None,
        ),
    }
}

fn unknown_model_refusal() -> RoutingRefusal {
    RoutingRefusal {
        audit_reason: "unknown_model",
        response: gateway_api_error(
            StatusCode::NOT_FOUND,
            "requested model is not available from this gateway",
            "invalid_request",
            Some("model_not_found"),
            Some("model"),
        ),
    }
}

fn catalog_models_response(catalog: &VerifiedModelCatalog) -> Response {
    Json(serde_json::json!({
        "object": "list",
        "data": catalog.model_ids().map(catalog_model_item).collect::<Vec<_>>(),
    }))
    .into_response()
}

fn catalog_model_response(model_id: &str) -> Response {
    Json(catalog_model_item(model_id)).into_response()
}

fn catalog_model_item(model_id: &str) -> serde_json::Value {
    serde_json::json!({
        "id": model_id,
        "object": "model",
        "created": 0,
        "owned_by": "camelid",
    })
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
    /// Why the gateway refused this request, when it refused it itself.
    /// `None` for anything it forwarded: the upstream's status speaks for
    /// those. This is what separates two refusals that share a status code and
    /// a null principal, which authentication failures always do.
    reason: Option<&'static str>,
    method: &'a str,
    path: &'a str,
    /// The catalog entry selected for this request. `None` means transparent
    /// legacy routing, catalog discovery, or a refusal before selection.
    model_id: Option<&'a str>,
    usage: Option<Arc<GatewayLog>>,
    request_metrics: Option<Arc<RequestBodyMetrics>>,
    connection_termination: Option<Arc<ConnectionTermination>>,
    started_ts: f64,
    started_at: Instant,
}

fn audited(state: &GatewayState, request: AuditedRequest<'_>, response: Response) -> Response {
    if let Some(log) = &state.audit {
        write_audit_record(log, &request, response.status());
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
        model_id: request.model_id.map(str::to_string),
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
/// reason, method, path, model_id, status}`. `principal` and `organization` are opaque
/// gateway-local identity fields, or JSON `null` when authentication is
/// disabled or rejected before identity was established. `reason` names the
/// gateway's own refusal, or is `null` for a request it forwarded. They are
/// never forwarded to a replica. `request_id` is the only correlation value
/// sent upstream.
///
/// Best-effort and off the request's async context: a failed write must never
/// fail the request. The dedicated bounded writer queue can lose records when
/// full, after a write failure, or if an abrupt crash takes the process down;
/// [`GatewayLog`] emits
/// rate-limited warnings for each condition. Its single writer serializes
/// complete JSONL records so concurrent requests cannot interleave lines.
fn write_audit_record(log: &GatewayLog, request: &AuditedRequest<'_>, status: StatusCode) {
    let line = serde_json::json!({
        "ts": unix_timestamp(),
        "request_id": request.request_id,
        "principal": request.identity.map(|context| context.principal_id().as_str()),
        "organization": request.identity.map(|context| context.organization_id().as_str()),
        "reason": request.reason,
        "method": request.method,
        "path": request.path,
        "model_id": request.model_id,
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
    model_id: Option<String>,
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
/// path, model_id, response_head_status, request_bytes, response_bytes, stream_outcome}`.
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
        "model_id": record.model_id,
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
    gateway_api_error(status, message, "gateway_error", None, None)
}

/// A typed `503` with a jittered `Retry-After`: the shared shape for every
/// capacity the gateway sheds load on, so a client sees one backpressure
/// contract rather than one per bound.
fn overloaded(message: &'static str) -> Response {
    let mut response = gateway_error(StatusCode::SERVICE_UNAVAILABLE, message);
    let (low, high) = RETRY_AFTER_JITTER_SECONDS;
    let retry_after = fastrand::u8(low..=high);
    response.headers_mut().insert(
        "retry-after",
        HeaderValue::from_str(&retry_after.to_string())
            .expect("a small decimal integer is a valid header value"),
    );
    response
}

fn gateway_api_error(
    status: StatusCode,
    message: &'static str,
    error_type: &'static str,
    code: Option<&'static str>,
    parameter: Option<&'static str>,
) -> Response {
    let mut error = serde_json::Map::new();
    error.insert("message".into(), serde_json::Value::String(message.into()));
    error.insert("type".into(), serde_json::Value::String(error_type.into()));
    if let Some(code) = code {
        error.insert("code".into(), serde_json::Value::String(code.into()));
    }
    if let Some(parameter) = parameter {
        error.insert("param".into(), serde_json::Value::String(parameter.into()));
    }
    let mut response = (status, Json(serde_json::json!({ "error": error }))).into_response();
    response
        .headers_mut()
        .insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
    response
}

/// Resolves the request's bearer token to its principal and organization, or
/// returns the refusal the gateway should send without forwarding.
async fn authenticate(
    store: &Arc<SqliteIdentityStore>,
    headers: &HeaderMap,
) -> Result<AuthenticatedContext, AuthRefusal> {
    let Some(token) = bearer_token(headers) else {
        return Err(AuthRefusal::MissingToken);
    };
    let store = Arc::clone(store);
    let token = token.to_string();
    let resolved = tokio::task::spawn_blocking(move || store.resolve_context(&token))
        .await
        .expect("identity resolution task panicked");

    resolved.map_err(|error| match error {
        IdentityError::InvalidToken => AuthRefusal::InvalidToken,
        IdentityError::ExpiredToken => AuthRefusal::ExpiredToken,
        error => {
            tracing::error!(%error, "identity store error while authenticating request");
            AuthRefusal::StoreUnavailable
        }
    })
}

/// Why a request was refused before it reached a replica.
///
/// A type rather than a prebuilt [`Response`], because the reason has to reach
/// three places and a formatted body only reaches one: the client's error
/// `type`, the `WWW-Authenticate` challenge, and the audit record. An earlier
/// revision distinguished expiry from invalidity in English prose alone, which
/// meant the distinction existed for nobody -- a client would have had to
/// string-match a human sentence, and the audit log recorded the two
/// identically.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum AuthRefusal {
    MissingToken,
    InvalidToken,
    ExpiredToken,
    StoreUnavailable,
}

impl AuthRefusal {
    /// The stable machine discriminator, in the response body's `error.type`
    /// field -- the same slot `quota_exceeded` and `gateway_error` use, and
    /// the one a client is entitled to branch on.
    fn error_type(self) -> &'static str {
        match self {
            // Unchanged for the two pre-existing cases: clients already branch
            // on this value, and "you are not authorized" is still all that can
            // be said about a request carrying no or an unknown credential.
            Self::MissingToken | Self::InvalidToken => "unauthorized",
            // The one case where a correct client has a distinct action
            // available: stop retrying this credential and obtain another.
            Self::ExpiredToken => "token_expired",
            Self::StoreUnavailable => "gateway_error",
        }
    }

    fn message(self) -> &'static str {
        match self {
            Self::MissingToken => "missing bearer token",
            Self::InvalidToken => "invalid bearer token",
            Self::ExpiredToken => "expired bearer token",
            Self::StoreUnavailable => "gateway could not verify the bearer token",
        }
    }

    /// The value recorded in the audit log, so an operator reading it can tell
    /// a client using a lapsed credential from one presenting an unknown
    /// secret. Both are `401` with a null principal, so without this the two
    /// lines are byte-identical.
    fn audit_reason(self) -> &'static str {
        match self {
            Self::MissingToken => "missing_token",
            Self::InvalidToken => "invalid_token",
            Self::ExpiredToken => "expired_token",
            Self::StoreUnavailable => "identity_store_unavailable",
        }
    }

    /// The `WWW-Authenticate` challenge, per RFC 6750 section 3.
    ///
    /// `invalid_token` is the registered code covering expired, revoked and
    /// malformed credentials alike -- there is no separate `expired` code, so
    /// the distinction rides in `error_description` and in the body's `type`.
    /// A request that carried no credential at all gets a bare challenge:
    /// section 3 says the server SHOULD NOT emit an error code when none was
    /// presented, since nothing was wrong with a credential that was never
    /// sent.
    fn challenge(self) -> Option<HeaderValue> {
        match self {
            Self::MissingToken => Some(HeaderValue::from_static("Bearer")),
            Self::InvalidToken => Some(HeaderValue::from_static(
                r#"Bearer error="invalid_token", error_description="invalid bearer token""#,
            )),
            Self::ExpiredToken => Some(HeaderValue::from_static(
                r#"Bearer error="invalid_token", error_description="expired bearer token""#,
            )),
            Self::StoreUnavailable => None,
        }
    }

    fn status(self) -> StatusCode {
        match self {
            Self::StoreUnavailable => StatusCode::INTERNAL_SERVER_ERROR,
            _ => StatusCode::UNAUTHORIZED,
        }
    }

    fn into_response(self) -> Response {
        let mut response = (
            self.status(),
            Json(serde_json::json!({
                "error": {
                    "message": self.message(),
                    "type": self.error_type()
                }
            })),
        )
            .into_response();
        response
            .headers_mut()
            .insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
        if let Some(challenge) = self.challenge() {
            response.headers_mut().insert(WWW_AUTHENTICATE, challenge);
        }
        response
    }
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
    fn model_catalog_rejects_ambiguous_ids_before_backend_verification() {
        let alpha = UpstreamOrigin::parse("http://alpha.test").unwrap();
        let bravo = UpstreamOrigin::parse("http://bravo.test").unwrap();
        assert!(ModelCatalog::new([
            ("bravo".to_string(), bravo.clone()),
            ("alpha".to_string(), alpha.clone()),
        ])
        .is_ok());

        let duplicate = ModelCatalog::new([
            ("alpha".to_string(), alpha.clone()),
            ("alpha".to_string(), bravo.clone()),
        ])
        .unwrap_err();
        assert!(duplicate.to_string().contains("duplicate model id"));
        assert!(ModelCatalog::new([(String::new(), alpha.clone())]).is_err());
        assert!(ModelCatalog::new([("line\nbreak".to_string(), alpha)]).is_err());
        assert!(ModelCatalog::new(Vec::<(String, UpstreamOrigin)>::new()).is_err());
    }

    #[test]
    fn model_selection_limits_bound_materialized_bodies_by_the_declared_budget() {
        let limits = ModelSelectionLimits::new(
            NonZeroUsize::new(2 * 1024 * 1024).unwrap(),
            NonZeroUsize::new(32 * 1024 * 1024).unwrap(),
        )
        .unwrap();
        assert_eq!(limits.max_concurrent().get(), 8);
        assert_eq!(
            limits
                .max_concurrent()
                .get()
                .saturating_mul(limits.bytes_per_selector().get()),
            limits.memory_budget_bytes().get()
        );
        assert!(ModelSelectionLimits::new(
            NonZeroUsize::new(2).unwrap(),
            NonZeroUsize::new(3).unwrap(),
        )
        .is_err());
    }

    /// The per-organization default is derived from the global capacity so the
    /// invariant survives a reconfigured budget: no organization takes more
    /// than half of it, and something is always left for another tenant. A
    /// flat constant cannot state that, and the flat `1` this replaces made a
    /// single-tenant deployment serialize its own selector work.
    #[test]
    fn the_per_organization_selector_default_is_half_the_global_capacity() {
        let limits = |budget_multiple: usize| {
            ModelSelectionLimits::new(
                NonZeroUsize::new(1024).unwrap(),
                NonZeroUsize::new(2048 * budget_multiple).unwrap(),
            )
            .unwrap()
        };
        assert_eq!(limits(16).max_concurrent().get(), 16);
        assert_eq!(limits(16).default_max_org_selections().get(), 8);
        assert_eq!(limits(2).default_max_org_selections().get(), 1);
        // A budget with room for a single selector cannot reserve half of one.
        // One is the floor, never zero.
        assert_eq!(limits(1).max_concurrent().get(), 1);
        assert_eq!(limits(1).default_max_org_selections().get(), 1);
    }

    /// The selector decides where a request goes, so it must not read a body
    /// shape that names no model. serde's derived struct impl fills fields
    /// from a JSON sequence positionally, which would make `["alpha"]` a
    /// routable selection of `alpha`.
    #[test]
    fn model_selection_accepts_only_a_json_object() {
        let model = |body: &str| {
            serde_json::from_str::<ModelSelection>(body).map(|selection| selection.model)
        };
        assert_eq!(
            model(r#"{"model":"alpha"}"#).unwrap().as_deref(),
            Some("alpha")
        );
        assert_eq!(
            model(r#"{"messages":[],"model":"alpha"}"#)
                .unwrap()
                .as_deref(),
            Some("alpha")
        );
        assert_eq!(model(r#"{"model":null}"#).unwrap(), None);
        assert_eq!(model("{}").unwrap(), None);
        // Escapes decode, so the gateway matches on the same value the replica
        // will parse out of the very bytes it forwards.
        assert_eq!(
            model(r#"{"model":"\u0061lpha"}"#).unwrap().as_deref(),
            Some("alpha")
        );
        for not_an_object in [r#"["alpha"]"#, "[]", r#""alpha""#, "null", "7"] {
            assert!(
                model(not_an_object).is_err(),
                "{not_an_object} names no top-level model member"
            );
        }
        // Two answers is no answer: picking one would route somewhere a parser
        // that picked the other would not.
        assert!(model(r#"{"model":"alpha","model":"bravo"}"#).is_err());
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

    /// Reads the sink without retrying. Every test below asserts on the file
    /// the instant `flush_and_stop` returns, because polling until the writer
    /// happens to catch up would pass whether or not shutdown drains anything
    /// -- which is precisely the bug these tests exist to hold closed.
    fn read_lines_now(path: &Path) -> Vec<String> {
        std::fs::read_to_string(path)
            .unwrap()
            .lines()
            .filter(|line| !line.trim().is_empty())
            .map(str::to_string)
            .collect()
    }

    #[test]
    fn flush_and_stop_persists_every_accepted_record() {
        // Comfortably more than the writer can drain in the time it takes to
        // enqueue them, and well under MAX_PENDING_LOG_RECORDS so nothing is
        // dropped for a full queue: every record here is *accepted*, which is
        // what shutdown promises to persist.
        const RECORDS: usize = 500;
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("audit.jsonl");
        let log = GatewayLog::open(&path).unwrap();

        for index in 0..RECORDS {
            log.write(format!("{{\"n\":{index}}}\n"));
        }
        assert_eq!(
            log.flush_and_stop(Duration::from_secs(30)),
            LogFlush::Drained
        );

        let lines = read_lines_now(&path);
        assert_eq!(
            lines.len(),
            RECORDS,
            "every accepted record must be on disk once flush reports Drained"
        );
        // Order is the queue's order, so the sink is append-only and in
        // sequence rather than merely complete.
        for (index, line) in lines.iter().enumerate() {
            assert_eq!(line, &format!("{{\"n\":{index}}}"));
        }
        assert_eq!(log.dropped_records.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn flush_and_stop_is_idempotent() {
        let dir = tempfile::tempdir().unwrap();
        let log = GatewayLog::open(dir.path().join("audit.jsonl")).unwrap();

        assert_eq!(
            log.flush_and_stop(Duration::from_secs(30)),
            LogFlush::Drained
        );
        assert_eq!(
            log.flush_and_stop(Duration::from_secs(30)),
            LogFlush::AlreadyStopped,
            "a second flush must be a no-op, not a hang or a panic"
        );
    }

    #[test]
    fn writes_after_flush_are_dropped_and_counted() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("audit.jsonl");
        let log = GatewayLog::open(&path).unwrap();

        log.write("{\"before\":true}\n".to_string());
        assert_eq!(
            log.flush_and_stop(Duration::from_secs(30)),
            LogFlush::Drained
        );
        // A request that finishes after shutdown began must not panic the
        // gateway, and must not silently look like it was recorded.
        log.write("{\"after\":true}\n".to_string());

        assert_eq!(read_lines_now(&path), vec!["{\"before\":true}".to_string()]);
        assert_eq!(log.dropped_records.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn flush_outcome_is_self_consistent_under_an_expired_deadline() {
        // A zero deadline races the writer by construction, so this asserts the
        // invariant that holds either way rather than pinning one arm and
        // becoming flaky: whatever is reported pending was accepted and not yet
        // written. It is an upper bound on loss, not a measurement -- the
        // writer keeps draining after the wait gives up.
        const RECORDS: u64 = 500;
        let dir = tempfile::tempdir().unwrap();
        let log = GatewayLog::open(dir.path().join("audit.jsonl")).unwrap();
        for index in 0..RECORDS {
            log.write(format!("{{\"n\":{index}}}\n"));
        }

        let queued = log.queued_records.load(Ordering::Relaxed);
        match log.flush_and_stop(Duration::ZERO) {
            LogFlush::Drained => {}
            LogFlush::TimedOut {
                pending_at_deadline,
            } => {
                assert!(
                    pending_at_deadline <= queued,
                    "pending {pending_at_deadline} cannot exceed the {queued} accepted"
                );
            }
            LogFlush::AlreadyStopped => panic!("a first flush cannot report AlreadyStopped"),
        }
        assert_eq!(queued, RECORDS, "no record should have been rejected");
    }

    #[test]
    fn concurrent_flushes_still_drain_exactly_once() {
        // Pins the invariant that exactly one caller owns the drain and the
        // rest are no-ops. It does not claim to reproduce the interleaving that
        // motivated putting both halves behind one lock -- that window is a few
        // instructions wide and would not fail reliably -- but the invariant it
        // asserts is the one that interleaving violates, and it also holds
        // shutdown idempotency closed against simpler regressions.
        const FLUSHERS: usize = 16;
        const RECORDS: usize = 200;
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("audit.jsonl");
        let log = GatewayLog::open(&path).unwrap();
        for index in 0..RECORDS {
            log.write(format!("{{\"n\":{index}}}\n"));
        }

        let outcomes: Vec<LogFlush> = std::thread::scope(|scope| {
            let handles: Vec<_> = (0..FLUSHERS)
                .map(|_| {
                    let log = Arc::clone(&log);
                    scope.spawn(move || log.flush_and_stop(Duration::from_secs(30)))
                })
                .collect();
            handles.into_iter().map(|h| h.join().unwrap()).collect()
        });

        let drained = outcomes
            .iter()
            .filter(|outcome| **outcome == LogFlush::Drained)
            .count();
        let already = outcomes
            .iter()
            .filter(|outcome| **outcome == LogFlush::AlreadyStopped)
            .count();
        assert_eq!(
            drained, 1,
            "exactly one caller owns the drain: {outcomes:?}"
        );
        assert_eq!(already, FLUSHERS - 1, "every other caller is a no-op");
        // The caller that reported Drained really did wait for it.
        assert_eq!(read_lines_now(&path).len(), RECORDS);
    }
}
