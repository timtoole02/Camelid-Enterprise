//! PostgreSQL-backed platform state shared by every gateway replica.
//!
//! This crate owns the *platform* store described in
//! `docs/architecture/platform-datastore.md`: state that every gateway pod must
//! agree on. It deliberately does not own principals, organizations or tokens —
//! those stay in `camelid-enterprise-identity` until that separate question is
//! settled.
//!
//! It holds two things.
//!
//! **Shared per-organization request quota.** The gateway's in-process
//! `OrgQuota` counter is correct for one process and wrong for a replica set —
//! two pods each admit `limit` requests per window, so the deployed manifest's
//! two replicas admit up to `2 x limit` in a window and up to `4 x limit`
//! across a window boundary. Admission here is one atomic statement against one
//! row, so the limit is the deployment's, not the pod's.
//!
//! **Aggregated evidence.** The gateway writes an audit line and a usage line
//! per request, and each replica writes a serving receipt; all three are
//! per-pod JSONL files that nobody joins. [`PlatformStore::ingest_evidence`]
//! reads what those files contain into three tables keyed so they can be joined
//! on the gateway's `request_id`. Ingestion is deliberately *not* on the
//! request path: those writers are best-effort by design, and making a serving
//! request wait on this database would trade the property they were built to
//! protect for the aggregate they feed. It reads what survived and cannot
//! retroactively complete it.
//!
//! Three properties are load-bearing for quota and are asserted by the
//! integration tests in `tests/postgres_quota.rs`:
//!
//! - **Atomic.** Admission is a single autocommit statement whose `ON CONFLICT
//!   DO UPDATE ... WHERE` takes the row lock and re-evaluates the limit after
//!   the waiter wakes. No read-then-write, no explicit transaction, no deadlock
//!   class.
//! - **Globally aligned.** Windows are anchored to the database clock, not to
//!   a pod's first request, so every pod computes the same window start and
//!   therefore the same `Retry-After`. No pod's wall clock is consulted.
//! - **Fail-closed.** Every failure to reach or query the store is a refusal.
//!   There is no fallback to per-pod counting: silently degrading to the very
//!   behaviour this exists to replace is worse than a `503`.
//!
//! The database must belong to one gateway deployment. Quota state is keyed by
//! organization and window and by nothing else, so two deployments pointed at
//! one database share every counter — quietly, if they also agree on the
//! window. Give staging its own database.

use std::num::{NonZeroU32, NonZeroU64};
use std::path::{Path, PathBuf};
use std::sync::Once;
use std::time::Duration;

use deadpool_postgres::{Manager, ManagerConfig, Pool, PoolConfig, RecyclingMethod, Runtime};
use tokio_postgres::config::SslMode;
use tokio_postgres::NoTls;

/// Schema version this build understands. A store stamped higher was written by
/// a newer gateway; refusing it is the only safe reading, exactly as the
/// identity store refuses a `user_version` it does not know.
pub const PLATFORM_SCHEMA_VERSION: i32 = 2;

/// Serializes schema initialization and quota reconfiguration across pods.
/// Advisory locks need no table, which is what makes them usable *before* the
/// schema exists — the ordering problem `CREATE TABLE IF NOT EXISTS` cannot
/// solve, because concurrent creates of the same table still collide in the
/// system catalog.
const SCHEMA_LOCK_KEY: i64 = 0x0043_414d_454c_4944;

/// Distinct from [`SCHEMA_LOCK_KEY`] on purpose. Sharing one key coupled
/// housekeeping to startup: a sweep that held the lock made every *new* pod
/// block on it and then die of `statement_timeout`, reporting a timeout rather
/// than a lock.
const SWEEP_LOCK_KEY: i64 = 0x0043_414d_454c_5357;

/// How many elapsed windows of quota rows to keep before the sweeper removes
/// them. Rows are tiny and admission never reads an elapsed window, so this is
/// hygiene, not correctness; keeping a few makes an operator's post-hoc "was
/// this tenant throttled" query possible.
const RETENTION_WINDOWS: i64 = 3;

/// Upper bound on rows one `DELETE` may remove, so no single statement runs
/// long enough to matter.
const SWEEP_BATCH_ROWS: i64 = 10_000;

/// Lines sent to the database in one `INSERT`. A round trip per line is roughly
/// an eightieth of the throughput of a batched one, which put ingestion below
/// the rate a single busy gateway writes evidence.
const INGEST_BATCH_LINES: usize = 1_000;

/// Batches one sweep may run. One bounded batch per sweep could not keep up
/// with a deployment producing more than `SWEEP_BATCH_ROWS` organization
/// windows per window: the backlog would grow faster than it drained.
const SWEEP_MAX_BATCHES: usize = 32;

/// Admission. One statement, one row, no transaction.
///
/// The `WHERE` on the conflict path is what makes this exact: when the limit is
/// reached the row is locked but not updated and nothing is returned, so a
/// refusal cannot race an admission. `Retry-After` comes from the same `now()`
/// as the window arithmetic — `now()` is transaction time, so both CTEs see one
/// instant.
///
/// The limit is read from `quota_config` rather than bound from this process,
/// so it is the deployment's and not the pod's. A pod still running an older
/// limit therefore enforces the current one from the moment a newer pod records
/// it, instead of over-admitting for the rest of the window.
///
/// Both branches consult it. Guarding only the conflict path would leave the
/// insert — every organization's *first* request in every window — admitting
/// against no limit at all, so a missing configuration row would silently
/// become "one request per organization per window" instead of an error. The
/// row is returned as well, because a caller cannot otherwise tell a spent
/// window from a store that cannot say what the limit is.
const ADMIT_SQL: &str = "\
WITH t AS (
  SELECT floor(extract(epoch FROM now()) / $2::bigint)::bigint * $2::bigint AS window_start,
         extract(epoch FROM now()) AS now_epoch
), admitted AS (
  INSERT INTO quota_windows AS q (organization_id, window_start_epoch, request_count)
  SELECT $1::text, window_start, 1 FROM t
  WHERE (SELECT request_limit FROM quota_config) IS NOT NULL
  ON CONFLICT (organization_id, window_start_epoch)
  DO UPDATE SET request_count = q.request_count + 1
  WHERE q.request_count < (SELECT request_limit FROM quota_config)
  RETURNING q.request_count
)
SELECT (SELECT count(*) FROM admitted) = 1 AS admitted,
       (SELECT (window_start + $2::bigint - now_epoch) FROM t)::double precision
         AS retry_after_seconds,
       (SELECT request_limit FROM quota_config) AS request_limit";

const SWEEP_SQL: &str = "\
DELETE FROM quota_windows WHERE ctid IN (
  SELECT ctid FROM quota_windows
  WHERE window_start_epoch < floor(extract(epoch FROM now()))::bigint - $1::bigint
  LIMIT $2::bigint
)";

const SCHEMA_V1: &str = "\
CREATE TABLE platform_schema_version (
  singleton boolean PRIMARY KEY DEFAULT true CHECK (singleton),
  version   integer NOT NULL
);
CREATE TABLE quota_config (
  singleton      boolean PRIMARY KEY DEFAULT true CHECK (singleton),
  request_limit  bigint NOT NULL CHECK (request_limit > 0),
  window_seconds bigint NOT NULL CHECK (window_seconds > 0),
  updated_at     timestamptz NOT NULL DEFAULT now()
);
CREATE TABLE quota_windows (
  organization_id    text   NOT NULL,
  window_start_epoch bigint NOT NULL,
  request_count      bigint NOT NULL,
  PRIMARY KEY (organization_id, window_start_epoch)
);
-- The primary key leads with the organization, so the sweeper's predicate on
-- the window alone cannot use it and would sequentially scan a growing table.
CREATE INDEX quota_windows_window_start_epoch_idx
  ON quota_windows (window_start_epoch);
INSERT INTO platform_schema_version (version) VALUES (1);";

/// The evidence tables, named for the relations
/// `docs/architecture/service-separation.md` already joins on paper.
///
/// Each row keeps the record as `jsonb` and lifts out only what a query has to
/// filter or join on. The alternative — a column per field — would need a
/// migration every time a writer learns to say something new, and these writers
/// do: `posture` and `engine_sha256` joined the receipt in ADR 0004. A record
/// this store cannot fully describe is still evidence worth keeping.
///
/// `line_sha256` is the digest of the line *as written*, which is what makes
/// re-reading a file cheap to make idempotent. Note that it is not a checksum
/// of the stored `jsonb`: `jsonb` normalizes key order and whitespace, so the
/// digest ties a row to the bytes it came from rather than to its own column.
///
/// Identity by content cannot distinguish a line read twice from the same line
/// written twice, and one stream can produce the latter. A receipt from a
/// directly-reached replica has no `request_id`, so its only per-request fields
/// are `ts`, `method`, `path` and `status`; `ts` is `f64` seconds, whose spacing
/// at the current epoch is about 238ns. Two such requests that finish within one
/// of those and agree on path and status write one line, and the second is
/// counted as already present. The gateway's own streams cannot do this — a
/// 128-bit `request_id` is on every line. Removing it for receipts means giving
/// them something per-request of their own, which is a change to what a replica
/// writes and not to this schema.
///
/// Nothing here deletes evidence. `sweep` covers `quota_windows` only, and it is
/// deliberate that these tables have no retention: how long audit evidence is
/// kept is a policy this crate should not pick a default for. It becomes urgent
/// as soon as something ingests on a schedule, which nothing yet does.
///
/// A receipt's `request_id` is nullable because a replica reached directly, not
/// through a gateway, is never given one. Such a receipt still records what
/// served the request; it simply cannot be joined.
const SCHEMA_V2: &str = "\
CREATE TABLE gateway_audit (
  line_sha256  bytea PRIMARY KEY,
  request_id   text NOT NULL,
  organization text,
  ts           double precision NOT NULL,
  record       jsonb NOT NULL
);
CREATE INDEX gateway_audit_request_id_idx ON gateway_audit (request_id);
CREATE INDEX gateway_audit_organization_idx ON gateway_audit (organization);
CREATE TABLE gateway_usage (
  line_sha256  bytea PRIMARY KEY,
  request_id   text NOT NULL,
  organization text,
  ts           double precision NOT NULL,
  record       jsonb NOT NULL
);
CREATE INDEX gateway_usage_request_id_idx ON gateway_usage (request_id);
CREATE INDEX gateway_usage_organization_idx ON gateway_usage (organization);
-- No organization column: identity never reaches a replica, so a receipt
-- cannot name one without inventing it.
CREATE TABLE replica_receipt (
  line_sha256 bytea PRIMARY KEY,
  request_id  text,
  ts          double precision NOT NULL,
  record      jsonb NOT NULL
);
CREATE INDEX replica_receipt_request_id_idx ON replica_receipt (request_id);
UPDATE platform_schema_version SET version = 2;";

/// How a gateway reaches the platform store.
///
/// The URL is a credential. It is read from the environment by preference and
/// is never logged: [`PlatformStoreConfig::redacted_target`] is what callers
/// should print.
#[derive(Clone, Debug)]
pub struct PlatformStoreConfig {
    /// `postgresql://user:password@host:port/database`.
    pub url: String,
    /// PEM file holding the certificate authority that signed the database's
    /// server certificate. `Some` turns TLS on and makes it mandatory; `None`
    /// leaves the connection in cleartext.
    pub ca_file: Option<PathBuf>,
    /// Pool size per gateway process. Measured, not guessed: admissions for one
    /// organization serialize on one row, so 32 and 64 connections were both
    /// *slower* than 8 at 256 concurrent admissions. Raising this does not
    /// raise a single tenant's admission rate.
    pub max_connections: usize,
    /// How long a request may wait for a pooled connection before the store is
    /// declared unavailable. This is the bound that keeps a slow database from
    /// becoming an unbounded queue of requests waiting ahead of the gateway's
    /// admission semaphore.
    pub acquire_timeout: Duration,
    /// Server-side `statement_timeout` applied to every pooled connection, so a
    /// query that hangs is refused rather than held.
    pub statement_timeout: Duration,
}

impl PlatformStoreConfig {
    pub const DEFAULT_MAX_CONNECTIONS: usize = 8;
    pub const DEFAULT_ACQUIRE_TIMEOUT: Duration = Duration::from_millis(500);
    pub const DEFAULT_STATEMENT_TIMEOUT: Duration = Duration::from_millis(1_000);

    pub fn new(url: String) -> Self {
        Self {
            url,
            ca_file: None,
            max_connections: Self::DEFAULT_MAX_CONNECTIONS,
            acquire_timeout: Self::DEFAULT_ACQUIRE_TIMEOUT,
            statement_timeout: Self::DEFAULT_STATEMENT_TIMEOUT,
        }
    }

    /// `host:port/database` — never the user or password.
    pub fn redacted_target(&self) -> String {
        let config: Result<tokio_postgres::Config, _> = self.url.parse();
        let Ok(config) = config else {
            return "<unparseable platform database url>".to_string();
        };
        let host = match config.get_hosts().first() {
            Some(tokio_postgres::config::Host::Tcp(host)) => host.clone(),
            #[cfg(unix)]
            Some(tokio_postgres::config::Host::Unix(path)) => path.display().to_string(),
            _ => "<unknown host>".to_string(),
        };
        let port = config.get_ports().first().copied().unwrap_or(5432);
        let database = config.get_dbname().unwrap_or("<default database>");
        format!("{host}:{port}/{database}")
    }
}

#[derive(Debug)]
#[non_exhaustive]
pub enum PlatformStoreError {
    /// The configuration cannot produce a usable connection at all.
    Config(String),
    /// The certificate authority could not be read or parsed.
    Tls(String),
    /// The store could not be reached, or a statement failed against it.
    Database(String),
    /// The store was written by a newer gateway than this one.
    UnsupportedSchemaVersion { found: i32, supported: i32 },
    /// Another pod configured a different window length. Pods that disagree
    /// about the window write to different rows and therefore enforce the
    /// *sum* of their limits — precisely the defect a shared counter exists to
    /// remove — so this is fatal rather than a warning. A limit that differs is
    /// not: the limit lives in the store and every pod reads it there, so the
    /// deployment converges on the most recently recorded one.
    QuotaWindowMismatch { configured: u64, stored: i64 },
}

impl std::fmt::Display for PlatformStoreError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Config(message) => write!(f, "platform database configuration: {message}"),
            Self::Tls(message) => write!(f, "platform database TLS: {message}"),
            Self::Database(message) => write!(f, "platform database: {message}"),
            Self::UnsupportedSchemaVersion { found, supported } => write!(
                f,
                "platform store schema version {found} is newer than this gateway supports \
                 ({supported}); run the newer gateway or restore the store"
            ),
            Self::QuotaWindowMismatch { configured, stored } => write!(
                f,
                "platform store quota window is {stored}s but this gateway is configured for \
                 {configured}s; every replica must use the same \
                 --org-request-quota-window-seconds or the shared limit is not shared. To \
                 change it, roll out the new value with --reconfigure-quota-window set on the \
                 replicas, which rewrites the stored window and discards the counters measured \
                 against the old one."
            ),
        }
    }
}

impl std::error::Error for PlatformStoreError {}

/// `tokio_postgres::Error` displays as a bare "db error"; everything an
/// operator needs is in its source chain.
fn database_error(error: impl std::error::Error) -> PlatformStoreError {
    let mut message = error.to_string();
    let mut source = error.source();
    while let Some(cause) = source {
        message.push_str(": ");
        message.push_str(&cause.to_string());
        source = cause.source();
    }
    PlatformStoreError::Database(message)
}

/// Why a request was not admitted.
#[derive(Debug)]
pub enum QuotaRefusal {
    /// The organization has spent its window. `retry_after` is the remaining
    /// time in that window, measured by the database, so every pod answers the
    /// same number.
    Exceeded { retry_after: Duration },
    /// The quota could not be evaluated. Deliberately distinct from
    /// [`Self::Exceeded`]: one is the tenant's fault and one is the
    /// deployment's, they map to different status codes, and an operator
    /// reading the audit log must be able to tell them apart.
    Unavailable(PlatformStoreError),
}

/// What [`PlatformStore::configure_quota`] found already in the store.
#[derive(Debug, Eq, PartialEq)]
pub enum QuotaConfiguration {
    Initialized,
    Unchanged,
    LimitChanged { previous: u64 },
    WindowChanged { previous: u64 },
}

/// What to do about a stored quota window that differs from this pod's.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum QuotaWindowChange {
    /// Refuse to start. The right default: a pod that disagrees about the
    /// window silently doubles the deployment's effective limit.
    Refuse,
    /// Rewrite the stored window, discarding counters measured against the old
    /// one. An operator has to ask for this, because during the rollout that
    /// carries it the replicas still running the previous window are the split
    /// counter this refuses to become by accident.
    Reconfigure,
}

/// What one sweep did.
#[derive(Debug, Eq, PartialEq)]
pub enum SweepOutcome {
    /// Another pod holds the sweep lock. Exactly one pod sweeps at a time.
    SkippedLockHeld,
    Deleted(u64),
}

/// Which append-only JSONL stream a line was read from.
///
/// The three are separate relations rather than one table with a discriminator
/// because they answer different questions and only one of them can be missing
/// a correlation id. A gateway writes the first two; every replica writes the
/// third.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EvidenceStream {
    /// `--audit-log`: one line per handled request, written once the response
    /// status is known.
    GatewayAudit,
    /// `--usage-log`: one terminal line per request, written when the response
    /// body finishes, errors, or is dropped.
    GatewayUsage,
    /// `--serving-receipts`: one line per request, naming the replica that
    /// served it.
    ReplicaReceipt,
}

impl EvidenceStream {
    fn table(self) -> &'static str {
        match self {
            Self::GatewayAudit => "gateway_audit",
            Self::GatewayUsage => "gateway_usage",
            Self::ReplicaReceipt => "replica_receipt",
        }
    }

    /// Whether a line without a `request_id` is a record or a defect. Only a
    /// receipt may lack one, and only because a replica reached directly was
    /// never given one.
    fn correlation_is_optional(self) -> bool {
        matches!(self, Self::ReplicaReceipt)
    }

    /// `sha256(...)` is computed by the database over the same parameter that
    /// is cast to `jsonb`, so the digest cannot drift from the row it
    /// identifies. `ON CONFLICT DO NOTHING` is what makes re-reading a file
    /// safe; `RETURNING` is how the caller learns which happened, since a
    /// conflict returns no row.
    fn insert_sql(self) -> &'static str {
        match self {
            Self::GatewayAudit => {
                "INSERT INTO gateway_audit (line_sha256, request_id, organization, ts, record) \
                 VALUES (sha256(convert_to($1, 'UTF8')), $2, $3, $4, $1::jsonb) \
                 ON CONFLICT (line_sha256) DO NOTHING RETURNING true"
            }
            Self::GatewayUsage => {
                "INSERT INTO gateway_usage (line_sha256, request_id, organization, ts, record) \
                 VALUES (sha256(convert_to($1, 'UTF8')), $2, $3, $4, $1::jsonb) \
                 ON CONFLICT (line_sha256) DO NOTHING RETURNING true"
            }
            Self::ReplicaReceipt => {
                "INSERT INTO replica_receipt (line_sha256, request_id, ts, record) \
                 VALUES (sha256(convert_to($1, 'UTF8')), $2, $3, $1::jsonb) \
                 ON CONFLICT (line_sha256) DO NOTHING RETURNING true"
            }
        }
    }

    /// The same statement over parallel arrays instead of one row.
    ///
    /// A round trip per line put ingestion an order of magnitude below what one
    /// gateway emits, so a file could be read more slowly than it was written
    /// and never catch up. Rows affected is the count actually inserted, so the
    /// caller still learns how many of a batch were already there without a
    /// second query.
    fn insert_batch_sql(self) -> &'static str {
        match self {
            Self::GatewayAudit => {
                "INSERT INTO gateway_audit (line_sha256, request_id, organization, ts, record) \
                 SELECT sha256(convert_to(l, 'UTF8')), r, o, t, l::jsonb \
                 FROM unnest($1::text[], $2::text[], $3::text[], $4::float8[]) AS s(l, r, o, t) \
                 ON CONFLICT (line_sha256) DO NOTHING"
            }
            Self::GatewayUsage => {
                "INSERT INTO gateway_usage (line_sha256, request_id, organization, ts, record) \
                 SELECT sha256(convert_to(l, 'UTF8')), r, o, t, l::jsonb \
                 FROM unnest($1::text[], $2::text[], $3::text[], $4::float8[]) AS s(l, r, o, t) \
                 ON CONFLICT (line_sha256) DO NOTHING"
            }
            Self::ReplicaReceipt => {
                "INSERT INTO replica_receipt (line_sha256, request_id, ts, record) \
                 SELECT sha256(convert_to(l, 'UTF8')), r, t, l::jsonb \
                 FROM unnest($1::text[], $2::text[], $3::float8[]) AS s(l, r, t) \
                 ON CONFLICT (line_sha256) DO NOTHING"
            }
        }
    }
}

/// Whether the database refused this *record* or failed to answer at all.
///
/// Class 22 is SQL's data exception: the value cannot be represented. That is a
/// property of the line, so it makes the line unreadable rather than the store
/// unavailable, and it is the only way the two can be told apart from here. A
/// client-side error carries no code and is never a verdict on the record.
fn is_unstorable_record(error: &tokio_postgres::Error) -> bool {
    error
        .code()
        .is_some_and(|state| state.code().starts_with("22"))
}

/// What became of one line.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Ingested {
    Stored,
    /// A byte-identical line was already stored. Reading a file twice is an
    /// expected way to run this, not an error.
    AlreadyPresent,
    /// The line is not a record this store can hold, either because it does not
    /// parse or because the database refused the value. Reported rather than
    /// returned as an error because the last line of an append-only log written
    /// by a process that was killed is routinely half a record, and one such
    /// line must not cost the operator the whole file behind it.
    ///
    /// The two conditions are one variant on purpose: `serde_json` and `jsonb`
    /// do not agree on what a record is — a `\u0000` escape is a string to one
    /// and unstorable to the other — and a caller that has to retry cannot use
    /// the distinction, because neither will ever succeed.
    Unreadable(String),
}

/// What became of a whole file.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct IngestSummary {
    pub stored: u64,
    pub already_present: u64,
    pub unreadable: u64,
}

impl IngestSummary {
    fn count_unreadable(&mut self, stream: EvidenceStream, why: &str) {
        self.unreadable += 1;
        tracing::warn!(
            target: "camelid_platform_store",
            table = stream.table(),
            why,
            "skipped an unreadable evidence line"
        );
    }
}

/// The fields lifted out of a record so rows can be joined and filtered
/// without opening the `jsonb` on every query.
struct EvidenceKeys {
    request_id: Option<String>,
    organization: Option<String>,
    ts: f64,
}

/// Reads the join key and timestamp out of one line, or says why it is not a
/// record. Deliberately strict about only those: everything else is preserved
/// verbatim and interpreted by whoever queries it.
fn evidence_keys(stream: EvidenceStream, line: &str) -> Result<EvidenceKeys, String> {
    let value: serde_json::Value =
        serde_json::from_str(line).map_err(|error| format!("not JSON: {error}"))?;
    let serde_json::Value::Object(record) = value else {
        return Err("not a JSON object".to_string());
    };

    let ts = match record.get("ts").and_then(serde_json::Value::as_f64) {
        Some(ts) if ts.is_finite() => ts,
        _ => return Err("no finite numeric `ts`".to_string()),
    };

    let request_id = match record.get("request_id") {
        Some(serde_json::Value::String(id)) if !id.is_empty() => Some(id.clone()),
        Some(serde_json::Value::Null) | None if stream.correlation_is_optional() => None,
        _ => return Err("no non-empty string `request_id`".to_string()),
    };

    let organization = match record.get("organization") {
        Some(serde_json::Value::String(organization)) => Some(organization.clone()),
        _ => None,
    };

    Ok(EvidenceKeys {
        request_id,
        organization,
        ts,
    })
}

pub struct PlatformStore {
    pool: Pool,
}

impl PlatformStore {
    /// Connects, applies migrations, and verifies the schema version.
    ///
    /// Failing here is deliberate: a pod that cannot reach the store it needs
    /// should not pass readiness, and crash-loop backoff is the right retry.
    pub async fn connect(config: &PlatformStoreConfig) -> Result<Self, PlatformStoreError> {
        let store = Self {
            pool: build_pool(config)?,
        };
        store.migrate().await?;
        Ok(store)
    }

    async fn client(&self) -> Result<deadpool_postgres::Client, PlatformStoreError> {
        self.pool.get().await.map_err(database_error)
    }

    /// Applies every migration this build knows about, under the advisory lock.
    ///
    /// The lock is taken before the version is read, and the version is read
    /// inside the same transaction that applies the migration. PostgreSQL's DDL
    /// is transactional, so the whole chain either lands or does not — the
    /// step-at-a-time care the identity store's SQLite migrations need has no
    /// counterpart here.
    async fn migrate(&self) -> Result<(), PlatformStoreError> {
        let mut client = self.client().await?;
        let transaction = client.transaction().await.map_err(database_error)?;
        transaction
            .execute("SELECT pg_advisory_xact_lock($1)", &[&SCHEMA_LOCK_KEY])
            .await
            .map_err(database_error)?;

        let version = read_schema_version(&transaction).await?;
        if version > PLATFORM_SCHEMA_VERSION {
            return Err(PlatformStoreError::UnsupportedSchemaVersion {
                found: version,
                supported: PLATFORM_SCHEMA_VERSION,
            });
        }
        if version < 1 {
            transaction
                .batch_execute(SCHEMA_V1)
                .await
                .map_err(database_error)?;
        }
        if version < 2 {
            transaction
                .batch_execute(SCHEMA_V2)
                .await
                .map_err(database_error)?;
        }
        transaction.commit().await.map_err(database_error)
    }

    /// Records this pod's quota configuration, or refuses to run against a
    /// store configured with a different window.
    pub async fn configure_quota(
        &self,
        limit: NonZeroU32,
        window_seconds: NonZeroU64,
        on_window_change: QuotaWindowChange,
    ) -> Result<QuotaConfiguration, PlatformStoreError> {
        let window = window_seconds_as_i64(window_seconds)?;
        let limit = i64::from(limit.get());
        let mut client = self.client().await?;
        let transaction = client.transaction().await.map_err(database_error)?;
        transaction
            .execute("SELECT pg_advisory_xact_lock($1)", &[&SCHEMA_LOCK_KEY])
            .await
            .map_err(database_error)?;

        let existing = transaction
            .query_opt(
                "SELECT request_limit, window_seconds FROM quota_config LIMIT 1",
                &[],
            )
            .await
            .map_err(database_error)?;

        let outcome = match existing {
            None => {
                transaction
                    .execute(
                        "INSERT INTO quota_config (request_limit, window_seconds) VALUES ($1, $2)",
                        &[&limit, &window],
                    )
                    .await
                    .map_err(database_error)?;
                QuotaConfiguration::Initialized
            }
            Some(row) => {
                let stored_limit: i64 = row.get(0);
                let stored_window: i64 = row.get(1);
                if stored_window != window {
                    if on_window_change == QuotaWindowChange::Refuse {
                        return Err(PlatformStoreError::QuotaWindowMismatch {
                            configured: window_seconds.get(),
                            stored: stored_window,
                        });
                    }
                    transaction
                        .execute(
                            "UPDATE quota_config \
                             SET request_limit = $1, window_seconds = $2, updated_at = now()",
                            &[&limit, &window],
                        )
                        .await
                        .map_err(database_error)?;
                    // Every stored counter was measured against window starts
                    // the new length does not produce. Keeping them would leave
                    // rows no admission can ever reach again.
                    transaction
                        .execute("DELETE FROM quota_windows", &[])
                        .await
                        .map_err(database_error)?;
                    QuotaConfiguration::WindowChanged {
                        previous: stored_window.unsigned_abs(),
                    }
                } else if stored_limit == limit {
                    QuotaConfiguration::Unchanged
                } else {
                    transaction
                        .execute(
                            "UPDATE quota_config SET request_limit = $1, updated_at = now()",
                            &[&limit],
                        )
                        .await
                        .map_err(database_error)?;
                    QuotaConfiguration::LimitChanged {
                        previous: stored_limit.unsigned_abs(),
                    }
                }
            }
        };
        transaction.commit().await.map_err(database_error)?;
        Ok(outcome)
    }

    /// Deletes quota rows for windows that ended more than a few windows ago,
    /// if no other pod is already doing it.
    ///
    /// `pg_try_advisory_xact_lock` rather than the session-scoped form: a
    /// session lock outlives a cancelled or panicked sweep, because a pooled
    /// connection is recycled without resetting session state, and a leaked one
    /// is invisible to the pod that leaked it. A transaction lock is released
    /// by commit, rollback, drop and cancellation alike. `try` rather than
    /// blocking, because a pod that finds a sweep in progress has nothing to
    /// wait for.
    pub async fn sweep(
        &self,
        window_seconds: NonZeroU64,
    ) -> Result<SweepOutcome, PlatformStoreError> {
        let retention = window_seconds_as_i64(window_seconds)?.saturating_mul(RETENTION_WINDOWS);
        let mut client = self.client().await?;
        let transaction = client.transaction().await.map_err(database_error)?;
        let acquired: bool = transaction
            .query_one("SELECT pg_try_advisory_xact_lock($1)", &[&SWEEP_LOCK_KEY])
            .await
            .map_err(database_error)?
            .get(0);
        if !acquired {
            return Ok(SweepOutcome::SkippedLockHeld);
        }

        let mut deleted = 0;
        for _ in 0..SWEEP_MAX_BATCHES {
            let batch = transaction
                .execute(SWEEP_SQL, &[&retention, &SWEEP_BATCH_ROWS])
                .await
                .map_err(database_error)?;
            deleted += batch;
            if batch < SWEEP_BATCH_ROWS as u64 {
                break;
            }
        }
        transaction.commit().await.map_err(database_error)?;
        Ok(SweepOutcome::Deleted(deleted))
    }

    /// Stores one JSONL line of evidence.
    ///
    /// A line already present is reported, not an error: the intended way to
    /// run this is repeatedly over a growing file, and re-reading one must
    /// converge rather than duplicate. A line that is not a record is reported
    /// too, for the same reason the caller should not have to care — see
    /// [`Ingested::Unreadable`].
    pub async fn ingest_evidence_line(
        &self,
        stream: EvidenceStream,
        line: &str,
    ) -> Result<Ingested, PlatformStoreError> {
        let keys = match evidence_keys(stream, line) {
            Ok(keys) => keys,
            Err(why) => return Ok(Ingested::Unreadable(why)),
        };
        self.insert_evidence(stream, line, &keys).await
    }

    async fn insert_evidence(
        &self,
        stream: EvidenceStream,
        line: &str,
        keys: &EvidenceKeys,
    ) -> Result<Ingested, PlatformStoreError> {
        let client = self.client().await?;
        let stored = match stream {
            EvidenceStream::GatewayAudit | EvidenceStream::GatewayUsage => {
                client
                    .query_opt(
                        stream.insert_sql(),
                        &[&line, &keys.request_id, &keys.organization, &keys.ts],
                    )
                    .await
            }
            EvidenceStream::ReplicaReceipt => {
                client
                    .query_opt(stream.insert_sql(), &[&line, &keys.request_id, &keys.ts])
                    .await
            }
        };
        match stored {
            Ok(Some(_)) => Ok(Ingested::Stored),
            Ok(None) => Ok(Ingested::AlreadyPresent),
            Err(error) if is_unstorable_record(&error) => {
                Ok(Ingested::Unreadable(error.to_string()))
            }
            Err(error) => Err(database_error(error)),
        }
    }

    /// Stores every line of one evidence file.
    ///
    /// Takes the file's contents rather than a path so this crate stays out of
    /// the business of finding, rotating and following logs, which is where the
    /// deployment-shaped decisions live.
    pub async fn ingest_evidence(
        &self,
        stream: EvidenceStream,
        contents: &str,
    ) -> Result<IngestSummary, PlatformStoreError> {
        let mut summary = IngestSummary::default();
        let mut batch: Vec<(&str, EvidenceKeys)> = Vec::new();
        for line in contents.lines() {
            if line.trim().is_empty() {
                continue;
            }
            match evidence_keys(stream, line) {
                Ok(keys) => batch.push((line, keys)),
                Err(why) => summary.count_unreadable(stream, &why),
            }
            if batch.len() == INGEST_BATCH_LINES {
                self.ingest_batch(stream, &batch, &mut summary).await?;
                batch.clear();
            }
        }
        if !batch.is_empty() {
            self.ingest_batch(stream, &batch, &mut summary).await?;
        }
        Ok(summary)
    }

    /// One statement for up to [`INGEST_BATCH_LINES`] lines, falling back to one
    /// statement per line only when the batch is refused.
    ///
    /// The fallback is what keeps a record the database will not take from
    /// costing the whole batch. Retrying line by line is expensive, and it is
    /// supposed to be: it happens once per batch that contains such a line, and
    /// the alternative is discarding lines that are perfectly storable.
    async fn ingest_batch(
        &self,
        stream: EvidenceStream,
        batch: &[(&str, EvidenceKeys)],
        summary: &mut IngestSummary,
    ) -> Result<(), PlatformStoreError> {
        let lines: Vec<&str> = batch.iter().map(|(line, _)| *line).collect();
        let request_ids: Vec<Option<&str>> = batch
            .iter()
            .map(|(_, keys)| keys.request_id.as_deref())
            .collect();
        let timestamps: Vec<f64> = batch.iter().map(|(_, keys)| keys.ts).collect();

        let client = self.client().await?;
        let inserted = match stream {
            EvidenceStream::GatewayAudit | EvidenceStream::GatewayUsage => {
                let organizations: Vec<Option<&str>> = batch
                    .iter()
                    .map(|(_, keys)| keys.organization.as_deref())
                    .collect();
                client
                    .execute(
                        stream.insert_batch_sql(),
                        &[&lines, &request_ids, &organizations, &timestamps],
                    )
                    .await
            }
            EvidenceStream::ReplicaReceipt => {
                client
                    .execute(
                        stream.insert_batch_sql(),
                        &[&lines, &request_ids, &timestamps],
                    )
                    .await
            }
        };
        drop(client);

        match inserted {
            Ok(stored) => {
                summary.stored += stored;
                summary.already_present += batch.len() as u64 - stored;
                Ok(())
            }
            Err(error) if is_unstorable_record(&error) => {
                for (line, keys) in batch {
                    match self.insert_evidence(stream, line, keys).await? {
                        Ingested::Stored => summary.stored += 1,
                        Ingested::AlreadyPresent => summary.already_present += 1,
                        Ingested::Unreadable(why) => summary.count_unreadable(stream, &why),
                    }
                }
                Ok(())
            }
            Err(error) => Err(database_error(error)),
        }
    }
}

/// The gateway's shared quota: a [`PlatformStore`] plus the window it was
/// started with. The *limit* is deliberately not held here — it lives in the
/// store, so a pod cannot enforce one the deployment has moved on from.
pub struct PlatformQuota {
    store: PlatformStore,
    window_seconds: i64,
    window: NonZeroU64,
}

impl PlatformQuota {
    /// Connects, migrates, and reconciles this pod's quota configuration with
    /// whatever the store already holds, refusing a stored window that differs
    /// from this pod's.
    pub async fn connect(
        config: &PlatformStoreConfig,
        limit: NonZeroU32,
        window_seconds: NonZeroU64,
    ) -> Result<(Self, QuotaConfiguration), PlatformStoreError> {
        Self::open(config, limit, window_seconds, QuotaWindowChange::Refuse).await
    }

    /// As [`Self::connect`], but rewrites a stored window that differs instead
    /// of refusing it. Spelled out rather than passed as a flag because it is
    /// the destructive one: it discards every counter measured against the old
    /// window, and the replicas still running that window are, until they
    /// restart, the split counter [`Self::connect`] refuses to become.
    pub async fn connect_reconfiguring_window(
        config: &PlatformStoreConfig,
        limit: NonZeroU32,
        window_seconds: NonZeroU64,
    ) -> Result<(Self, QuotaConfiguration), PlatformStoreError> {
        Self::open(
            config,
            limit,
            window_seconds,
            QuotaWindowChange::Reconfigure,
        )
        .await
    }

    async fn open(
        config: &PlatformStoreConfig,
        limit: NonZeroU32,
        window_seconds: NonZeroU64,
        on_window_change: QuotaWindowChange,
    ) -> Result<(Self, QuotaConfiguration), PlatformStoreError> {
        let store = PlatformStore::connect(config).await?;
        let configuration = store
            .configure_quota(limit, window_seconds, on_window_change)
            .await?;
        Ok((
            Self {
                store,
                window_seconds: window_seconds_as_i64(window_seconds)?,
                window: window_seconds,
            },
            configuration,
        ))
    }

    pub fn window(&self) -> NonZeroU64 {
        self.window
    }

    /// Charges one request to `organization`.
    ///
    /// The organization id is bound as a parameter and never interpolated, so
    /// an id shaped like SQL stays data.
    pub async fn admit(&self, organization: &str) -> Result<(), QuotaRefusal> {
        let client = self
            .store
            .client()
            .await
            .map_err(QuotaRefusal::Unavailable)?;
        let row = client
            .query_one(ADMIT_SQL, &[&organization, &self.window_seconds])
            .await
            .map_err(|error| QuotaRefusal::Unavailable(database_error(error)))?;
        let admitted: bool = row.get(0);
        if admitted {
            return Ok(());
        }
        let configured_limit: Option<i64> = row.get(2);
        if configured_limit.is_none() {
            return Err(QuotaRefusal::Unavailable(PlatformStoreError::Database(
                "the platform store has no quota configuration row, so no limit can be \
                 enforced; restart a gateway against it to record one"
                    .to_string(),
            )));
        }
        let retry_after_seconds: f64 = row.get(1);
        Err(QuotaRefusal::Exceeded {
            retry_after: Duration::from_secs_f64(retry_after_seconds.clamp(0.0, u32::MAX as f64)),
        })
    }

    pub async fn sweep(&self) -> Result<SweepOutcome, PlatformStoreError> {
        self.store.sweep(self.window).await
    }
}

async fn read_schema_version(
    transaction: &tokio_postgres::Transaction<'_>,
) -> Result<i32, PlatformStoreError> {
    let present: bool = transaction
        .query_one(
            "SELECT to_regclass('platform_schema_version') IS NOT NULL",
            &[],
        )
        .await
        .map_err(database_error)?
        .get(0);
    if !present {
        return Ok(0);
    }
    Ok(transaction
        .query_one(
            "SELECT coalesce(max(version), 0) FROM platform_schema_version",
            &[],
        )
        .await
        .map_err(database_error)?
        .get(0))
}

fn window_seconds_as_i64(window_seconds: NonZeroU64) -> Result<i64, PlatformStoreError> {
    i64::try_from(window_seconds.get()).map_err(|_| {
        PlatformStoreError::Config(format!(
            "quota window of {}s exceeds what the store can hold",
            window_seconds.get()
        ))
    })
}

fn build_pool(config: &PlatformStoreConfig) -> Result<Pool, PlatformStoreError> {
    let mut pg_config: tokio_postgres::Config = config
        .url
        .parse()
        .map_err(|error| PlatformStoreError::Config(format!("{error}")))?;

    // Applied server-side per connection rather than per query: a hung
    // statement must not be able to hold a pooled connection open.
    let statement_timeout = format!(
        "-c statement_timeout={}",
        config.statement_timeout.as_millis().max(1)
    );
    let options = match pg_config.get_options() {
        Some(existing) => format!("{existing} {statement_timeout}"),
        None => statement_timeout,
    };
    pg_config.options(&options);

    let manager_config = ManagerConfig {
        recycling_method: RecyclingMethod::Fast,
    };
    // Whether the connection is encrypted is decided by the CA file, because
    // rustls cannot verify anything without one. What the URL asked for still
    // has to be honoured or refused, never quietly overridden: an operator who
    // wrote `sslmode=require` and got cleartext learns nothing from a warning
    // that says the connection is cleartext.
    let requested = pg_config.get_ssl_mode();
    let manager = match &config.ca_file {
        Some(ca_file) => {
            if matches!(requested, SslMode::Disable) {
                return Err(PlatformStoreError::Config(
                    "the platform database url asks for sslmode=disable but a certificate \
                     authority was supplied; drop one of the two"
                        .to_string(),
                ));
            }
            // `require` and not `verify-full`: tokio-postgres has no such mode,
            // because verification belongs to the connector. rustls always
            // verifies the chain and the hostname, so this *is* full
            // verification and cannot be downgraded by the URL.
            pg_config.ssl_mode(SslMode::Require);
            Manager::from_config(pg_config, tls_connector(ca_file)?, manager_config)
        }
        None => {
            if matches!(requested, SslMode::Require) {
                return Err(PlatformStoreError::Config(
                    "the platform database url asks for sslmode=require but no certificate \
                     authority was supplied, and TLS without one cannot be verified; supply \
                     the CA that signed the database's server certificate"
                        .to_string(),
                ));
            }
            pg_config.ssl_mode(SslMode::Disable);
            Manager::from_config(pg_config, NoTls, manager_config)
        }
    };

    let mut pool_config = PoolConfig::new(config.max_connections.max(1));
    pool_config.timeouts.wait = Some(config.acquire_timeout);
    pool_config.timeouts.create = Some(config.acquire_timeout);
    Pool::builder(manager)
        .config(pool_config)
        // Required whenever timeouts are set; without it every checkout fails.
        .runtime(Runtime::Tokio1)
        .build()
        .map_err(|error| PlatformStoreError::Config(error.to_string()))
}

fn tls_connector(
    ca_file: &Path,
) -> Result<tokio_postgres_rustls::MakeRustlsConnect, PlatformStoreError> {
    install_crypto_provider();
    let pem = std::fs::File::open(ca_file).map_err(|error| {
        PlatformStoreError::Tls(format!("cannot read {}: {error}", ca_file.display()))
    })?;
    let mut reader = std::io::BufReader::new(pem);
    let mut roots = rustls::RootCertStore::empty();
    let mut certificates = 0usize;
    for certificate in rustls_pemfile::certs(&mut reader) {
        let certificate = certificate.map_err(|error| {
            PlatformStoreError::Tls(format!("{} is not valid PEM: {error}", ca_file.display()))
        })?;
        roots
            .add(certificate)
            .map_err(|error| PlatformStoreError::Tls(format!("{error}")))?;
        certificates += 1;
    }
    if certificates == 0 {
        return Err(PlatformStoreError::Tls(format!(
            "{} contains no certificates",
            ca_file.display()
        )));
    }
    let client_config = rustls::ClientConfig::builder()
        .with_root_certificates(roots)
        .with_no_client_auth();
    Ok(tokio_postgres_rustls::MakeRustlsConnect::new(client_config))
}

fn install_crypto_provider() {
    static INSTALL: Once = Once::new();
    INSTALL.call_once(|| {
        // Fails only if something already installed one, which is equally fine.
        let _ = rustls::crypto::ring::default_provider().install_default();
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn redacted_target_never_carries_the_password() {
        let config = PlatformStoreConfig::new(
            "postgresql://camelid:hunter2@db.internal:6432/platform".to_string(),
        );
        let redacted = config.redacted_target();
        assert_eq!(redacted, "db.internal:6432/platform");
        assert!(!redacted.contains("hunter2"));
        assert!(!redacted.contains("camelid"));
    }

    #[test]
    fn redacted_target_survives_an_unparseable_url() {
        let config = PlatformStoreConfig::new("not a url".to_string());
        assert_eq!(
            config.redacted_target(),
            "<unparseable platform database url>"
        );
    }

    #[test]
    fn a_window_wider_than_the_store_can_hold_is_a_configuration_error() {
        let window = NonZeroU64::new(u64::MAX).unwrap();
        assert!(matches!(
            window_seconds_as_i64(window),
            Err(PlatformStoreError::Config(_))
        ));
    }

    #[test]
    fn defaults_come_from_the_measured_latency_curve() {
        let config = PlatformStoreConfig::new("postgresql://localhost/platform".to_string());
        assert_eq!(config.max_connections, 8);
        assert_eq!(config.acquire_timeout, Duration::from_millis(500));
        assert_eq!(config.statement_timeout, Duration::from_millis(1_000));
    }

    /// Forcing the mode down to cleartext because no CA was supplied handed an
    /// operator who asked for TLS an unencrypted connection, and said so only
    /// as a warning about cleartext that never mentioned being overruled.
    #[test]
    fn a_url_that_asks_for_tls_without_a_certificate_authority_is_refused() {
        let config = PlatformStoreConfig::new(
            "postgresql://camelid@db.internal/platform?sslmode=require".to_string(),
        );
        let Err(error) = build_pool(&config) else {
            panic!("sslmode=require without a CA cannot be honoured");
        };
        assert!(matches!(error, PlatformStoreError::Config(_)));
        assert!(error.to_string().contains("sslmode=require"));
    }

    #[test]
    fn a_certificate_authority_with_tls_switched_off_is_refused() {
        let mut config = PlatformStoreConfig::new(
            "postgresql://camelid@db.internal/platform?sslmode=disable".to_string(),
        );
        config.ca_file = Some(PathBuf::from("ca.pem"));
        assert!(matches!(
            build_pool(&config),
            Err(PlatformStoreError::Config(_))
        ));
    }

    #[test]
    fn a_url_that_says_nothing_about_tls_still_connects_in_cleartext() {
        let config = PlatformStoreConfig::new("postgresql://camelid@db.internal/platform".into());
        assert!(build_pool(&config).is_ok());
    }
}
