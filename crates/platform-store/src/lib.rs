//! PostgreSQL-backed platform state shared by every gateway replica.
//!
//! This crate owns the *platform* store described in
//! `docs/architecture/platform-datastore.md`: state that every gateway pod must
//! agree on. It deliberately does not own principals, organizations or tokens —
//! those stay in `camelid-enterprise-identity` until that separate question is
//! settled.
//!
//! Today it holds exactly one thing: the shared per-organization request quota.
//! The gateway's in-process `OrgQuota` counter is correct for one process and
//! wrong for a replica set — two pods each admit `limit` requests per window, so
//! the deployed manifest's two replicas admit up to `2 x limit` in a window and
//! up to `4 x limit` across a window boundary. Admission here is one atomic
//! statement against one row, so the limit is the deployment's, not the pod's.
//!
//! Three properties are load-bearing and are asserted by the integration tests
//! in `tests/postgres_quota.rs`:
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
pub const PLATFORM_SCHEMA_VERSION: i32 = 1;

/// Serializes schema initialization and quota reconfiguration across pods.
/// Advisory locks need no table, which is what makes them usable *before* the
/// schema exists — the ordering problem `CREATE TABLE IF NOT EXISTS` cannot
/// solve, because concurrent creates of the same table still collide in the
/// system catalog.
const ADVISORY_LOCK_KEY: i64 = 0x0043_414d_454c_4944;

/// How many elapsed windows of quota rows to keep before the sweeper removes
/// them. Rows are tiny and admission never reads an elapsed window, so this is
/// hygiene, not correctness; keeping a few makes an operator's post-hoc "was
/// this tenant throttled" query possible.
const RETENTION_WINDOWS: i64 = 3;

/// Upper bound on rows one sweep may delete, so the sweeper cannot turn into a
/// long transaction that blocks admission.
const SWEEP_BATCH_ROWS: i64 = 10_000;

/// Admission. One statement, one row, no transaction.
///
/// The `WHERE` on the conflict path is what makes this exact: when the limit is
/// reached the row is locked but not updated and nothing is returned, so a
/// refusal cannot race an admission. `Retry-After` comes from the same `now()`
/// as the window arithmetic — `now()` is transaction time, so both CTEs see one
/// instant.
const ADMIT_SQL: &str = "\
WITH t AS (
  SELECT floor(extract(epoch FROM now()) / $2::bigint)::bigint * $2::bigint AS window_start,
         extract(epoch FROM now()) AS now_epoch
), admitted AS (
  INSERT INTO quota_windows AS q (organization_id, window_start_epoch, request_count)
  SELECT $1::text, window_start, 1 FROM t
  ON CONFLICT (organization_id, window_start_epoch)
  DO UPDATE SET request_count = q.request_count + 1
  WHERE q.request_count < $3::bigint
  RETURNING q.request_count
)
SELECT (SELECT count(*) FROM admitted) = 1 AS admitted,
       (SELECT (window_start + $2::bigint - now_epoch) FROM t)::double precision
         AS retry_after_seconds";

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
INSERT INTO platform_schema_version (version) VALUES (1);";

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
    /// not: pods sharing a row converge on the largest configured limit, which
    /// is a degraded rollout, not a broken one.
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
                 --org-request-quota-window-seconds or the shared limit is not shared"
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
}

/// What one sweep did.
#[derive(Debug, Eq, PartialEq)]
pub enum SweepOutcome {
    /// Another pod holds the sweep lock. Exactly one pod sweeps at a time.
    SkippedLockHeld,
    Deleted(u64),
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
            .execute("SELECT pg_advisory_xact_lock($1)", &[&ADVISORY_LOCK_KEY])
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
        transaction.commit().await.map_err(database_error)
    }

    /// Records this pod's quota configuration, or refuses to run against a
    /// store configured with a different window.
    pub async fn configure_quota(
        &self,
        limit: NonZeroU32,
        window_seconds: NonZeroU64,
    ) -> Result<QuotaConfiguration, PlatformStoreError> {
        let window = window_seconds_as_i64(window_seconds)?;
        let limit = i64::from(limit.get());
        let mut client = self.client().await?;
        let transaction = client.transaction().await.map_err(database_error)?;
        transaction
            .execute("SELECT pg_advisory_xact_lock($1)", &[&ADVISORY_LOCK_KEY])
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
                    return Err(PlatformStoreError::QuotaWindowMismatch {
                        configured: window_seconds.get(),
                        stored: stored_window,
                    });
                }
                if stored_limit == limit {
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
    /// `pg_try_advisory_lock` rather than the blocking form: a pod that finds
    /// the sweep in progress has nothing to wait for.
    pub async fn sweep(
        &self,
        window_seconds: NonZeroU64,
    ) -> Result<SweepOutcome, PlatformStoreError> {
        let window = window_seconds_as_i64(window_seconds)?;
        let client = self.client().await?;
        let acquired: bool = client
            .query_one("SELECT pg_try_advisory_lock($1)", &[&ADVISORY_LOCK_KEY])
            .await
            .map_err(database_error)?
            .get(0);
        if !acquired {
            return Ok(SweepOutcome::SkippedLockHeld);
        }
        let deleted = client
            .execute(
                SWEEP_SQL,
                &[&window.saturating_mul(RETENTION_WINDOWS), &SWEEP_BATCH_ROWS],
            )
            .await;
        client
            .execute("SELECT pg_advisory_unlock($1)", &[&ADVISORY_LOCK_KEY])
            .await
            .map_err(database_error)?;
        Ok(SweepOutcome::Deleted(deleted.map_err(database_error)?))
    }
}

/// The gateway's shared quota: a [`PlatformStore`] plus the limit and window it
/// was started with.
pub struct PlatformQuota {
    store: PlatformStore,
    limit: i64,
    window_seconds: i64,
    window: NonZeroU64,
}

impl PlatformQuota {
    /// Connects, migrates, and reconciles this pod's quota configuration with
    /// whatever the store already holds.
    pub async fn connect(
        config: &PlatformStoreConfig,
        limit: NonZeroU32,
        window_seconds: NonZeroU64,
    ) -> Result<(Self, QuotaConfiguration), PlatformStoreError> {
        let store = PlatformStore::connect(config).await?;
        let configuration = store.configure_quota(limit, window_seconds).await?;
        Ok((
            Self {
                store,
                limit: i64::from(limit.get()),
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
            .query_one(
                ADMIT_SQL,
                &[&organization, &self.window_seconds, &self.limit],
            )
            .await
            .map_err(|error| QuotaRefusal::Unavailable(database_error(error)))?;
        let admitted: bool = row.get(0);
        if admitted {
            return Ok(());
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
    let manager = match &config.ca_file {
        Some(ca_file) => {
            // `require` and not `verify-full`: tokio-postgres has no such mode,
            // because verification belongs to the connector. rustls always
            // verifies the chain and the hostname, so this *is* full
            // verification and cannot be downgraded by the URL.
            pg_config.ssl_mode(SslMode::Require);
            Manager::from_config(pg_config, tls_connector(ca_file)?, manager_config)
        }
        None => {
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
}
