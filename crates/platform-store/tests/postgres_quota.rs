//! Integration tests against a real PostgreSQL.
//!
//! The database harness, and the reason these tests skip without a URL, live in
//! `support`.

use std::num::{NonZeroU32, NonZeroU64};
use std::sync::Arc;
use std::time::Duration;

use camelid_enterprise_platform_store::{
    PlatformQuota, PlatformStore, PlatformStoreConfig, PlatformStoreError, QuotaConfiguration,
    QuotaRefusal, SweepOutcome, PLATFORM_SCHEMA_VERSION,
};

mod support;
use support::TestDatabase;

fn limit(value: u32) -> NonZeroU32 {
    NonZeroU32::new(value).unwrap()
}

fn window(value: u64) -> NonZeroU64 {
    NonZeroU64::new(value).unwrap()
}

/// Guards against this whole file quietly becoming a no-op.
#[tokio::test]
async fn postgres_tests_are_not_silently_skipped() {
    support::assert_not_silently_skipped();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn two_independently_constructed_stores_admit_exactly_the_shared_limit() {
    let Some(database) = TestDatabase::create().await else {
        return;
    };
    let config = database.config();
    // Two stores, each built from configuration alone, exactly as two gateway
    // pods build theirs. Nothing is shared in this process except the database.
    let (left, _) = PlatformQuota::connect(&config, limit(100), window(3600))
        .await
        .expect("first store");
    let (right, _) = PlatformQuota::connect(&config, limit(100), window(3600))
        .await
        .expect("second store");
    let left = Arc::new(left);
    let right = Arc::new(right);

    let mut tasks = Vec::new();
    for attempt in 0..400 {
        let quota = if attempt % 2 == 0 {
            Arc::clone(&left)
        } else {
            Arc::clone(&right)
        };
        tasks.push(tokio::spawn(async move { quota.admit("org_shared").await }));
    }

    let mut admitted = 0;
    let mut exceeded = 0;
    for task in tasks {
        match task.await.expect("join") {
            Ok(()) => admitted += 1,
            Err(QuotaRefusal::Exceeded { .. }) => exceeded += 1,
            Err(QuotaRefusal::Unavailable(error)) => panic!("unexpected refusal: {error}"),
        }
    }
    assert_eq!(
        admitted, 100,
        "the limit is the deployment's, not the pod's"
    );
    assert_eq!(exceeded, 300);

    let counted: i64 = database
        .client()
        .await
        .query_one(
            "SELECT request_count FROM quota_windows WHERE organization_id = 'org_shared'",
            &[],
        )
        .await
        .expect("counter")
        .get(0);
    assert_eq!(counted, 100, "no admission was lost or double counted");
    database.drop_database().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn organizations_are_isolated() {
    let Some(database) = TestDatabase::create().await else {
        return;
    };
    let (quota, _) = PlatformQuota::connect(&database.config(), limit(2), window(3600))
        .await
        .expect("store");

    assert!(quota.admit("org_a").await.is_ok());
    assert!(quota.admit("org_a").await.is_ok());
    assert!(matches!(
        quota.admit("org_a").await,
        Err(QuotaRefusal::Exceeded { .. })
    ));
    assert!(
        quota.admit("org_b").await.is_ok(),
        "one tenant's exhausted window must not spend another's"
    );
    database.drop_database().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_window_rollover_resets_the_counter_and_retry_after_stays_inside_the_window() {
    let Some(database) = TestDatabase::create().await else {
        return;
    };
    let (quota, _) = PlatformQuota::connect(&database.config(), limit(1), window(2))
        .await
        .expect("store");

    assert!(quota.admit("org_roll").await.is_ok());
    let Err(QuotaRefusal::Exceeded { retry_after }) = quota.admit("org_roll").await else {
        panic!("the second request in a one-request window must be refused");
    };
    assert!(
        retry_after > Duration::ZERO && retry_after <= Duration::from_secs(2),
        "retry_after {retry_after:?} must fall inside the window"
    );

    // The window is aligned to the database clock, so waiting out the refusal's
    // own answer is enough: it names the boundary every pod would name.
    tokio::time::sleep(retry_after + Duration::from_millis(250)).await;
    if let Err(refusal) = quota.admit("org_roll").await {
        panic!("the window named by retry_after must have rolled over: {refusal:?}");
    }

    let windows: i64 = database
        .client()
        .await
        .query_one(
            "SELECT count(*) FROM quota_windows WHERE organization_id = 'org_roll'",
            &[],
        )
        .await
        .expect("windows")
        .get(0);
    assert_eq!(
        windows, 2,
        "a rollover starts a new row, it does not mutate"
    );
    database.drop_database().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn window_starts_are_aligned_to_the_database_clock() {
    let Some(database) = TestDatabase::create().await else {
        return;
    };
    let (quota, _) = PlatformQuota::connect(&database.config(), limit(10), window(60))
        .await
        .expect("store");
    quota.admit("org_aligned").await.expect("admit");

    let start: i64 = database
        .client()
        .await
        .query_one(
            "SELECT window_start_epoch FROM quota_windows WHERE organization_id = 'org_aligned'",
            &[],
        )
        .await
        .expect("window start")
        .get(0);
    assert_eq!(
        start % 60,
        0,
        "an unaligned start would give each pod its own window and its own Retry-After"
    );
    database.drop_database().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn concurrent_stores_all_migrate() {
    let Some(database) = TestDatabase::create().await else {
        return;
    };
    let config = database.config();
    let mut tasks = Vec::new();
    for _ in 0..8 {
        let config = config.clone();
        tasks.push(tokio::spawn(async move {
            PlatformStore::connect(&config).await.map(|_| ())
        }));
    }
    let mut failures = Vec::new();
    for task in tasks {
        if let Err(error) = task.await.expect("join") {
            failures.push(error.to_string());
        }
    }
    assert!(
        failures.is_empty(),
        "every pod starting at once must migrate: {failures:?}"
    );
    database.drop_database().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_store_from_a_newer_gateway_is_refused() {
    let Some(database) = TestDatabase::create().await else {
        return;
    };
    PlatformStore::connect(&database.config())
        .await
        .expect("store");
    database
        .client()
        .await
        .execute("UPDATE platform_schema_version SET version = 99", &[])
        .await
        .expect("stamp a future version");

    let error = PlatformStore::connect(&database.config())
        .await
        .err()
        .expect("a newer schema must not be adopted");
    assert!(matches!(
        error,
        PlatformStoreError::UnsupportedSchemaVersion {
            found: 99,
            supported: PLATFORM_SCHEMA_VERSION,
        }
    ));
    database.drop_database().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_different_window_is_fatal_but_a_different_limit_is_not() {
    let Some(database) = TestDatabase::create().await else {
        return;
    };
    let config = database.config();
    let (_, first) = PlatformQuota::connect(&config, limit(10), window(60))
        .await
        .expect("first pod");
    assert_eq!(first, QuotaConfiguration::Initialized);

    let (_, same) = PlatformQuota::connect(&config, limit(10), window(60))
        .await
        .expect("second pod");
    assert_eq!(same, QuotaConfiguration::Unchanged);

    let (_, changed) = PlatformQuota::connect(&config, limit(25), window(60))
        .await
        .expect("a rolling limit change must not crash-loop the new pod");
    assert_eq!(changed, QuotaConfiguration::LimitChanged { previous: 10 });

    let error = PlatformQuota::connect(&config, limit(25), window(30))
        .await
        .err()
        .expect("a pod that disagrees about the window must refuse to start");
    assert!(matches!(
        error,
        PlatformStoreError::QuotaWindowMismatch {
            configured: 30,
            stored: 60
        }
    ));
    database.drop_database().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn an_unreachable_store_is_refused_at_startup() {
    if std::env::var(support::ADMIN_URL_VAR).is_err() {
        return;
    }
    let mut config = PlatformStoreConfig::new("postgresql://camelid@127.0.0.1:1/platform".into());
    config.acquire_timeout = Duration::from_millis(500);
    let error = PlatformStore::connect(&config)
        .await
        .err()
        .expect("a pod that cannot reach the store must not start");
    assert!(matches!(error, PlatformStoreError::Database(_)));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_store_that_stops_answering_refuses_rather_than_admitting() {
    let Some(database) = TestDatabase::create().await else {
        return;
    };
    let (quota, _) = PlatformQuota::connect(&database.config(), limit(10), window(60))
        .await
        .expect("store");
    assert!(quota.admit("org_gone").await.is_ok());

    database
        .client()
        .await
        .execute("DROP TABLE quota_windows", &[])
        .await
        .expect("break the store");

    match quota.admit("org_gone").await {
        Err(QuotaRefusal::Unavailable(_)) => {}
        Ok(()) => panic!("a store that cannot answer must never admit"),
        Err(QuotaRefusal::Exceeded { .. }) => {
            panic!("an outage must not be reported to the tenant as their own quota")
        }
    }
    database.drop_database().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn an_organization_id_shaped_like_sql_stays_data() {
    let Some(database) = TestDatabase::create().await else {
        return;
    };
    let (quota, _) = PlatformQuota::connect(&database.config(), limit(10), window(3600))
        .await
        .expect("store");
    let injection = "org_x'); DROP TABLE quota_windows; --";
    quota.admit(injection).await.expect("admit");

    let client = database.client().await;
    let survives: bool = client
        .query_one("SELECT to_regclass('quota_windows') IS NOT NULL", &[])
        .await
        .expect("table")
        .get(0);
    assert!(survives);
    let rows: i64 = client
        .query_one(
            "SELECT count(*) FROM quota_windows WHERE organization_id = $1",
            &[&injection],
        )
        .await
        .expect("rows")
        .get(0);
    assert_eq!(rows, 1, "the id was stored verbatim, as data");
    database.drop_database().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn the_sweeper_removes_only_windows_past_retention() {
    let Some(database) = TestDatabase::create().await else {
        return;
    };
    let (quota, _) = PlatformQuota::connect(&database.config(), limit(10), window(60))
        .await
        .expect("store");
    quota.admit("org_live").await.expect("admit");

    let client = database.client().await;
    client
        .execute(
            "INSERT INTO quota_windows (organization_id, window_start_epoch, request_count)
             VALUES ('org_stale', floor(extract(epoch FROM now()))::bigint - 6000, 5),
                    ('org_recent', floor(extract(epoch FROM now()))::bigint - 120, 5)",
            &[],
        )
        .await
        .expect("seed");

    assert_eq!(
        quota.sweep().await.expect("sweep"),
        SweepOutcome::Deleted(1)
    );
    let remaining: Vec<String> = client
        .query("SELECT organization_id FROM quota_windows ORDER BY 1", &[])
        .await
        .expect("remaining")
        .into_iter()
        .map(|row| row.get(0))
        .collect();
    assert_eq!(remaining, vec!["org_live", "org_recent"]);
    database.drop_database().await;
}

/// A session-scoped sweep lock survived a cancelled sweep on a recycled
/// connection, and because it shared a key with the schema lock, every later
/// pod blocked on it and died reporting a statement timeout.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_sweep_releases_its_lock_and_does_not_hold_the_schema_lock() {
    let Some(database) = TestDatabase::create().await else {
        return;
    };
    let (quota, _) = PlatformQuota::connect(&database.config(), limit(10), window(60))
        .await
        .expect("store");
    quota.admit("org_a").await.expect("admit");

    for _ in 0..3 {
        assert!(
            matches!(
                quota.sweep().await.expect("sweep"),
                SweepOutcome::Deleted(_)
            ),
            "a sweep that found the lock held would mean the previous one leaked it"
        );
    }

    let held: i64 = database
        .client()
        .await
        .query_one(
            "SELECT count(*) FROM pg_locks
             WHERE locktype = 'advisory'
               AND database = (SELECT oid FROM pg_database WHERE datname = current_database())",
            &[],
        )
        .await
        .expect("advisory locks")
        .get(0);
    assert_eq!(
        held, 0,
        "no advisory lock may outlive the sweep that took it"
    );

    // Startup takes the schema lock. A sweeper sharing that key would make this
    // block until it timed out.
    PlatformStore::connect(&database.config())
        .await
        .expect("a later pod must still be able to start");
    database.drop_database().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn the_sweeper_drains_more_than_one_batch() {
    let Some(database) = TestDatabase::create().await else {
        return;
    };
    let (quota, _) = PlatformQuota::connect(&database.config(), limit(10), window(60))
        .await
        .expect("store");
    let client = database.client().await;
    client
        .execute(
            "INSERT INTO quota_windows (organization_id, window_start_epoch, request_count)
             SELECT 'org_' || i, floor(extract(epoch FROM now()))::bigint - 6000 - i, 1
             FROM generate_series(1, 15000) AS i",
            &[],
        )
        .await
        .expect("seed a backlog larger than one batch");

    assert_eq!(
        quota.sweep().await.expect("sweep"),
        SweepOutcome::Deleted(15_000),
        "one bounded batch per sweep could never drain a backlog that grows faster"
    );
    database.drop_database().await;
}

/// The sweeper's predicate is on the second column of the primary key, so
/// without its own index it degrades to a sequential scan as the table grows.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn the_sweep_predicate_has_an_index() {
    let Some(database) = TestDatabase::create().await else {
        return;
    };
    PlatformStore::connect(&database.config())
        .await
        .expect("store");
    let indexed: bool = database
        .client()
        .await
        .query_one(
            "SELECT count(*) = 1 FROM pg_indexes
             WHERE tablename = 'quota_windows' AND indexdef LIKE '%(window_start_epoch)'",
            &[],
        )
        .await
        .expect("indexes")
        .get(0);
    assert!(indexed);
    database.drop_database().await;
}

/// The store holds the limit, so a pod cannot go on enforcing one the
/// deployment has moved off. Without this, lowering a limit over a rolling
/// update leaves every not-yet-restarted pod admitting against its old, higher
/// value for the rest of the window.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn the_limit_is_the_stores_and_not_the_pods() {
    let Some(database) = TestDatabase::create().await else {
        return;
    };
    let (quota, _) = PlatformQuota::connect(&database.config(), limit(100), window(3600))
        .await
        .expect("store");
    database
        .client()
        .await
        .execute("UPDATE quota_config SET request_limit = 2", &[])
        .await
        .expect("another pod records a lower limit");

    assert!(quota.admit("org_lowered").await.is_ok());
    assert!(quota.admit("org_lowered").await.is_ok());
    assert!(
        matches!(
            quota.admit("org_lowered").await,
            Err(QuotaRefusal::Exceeded { .. })
        ),
        "a pod started with limit 100 must enforce the store's 2"
    );
    database.drop_database().await;
}

/// A limit nobody can read is an outage, not a spent window. Guarding only the
/// conflict path left the insert branch — every organization's first request in
/// every window — admitting against no limit at all.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_store_with_no_quota_configuration_admits_nobody() {
    let Some(database) = TestDatabase::create().await else {
        return;
    };
    let (quota, _) = PlatformQuota::connect(&database.config(), limit(10), window(3600))
        .await
        .expect("store");
    quota.admit("org_seen").await.expect("admit");
    database
        .client()
        .await
        .execute("DELETE FROM quota_config", &[])
        .await
        .expect("lose the configuration");

    for organization in ["org_seen", "org_never_seen"] {
        match quota.admit(organization).await {
            Err(QuotaRefusal::Unavailable(_)) => {}
            Ok(()) => panic!("{organization} was admitted against no limit at all"),
            Err(QuotaRefusal::Exceeded { .. }) => {
                panic!("{organization} was told it had spent a quota nobody can read")
            }
        }
    }
    database.drop_database().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn the_window_can_be_reconfigured_but_only_on_purpose() {
    let Some(database) = TestDatabase::create().await else {
        return;
    };
    let config = database.config();
    let (quota, _) = PlatformQuota::connect(&config, limit(10), window(60))
        .await
        .expect("first pod");
    quota.admit("org_a").await.expect("admit");

    assert!(
        PlatformQuota::connect(&config, limit(10), window(30))
            .await
            .is_err(),
        "changing the window must not happen by restarting with a new value"
    );

    let (_, outcome) = PlatformQuota::connect_reconfiguring_window(&config, limit(10), window(30))
        .await
        .expect("an explicit reconfiguration must be possible");
    assert_eq!(outcome, QuotaConfiguration::WindowChanged { previous: 60 });

    let client = database.client().await;
    let stored: i64 = client
        .query_one("SELECT window_seconds FROM quota_config", &[])
        .await
        .expect("window")
        .get(0);
    assert_eq!(stored, 30);
    let stale: i64 = client
        .query_one("SELECT count(*) FROM quota_windows", &[])
        .await
        .expect("counters")
        .get(0);
    assert_eq!(
        stale, 0,
        "counters measured against window starts the new length cannot produce are unreachable"
    );

    // And the deployment is usable again on the new value.
    let (_, outcome) = PlatformQuota::connect(&config, limit(10), window(30))
        .await
        .expect("a pod on the new window must start normally");
    assert_eq!(outcome, QuotaConfiguration::Unchanged);
    database.drop_database().await;
}
