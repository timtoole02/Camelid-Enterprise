//! Shared harness for the tests that need a real PostgreSQL.
//!
//! Set `CAMELID_TEST_PLATFORM_DATABASE_URL` to run them; without it every test
//! that uses this skips, because the workspace suite also runs on hosts and CI
//! runners with no database. A skip that nobody notices is the same as no test
//! at all, so `postgres_tests_are_not_silently_skipped` fails when
//! `CAMELID_REQUIRE_PLATFORM_DATABASE_TESTS=1` — set only by the CI job that
//! provides the service container — and no URL is configured.

// Each integration test binary compiles this module separately, so whatever one
// of them does not use is dead code only from that binary's point of view.
#![allow(dead_code)]

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use camelid_enterprise_platform_store::PlatformStoreConfig;
use tokio_postgres::NoTls;

pub const ADMIN_URL_VAR: &str = "CAMELID_TEST_PLATFORM_DATABASE_URL";
pub const REQUIRE_VAR: &str = "CAMELID_REQUIRE_PLATFORM_DATABASE_TESTS";

/// A database of its own per test: quota configuration and the schema version
/// are deployment-wide singletons, so tests that share a database would be
/// testing each other.
pub struct TestDatabase {
    admin_url: String,
    name: String,
    url: String,
}

impl TestDatabase {
    pub async fn create() -> Option<Self> {
        let admin_url = std::env::var(ADMIN_URL_VAR).ok()?;
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let name = format!(
            "camelid_test_{}_{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos(),
            COUNTER.fetch_add(1, Ordering::Relaxed)
        );
        let admin = connect(&admin_url).await;
        admin
            .execute(&format!("CREATE DATABASE {name}"), &[])
            .await
            .expect("create test database");
        let url = replace_database(&admin_url, &name);
        Some(Self {
            admin_url,
            name,
            url,
        })
    }

    pub fn config(&self) -> PlatformStoreConfig {
        let mut config = PlatformStoreConfig::new(self.url.clone());
        config.acquire_timeout = Duration::from_secs(5);
        // Every test in a file runs concurrently against one server, each
        // holding its own pool. The production default of 8 would exhaust
        // `max_connections` and surface as unrelated failures.
        config.max_connections = 2;
        config
    }

    pub async fn client(&self) -> tokio_postgres::Client {
        connect(&self.url).await
    }

    pub async fn drop_database(self) {
        let admin = connect(&self.admin_url).await;
        let _ = admin
            .execute(
                &format!("DROP DATABASE IF EXISTS {} WITH (FORCE)", self.name),
                &[],
            )
            .await;
    }
}

pub async fn connect(url: &str) -> tokio_postgres::Client {
    let (client, connection) = tokio_postgres::connect(url, NoTls)
        .await
        .expect("connect to the test database");
    tokio::spawn(async move {
        let _ = connection.await;
    });
    client
}

fn replace_database(url: &str, database: &str) -> String {
    let (prefix, rest) = url.split_once("://").expect("a postgresql:// url");
    let (authority, tail) = match rest.split_once('/') {
        Some((authority, tail)) => (authority, tail),
        None => (rest, ""),
    };
    let query = tail.split_once('?').map(|(_, query)| query);
    match query {
        Some(query) => format!("{prefix}://{authority}/{database}?{query}"),
        None => format!("{prefix}://{authority}/{database}"),
    }
}

/// Guards against a whole test file quietly becoming a no-op.
pub fn assert_not_silently_skipped() {
    if std::env::var(REQUIRE_VAR).as_deref() != Ok("1") {
        return;
    }
    assert!(
        std::env::var(ADMIN_URL_VAR).is_ok(),
        "{REQUIRE_VAR}=1 but {ADMIN_URL_VAR} is unset: the PostgreSQL integration tests \
         would have skipped and CI would have stayed green"
    );
}
