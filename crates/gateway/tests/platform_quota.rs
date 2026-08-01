//! End-to-end quota behaviour against a real PostgreSQL.
//!
//! Skips unless `CAMELID_TEST_PLATFORM_DATABASE_URL` is set; the guard against
//! that skip going unnoticed lives in the platform-store crate's suite.
//!
//! `in_process_quotas_over_admit_across_two_gateways` is the control for the
//! test above it: it runs the identical load against the in-process quota and
//! asserts that it *does* over-admit. Without it, a passing shared-limit test
//! would only prove that the code ran.

use axum::body::{to_bytes, Body};
use axum::extract::Request;
use axum::http::StatusCode;
use axum::routing::any;
use axum::Router;
use camelid_enterprise_gateway::{
    router_with_options, GatewayAuth, GatewayLog, OrgQuota, UpstreamOrigin, DEFAULT_MAX_IN_FLIGHT,
};
use hyper_util::client::legacy::connect::HttpConnector;
use hyper_util::client::legacy::Client;
use hyper_util::rt::TokioExecutor;
use identity::{SqliteIdentityStore, TokenLifetime};
use platform_store::{PlatformQuota, PlatformStoreConfig};
use std::net::SocketAddr;
use std::num::{NonZeroU32, NonZeroU64};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tokio::task::JoinHandle;
use tokio_postgres::NoTls;

const ADMIN_URL_VAR: &str = "CAMELID_TEST_PLATFORM_DATABASE_URL";

struct TestServer {
    addr: SocketAddr,
    task: JoinHandle<()>,
}

impl Drop for TestServer {
    fn drop(&mut self) {
        self.task.abort();
    }
}

struct TestDatabase {
    admin_url: String,
    name: String,
    url: String,
}

impl TestDatabase {
    async fn create() -> Option<Self> {
        let admin_url = std::env::var(ADMIN_URL_VAR).ok()?;
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let name = format!(
            "camelid_gw_{}_{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos(),
            COUNTER.fetch_add(1, Ordering::Relaxed)
        );
        connect(&admin_url)
            .await
            .execute(&format!("CREATE DATABASE {name}"), &[])
            .await
            .expect("create test database");
        let (prefix, rest) = admin_url.split_once("://").expect("a postgresql:// url");
        let authority = rest.split('/').next().unwrap_or(rest);
        let url = format!("{prefix}://{authority}/{name}");
        Some(Self {
            admin_url,
            name,
            url,
        })
    }

    fn config(&self) -> PlatformStoreConfig {
        let mut config = PlatformStoreConfig::new(self.url.clone());
        config.acquire_timeout = Duration::from_secs(5);
        // These tests run concurrently against one server, each holding its own
        // pool; the production default of 8 would exhaust `max_connections`.
        config.max_connections = 2;
        config
    }

    async fn client(&self) -> tokio_postgres::Client {
        connect(&self.url).await
    }

    async fn drop_database(self) {
        let _ = connect(&self.admin_url)
            .await
            .execute(
                &format!("DROP DATABASE IF EXISTS {} WITH (FORCE)", self.name),
                &[],
            )
            .await;
    }
}

async fn connect(url: &str) -> tokio_postgres::Client {
    let (client, connection) = tokio_postgres::connect(url, NoTls)
        .await
        .expect("connect to the test database");
    tokio::spawn(async move {
        let _ = connection.await;
    });
    client
}

fn client() -> Client<HttpConnector, Body> {
    Client::builder(TokioExecutor::new()).build_http()
}

async fn spawn_counting_upstream(calls: Arc<AtomicUsize>) -> TestServer {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let app = Router::new().fallback(any(move || {
        let calls = Arc::clone(&calls);
        async move {
            calls.fetch_add(1, Ordering::SeqCst);
            StatusCode::NO_CONTENT
        }
    }));
    let task = tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    TestServer { addr, task }
}

async fn spawn_gateway(
    upstream: SocketAddr,
    store: Arc<SqliteIdentityStore>,
    quota: Arc<OrgQuota>,
    audit: Option<Arc<GatewayLog>>,
) -> TestServer {
    let upstream = UpstreamOrigin::parse(&format!("http://{upstream}")).unwrap();
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let router = router_with_options(
        upstream,
        DEFAULT_MAX_IN_FLIGHT,
        GatewayAuth::RequireToken {
            store,
            quota: Some(quota),
            usage: None,
        },
        audit,
    );
    let task = tokio::spawn(async move {
        camelid_enterprise_gateway::serve(
            listener,
            router,
            Duration::from_secs(30),
            std::future::pending(),
        )
        .await
        .unwrap();
    });
    TestServer { addr, task }
}

async fn get(addr: SocketAddr, token: &str) -> axum::http::Response<hyper::body::Incoming> {
    let request = Request::get(format!("http://{addr}/v1/models"))
        .header("authorization", format!("Bearer {token}"))
        .body(Body::empty())
        .unwrap();
    client().request(request).await.unwrap()
}

/// One identity store shared by both gateways, as two pods on one identity
/// database would have.
fn identity() -> (Arc<SqliteIdentityStore>, String) {
    let store = SqliteIdentityStore::open_in_memory().unwrap();
    let principal = store.create_user("ada").unwrap();
    let token = store.issue_token(&principal, TokenLifetime::Never).unwrap();
    (Arc::new(store), token)
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn two_gateways_admit_exactly_the_shared_limit_for_one_organization() {
    let Some(database) = TestDatabase::create().await else {
        return;
    };
    let (store, token) = identity();
    let calls = Arc::new(AtomicUsize::new(0));
    let upstream = spawn_counting_upstream(Arc::clone(&calls)).await;

    let config = database.config();
    let limit = NonZeroU32::new(10).unwrap();
    let window = NonZeroU64::new(3600).unwrap();
    let (left_quota, _) = PlatformQuota::connect(&config, limit, window)
        .await
        .unwrap();
    let (right_quota, _) = PlatformQuota::connect(&config, limit, window)
        .await
        .unwrap();

    let left = spawn_gateway(
        upstream.addr,
        Arc::clone(&store),
        Arc::new(OrgQuota::platform(Arc::new(left_quota))),
        None,
    )
    .await;
    let right = spawn_gateway(
        upstream.addr,
        Arc::clone(&store),
        Arc::new(OrgQuota::platform(Arc::new(right_quota))),
        None,
    )
    .await;

    let mut admitted = 0;
    let mut refused = 0;
    for attempt in 0..40 {
        let addr = if attempt % 2 == 0 {
            left.addr
        } else {
            right.addr
        };
        match get(addr, &token).await.status() {
            StatusCode::NO_CONTENT => admitted += 1,
            StatusCode::TOO_MANY_REQUESTS => refused += 1,
            other => panic!("unexpected status {other}"),
        }
    }

    assert_eq!(admitted, 10, "the limit belongs to the deployment");
    assert_eq!(refused, 30);
    assert_eq!(
        calls.load(Ordering::SeqCst),
        10,
        "a refused request must never reach a replica"
    );
    database.drop_database().await;
}

/// The control for the test above.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn in_process_quotas_over_admit_across_two_gateways() {
    let (store, token) = identity();
    let calls = Arc::new(AtomicUsize::new(0));
    let upstream = spawn_counting_upstream(Arc::clone(&calls)).await;
    let limit = NonZeroU32::new(10).unwrap();
    let window = NonZeroU64::new(3600).unwrap();

    let left = spawn_gateway(
        upstream.addr,
        Arc::clone(&store),
        Arc::new(OrgQuota::new(limit, window)),
        None,
    )
    .await;
    let right = spawn_gateway(
        upstream.addr,
        Arc::clone(&store),
        Arc::new(OrgQuota::new(limit, window)),
        None,
    )
    .await;

    let mut admitted = 0;
    for attempt in 0..40 {
        let addr = if attempt % 2 == 0 {
            left.addr
        } else {
            right.addr
        };
        if get(addr, &token).await.status() == StatusCode::NO_CONTENT {
            admitted += 1;
        }
    }

    assert_eq!(
        admitted, 20,
        "two processes counting separately admit twice the configured limit; this is the \
         defect the platform-backed quota removes, and the assertion above is only \
         meaningful because this one holds"
    );
    assert_eq!(calls.load(Ordering::SeqCst), 20);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn an_unavailable_platform_store_refuses_before_forwarding() {
    let Some(database) = TestDatabase::create().await else {
        return;
    };
    let (store, token) = identity();
    let calls = Arc::new(AtomicUsize::new(0));
    let upstream = spawn_counting_upstream(Arc::clone(&calls)).await;
    let audit_file = tempfile::NamedTempFile::new().unwrap();
    let audit = GatewayLog::open(audit_file.path()).unwrap();

    let (quota, _) = PlatformQuota::connect(
        &database.config(),
        NonZeroU32::new(10).unwrap(),
        NonZeroU64::new(3600).unwrap(),
    )
    .await
    .unwrap();
    let gateway = spawn_gateway(
        upstream.addr,
        store,
        Arc::new(OrgQuota::platform(Arc::new(quota))),
        Some(Arc::clone(&audit)),
    )
    .await;

    assert_eq!(
        get(gateway.addr, &token).await.status(),
        StatusCode::NO_CONTENT
    );
    database
        .client()
        .await
        .execute("DROP TABLE quota_windows", &[])
        .await
        .unwrap();

    let refused = get(gateway.addr, &token).await;
    assert_eq!(refused.status(), StatusCode::SERVICE_UNAVAILABLE);
    assert!(
        refused.headers().contains_key("retry-after"),
        "a shed request carries the gateway's backpressure contract"
    );
    let body = to_bytes(Body::new(refused.into_body()), 1024)
        .await
        .unwrap();
    assert_eq!(
        body,
        r#"{"error":{"message":"gateway quota store unavailable","type":"gateway_error"}}"#
    );
    assert_eq!(
        calls.load(Ordering::SeqCst),
        1,
        "an outage must fail closed, not fall back to this pod's own counter"
    );

    audit.flush_and_stop(Duration::from_secs(5));
    let recorded = std::fs::read_to_string(audit_file.path()).unwrap();
    assert!(
        recorded.contains(r#""reason":"quota_store_unavailable""#),
        "an outage and a spent quota must not be the same audit line: {recorded}"
    );
    database.drop_database().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn an_invalid_token_is_never_charged_to_a_quota() {
    let Some(database) = TestDatabase::create().await else {
        return;
    };
    let (store, _) = identity();
    let calls = Arc::new(AtomicUsize::new(0));
    let upstream = spawn_counting_upstream(Arc::clone(&calls)).await;
    let (quota, _) = PlatformQuota::connect(
        &database.config(),
        NonZeroU32::new(10).unwrap(),
        NonZeroU64::new(3600).unwrap(),
    )
    .await
    .unwrap();
    let gateway = spawn_gateway(
        upstream.addr,
        store,
        Arc::new(OrgQuota::platform(Arc::new(quota))),
        None,
    )
    .await;

    for _ in 0..5 {
        assert_eq!(
            get(gateway.addr, "not-a-real-token").await.status(),
            StatusCode::UNAUTHORIZED
        );
    }

    let charged: i64 = database
        .client()
        .await
        .query_one("SELECT count(*) FROM quota_windows", &[])
        .await
        .unwrap()
        .get(0);
    assert_eq!(
        charged, 0,
        "a request that never resolved an organization cannot have spent one's budget"
    );
    assert_eq!(calls.load(Ordering::SeqCst), 0);
    database.drop_database().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn both_gateways_answer_the_same_retry_after() {
    let Some(database) = TestDatabase::create().await else {
        return;
    };
    let (store, token) = identity();
    let calls = Arc::new(AtomicUsize::new(0));
    let upstream = spawn_counting_upstream(calls).await;
    let config = database.config();
    let limit = NonZeroU32::new(1).unwrap();
    let window = NonZeroU64::new(600).unwrap();
    let (left_quota, _) = PlatformQuota::connect(&config, limit, window)
        .await
        .unwrap();
    let (right_quota, _) = PlatformQuota::connect(&config, limit, window)
        .await
        .unwrap();

    let left = spawn_gateway(
        upstream.addr,
        Arc::clone(&store),
        Arc::new(OrgQuota::platform(Arc::new(left_quota))),
        None,
    )
    .await;
    let right = spawn_gateway(
        upstream.addr,
        Arc::clone(&store),
        Arc::new(OrgQuota::platform(Arc::new(right_quota))),
        None,
    )
    .await;

    assert_eq!(
        get(left.addr, &token).await.status(),
        StatusCode::NO_CONTENT
    );
    let from_left = retry_after(get(left.addr, &token).await);
    let from_right = retry_after(get(right.addr, &token).await);
    assert!(
        from_left.abs_diff(from_right) <= 1,
        "pods disagreeing about the window would hand clients different answers: \
         {from_left} vs {from_right}"
    );
    assert!(from_left <= 600);
    database.drop_database().await;
}

fn retry_after(response: axum::http::Response<hyper::body::Incoming>) -> u64 {
    assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);
    response.headers()["retry-after"]
        .to_str()
        .unwrap()
        .parse()
        .unwrap()
}
