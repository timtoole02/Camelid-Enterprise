use axum::body::{to_bytes, Body};
use axum::extract::{Request, State};
use axum::http::{Method, StatusCode};
use axum::response::Response;
use axum::routing::{any, post};
use axum::Router;
use bytes::Bytes;
use camelid_enterprise_gateway::{
    router as gateway_router, router_with_max_in_flight, router_with_model_catalog,
    router_with_options, GatewayAuth, GatewayLog, LogFlush, ModelCatalog, ModelSelectionLimits,
    OrgQuota, UpstreamOrigin, DEFAULT_MAX_IN_FLIGHT, DEFAULT_MAX_MODEL_SELECTION_BODY_BYTES,
    DEFAULT_MODEL_SELECTION_MEMORY_BUDGET_BYTES,
};
use futures_util::{stream, StreamExt};
use http_body_util::BodyExt;
use hyper_util::client::legacy::connect::HttpConnector;
use hyper_util::client::legacy::Client;
use hyper_util::rt::TokioExecutor;
use identity::{SqliteIdentityStore, TokenLifetime};
use replica_contract::{HttpMethod, PUBLIC_ROUTES};
use std::convert::Infallible;
use std::net::SocketAddr;
use std::num::{NonZeroU32, NonZeroU64, NonZeroUsize};
use std::path::Path;
use std::pin::Pin;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll};
use std::time::Duration;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::sync::oneshot;
use tokio::sync::Notify;
use tokio::task::JoinHandle;
use tower::ServiceExt;

struct TestServer {
    addr: SocketAddr,
    task: JoinHandle<()>,
}

impl Drop for TestServer {
    fn drop(&mut self) {
        self.task.abort();
    }
}

async fn spawn_server(app: Router) -> TestServer {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let task = tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    TestServer { addr, task }
}

async fn spawn_catalog_connection_dropper(calls: Arc<AtomicUsize>) -> TestServer {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let task = tokio::spawn(async move {
        loop {
            let (mut stream, _) = listener.accept().await.unwrap();
            let calls = Arc::clone(&calls);
            tokio::spawn(async move {
                let mut request = [0_u8; 1024];
                let read = stream.read(&mut request).await.unwrap_or(0);
                if request[..read].starts_with(b"GET /v1/models ") {
                    let body = r#"{"object":"list","data":[{"id":"alpha"}]}"#;
                    let response = format!(
                        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                        body.len()
                    );
                    stream.write_all(response.as_bytes()).await.unwrap();
                    return;
                }
                calls.fetch_add(1, Ordering::SeqCst);
                drop(stream);
            });
        }
    });
    TestServer { addr, task }
}

async fn spawn_gateway(upstream: SocketAddr) -> TestServer {
    spawn_gateway_with_limits(upstream, DEFAULT_MAX_IN_FLIGHT, Duration::from_secs(30)).await
}

async fn spawn_gateway_with_limits(
    upstream: SocketAddr,
    max_in_flight: NonZeroUsize,
    max_connection_duration: Duration,
) -> TestServer {
    spawn_gateway_with_options(
        upstream,
        max_in_flight,
        max_connection_duration,
        GatewayAuth::Disabled,
        None,
    )
    .await
}

async fn spawn_gateway_with_auth(upstream: SocketAddr, auth: GatewayAuth) -> TestServer {
    spawn_gateway_with_options(
        upstream,
        DEFAULT_MAX_IN_FLIGHT,
        Duration::from_secs(30),
        auth,
        None,
    )
    .await
}

/// Spawns a gateway that writes its request audit log to `audit`, with the
/// given authentication mode.
async fn spawn_gateway_with_audit(
    upstream: SocketAddr,
    auth: GatewayAuth,
    audit: Arc<GatewayLog>,
) -> TestServer {
    spawn_gateway_with_options(
        upstream,
        DEFAULT_MAX_IN_FLIGHT,
        Duration::from_secs(30),
        auth,
        Some(audit),
    )
    .await
}

async fn spawn_gateway_with_options(
    upstream: SocketAddr,
    max_in_flight: NonZeroUsize,
    max_connection_duration: Duration,
    auth: GatewayAuth,
    audit: Option<Arc<GatewayLog>>,
) -> TestServer {
    let upstream = UpstreamOrigin::parse(&format!("http://{upstream}")).unwrap();
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let router = router_with_options(upstream, max_in_flight, auth, audit);
    let task = tokio::spawn(async move {
        camelid_enterprise_gateway::serve(
            listener,
            router,
            max_connection_duration,
            std::future::pending(),
        )
        .await
        .unwrap();
    });
    TestServer { addr, task }
}

async fn spawn_catalog_gateway(routes: &[(&str, SocketAddr)]) -> TestServer {
    spawn_catalog_gateway_with_options(
        routes,
        DEFAULT_MAX_IN_FLIGHT,
        default_selection_limits(),
        GatewayAuth::Disabled,
    )
    .await
}

async fn spawn_catalog_gateway_with_options(
    routes: &[(&str, SocketAddr)],
    max_in_flight: NonZeroUsize,
    selection_limits: ModelSelectionLimits,
    auth: GatewayAuth,
) -> TestServer {
    spawn_catalog_gateway_with_audit(routes, max_in_flight, selection_limits, auth, None).await
}

async fn spawn_catalog_gateway_with_audit(
    routes: &[(&str, SocketAddr)],
    max_in_flight: NonZeroUsize,
    selection_limits: ModelSelectionLimits,
    auth: GatewayAuth,
    audit: Option<Arc<GatewayLog>>,
) -> TestServer {
    let catalog = ModelCatalog::new(routes.iter().map(|(model_id, upstream)| {
        (
            (*model_id).to_string(),
            UpstreamOrigin::parse(&format!("http://{upstream}")).unwrap(),
        )
    }))
    .unwrap();
    let catalog = catalog.verify_backend_model_ids().await.unwrap();
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let router =
        router_with_model_catalog(catalog, max_in_flight, selection_limits, None, auth, audit);
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

fn default_selection_limits() -> ModelSelectionLimits {
    ModelSelectionLimits::new(
        DEFAULT_MAX_MODEL_SELECTION_BODY_BYTES,
        DEFAULT_MODEL_SELECTION_MEMORY_BUDGET_BYTES,
    )
    .unwrap()
}

fn gateway_log(path: &Path) -> Arc<GatewayLog> {
    GatewayLog::open(path).unwrap()
}

#[tokio::test]
async fn audit_records_survive_a_graceful_shutdown() {
    // Covers the wiring and the ordering: real requests over real HTTP, a real
    // graceful shutdown, and the drain running after `serve` returns so that
    // records produced while connections were finishing are included.
    //
    // It does not prove the loss it guards against. In-process the writer
    // thread outlives `serve`, so without the drain it would catch up on its
    // own and this would still pass -- measured, not assumed. The loss only
    // bites when the *process* exits, and
    // `flush_and_stop_persists_every_accepted_record` carries that proof
    // deterministically against a stubbed-out drain.
    const REQUESTS: usize = 200;
    let upstream =
        spawn_server(Router::new().fallback(any(|| async { StatusCode::NO_CONTENT }))).await;
    let dir = tempfile::tempdir().unwrap();
    let audit_path = dir.path().join("audit.jsonl");
    let audit = gateway_log(&audit_path);

    let upstream_origin = UpstreamOrigin::parse(&format!("http://{}", upstream.addr)).unwrap();
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let router = router_with_options(
        upstream_origin,
        DEFAULT_MAX_IN_FLIGHT,
        GatewayAuth::Disabled,
        Some(Arc::clone(&audit)),
    );
    let (stop, stopped) = oneshot::channel::<()>();
    let served = tokio::spawn(async move {
        camelid_enterprise_gateway::serve(listener, router, Duration::from_secs(30), async {
            let _ = stopped.await;
        })
        .await
        .unwrap();
    });

    // Concurrent rather than sequential, so records genuinely pile up behind
    // the writer instead of trickling in at a pace it can always match.
    let mut requests = tokio::task::JoinSet::new();
    for _ in 0..REQUESTS {
        requests.spawn(async move {
            client()
                .request(
                    Request::get(format!("http://{addr}/v1/models"))
                        .body(Body::empty())
                        .unwrap(),
                )
                .await
                .unwrap()
                .status()
        });
    }
    while let Some(status) = requests.join_next().await {
        assert_eq!(status.unwrap(), StatusCode::NO_CONTENT);
    }

    // Shut down exactly as the binary does: stop accepting, drain connections,
    // and only then drain the log.
    stop.send(()).unwrap();
    served.await.unwrap();
    assert_eq!(
        audit.flush_and_stop(Duration::from_secs(30)),
        LogFlush::Drained
    );

    // Read once, with no retry loop.
    let contents = std::fs::read_to_string(&audit_path).unwrap();
    let lines: Vec<&str> = contents
        .lines()
        .filter(|line| !line.trim().is_empty())
        .collect();
    assert_eq!(
        lines.len(),
        REQUESTS,
        "every audited request must be on disk after shutdown drains the log"
    );
    let ids: std::collections::HashSet<String> = lines
        .iter()
        .map(|line| {
            let record: serde_json::Value = serde_json::from_str(line).unwrap();
            record["request_id"].as_str().unwrap().to_string()
        })
        .collect();
    assert_eq!(
        ids.len(),
        REQUESTS,
        "each record must be a distinct request"
    );
}

fn client() -> Client<HttpConnector, Body> {
    Client::builder(TokioExecutor::new()).build_http()
}

#[derive(Debug)]
struct CapturedRequest {
    method: Method,
    path_and_query: String,
    host: String,
    client_header: String,
    forwarded_header_present: bool,
    body: Bytes,
}

async fn capture_request(
    State(captured): State<Arc<Mutex<Option<CapturedRequest>>>>,
    request: Request,
) -> Response {
    let method = request.method().clone();
    let path_and_query = request.uri().path_and_query().unwrap().as_str().to_string();
    let host = request.headers()["host"].to_str().unwrap().to_string();
    let client_header = request.headers()["x-client-test"]
        .to_str()
        .unwrap()
        .to_string();
    let forwarded_header_present = [
        "forwarded",
        "x-forwarded-for",
        "x-forwarded-host",
        "x-forwarded-port",
        "x-forwarded-proto",
        "x-real-ip",
    ]
    .iter()
    .any(|name| request.headers().contains_key(*name));
    let body = to_bytes(request.into_body(), 1024).await.unwrap();
    *captured.lock().unwrap() = Some(CapturedRequest {
        method,
        path_and_query,
        host,
        client_header,
        forwarded_header_present,
        body,
    });

    let mut response = Response::builder()
        .status(StatusCode::TOO_MANY_REQUESTS)
        .header("content-type", "application/json")
        .header("retry-after", "7")
        .header("x-camelid-lane", "deterministic")
        .header("x-camelid-config-sha256", "30d77c260803")
        .header("x-camelid-host", "linux/x86_64 cores=8 simd=avx2+fma")
        .body(Body::from(r#"{"error":"replica queue full"}"#))
        .unwrap();
    response
        .headers_mut()
        .append("set-cookie", "session=one".parse().unwrap());
    response
        .headers_mut()
        .append("set-cookie", "csrf=two".parse().unwrap());
    response
}

async fn usage_response(request: Request) -> Response {
    let _ = to_bytes(request.into_body(), 1024).await.unwrap();
    tokio::time::sleep(Duration::from_millis(75)).await;
    Response::new(Body::from("generated"))
}

async fn delayed_failing_usage_response(State(release): State<Arc<Notify>>) -> Response {
    let chunks = stream::unfold((0, release), |(step, release)| async move {
        match step {
            0 => Some((
                Ok::<_, std::io::Error>(Bytes::from_static(b"partial")),
                (1, release),
            )),
            1 => {
                release.notified().await;
                Some((
                    Err(std::io::Error::other("replica stream failed")),
                    (2, release),
                ))
            }
            _ => None,
        }
    });
    Response::builder()
        .status(StatusCode::OK)
        .body(Body::from_stream(chunks))
        .unwrap()
}

#[tokio::test]
async fn preserves_request_and_replica_response_contract() {
    let captured = Arc::new(Mutex::new(None));
    let upstream = spawn_server(
        Router::new()
            .fallback(any(capture_request))
            .with_state(Arc::clone(&captured)),
    )
    .await;
    let gateway = spawn_gateway(upstream.addr).await;
    let request_body = r#"{"messages":[{"role":"user","content":"hello"}]}"#;
    let request = Request::builder()
        .method(Method::POST)
        .uri(format!(
            "http://{}/v1/chat/completions?model=camelid%2Ftest&stream=false",
            gateway.addr
        ))
        .header("host", "public-gateway.example")
        .header("content-type", "application/json")
        .header("x-client-test", "preserve-me")
        .header("forwarded", "for=203.0.113.10;proto=https")
        .header("x-forwarded-for", "203.0.113.10")
        .header("x-forwarded-host", "spoofed.example")
        .header("x-forwarded-port", "443")
        .header("x-forwarded-proto", "https")
        .header("x-real-ip", "203.0.113.10")
        .body(Body::from(request_body))
        .unwrap();

    let response = client().request(request).await.unwrap();

    assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);
    assert_eq!(response.headers()["retry-after"], "7");
    assert_eq!(response.headers()["x-camelid-lane"], "deterministic");
    assert_eq!(
        response.headers()["x-camelid-config-sha256"],
        "30d77c260803"
    );
    assert_eq!(
        response.headers()["x-camelid-host"],
        "linux/x86_64 cores=8 simd=avx2+fma"
    );
    assert_eq!(
        response
            .headers()
            .get_all("set-cookie")
            .iter()
            .map(|value| value.to_str().unwrap())
            .collect::<Vec<_>>(),
        ["session=one", "csrf=two"]
    );
    assert_eq!(
        to_bytes(Body::new(response.into_body()), 1024)
            .await
            .unwrap(),
        r#"{"error":"replica queue full"}"#
    );

    let captured = captured.lock().unwrap().take().unwrap();
    assert_eq!(captured.method, Method::POST);
    assert_eq!(
        captured.path_and_query,
        "/v1/chat/completions?model=camelid%2Ftest&stream=false"
    );
    assert_eq!(captured.host, upstream.addr.to_string());
    assert_eq!(captured.client_header, "preserve-me");
    assert!(!captured.forwarded_header_present);
    assert_eq!(captured.body, request_body);
}

async fn delayed_sse(State(release_second): State<Arc<Notify>>) -> Response {
    let events = stream::unfold((0, release_second), |(step, release)| async move {
        match step {
            0 => Some((
                Ok::<_, Infallible>(Bytes::from_static(b"data: first\n\n")),
                (1, release),
            )),
            1 => {
                release.notified().await;
                Some((
                    Ok::<_, Infallible>(Bytes::from_static(b"data: second\n\n")),
                    (2, release),
                ))
            }
            _ => None,
        }
    });
    Response::builder()
        .header("content-type", "text/event-stream")
        .header("x-camelid-lane", "deterministic")
        .body(Body::from_stream(events))
        .unwrap()
}

#[tokio::test]
async fn streams_replica_response_before_it_finishes() {
    let release_second = Arc::new(Notify::new());
    let upstream = spawn_server(
        Router::new()
            .route("/v1/chat/completions", any(delayed_sse))
            .with_state(Arc::clone(&release_second)),
    )
    .await;
    let gateway = spawn_gateway(upstream.addr).await;
    let request = Request::post(format!("http://{}/v1/chat/completions", gateway.addr))
        .body(Body::empty())
        .unwrap();

    let response = client().request(request).await.unwrap();
    assert_eq!(response.headers()["content-type"], "text/event-stream");
    assert_eq!(response.headers()["x-camelid-lane"], "deterministic");
    let mut body = response.into_body();

    let first = tokio::time::timeout(Duration::from_secs(1), body.frame())
        .await
        .expect("the first event must arrive before the upstream finishes")
        .unwrap()
        .unwrap()
        .into_data()
        .unwrap();
    assert_eq!(first, "data: first\n\n");
    assert!(
        tokio::time::timeout(Duration::from_millis(100), body.frame())
            .await
            .is_err(),
        "the second event must remain pending until the upstream sends it"
    );

    release_second.notify_one();
    let second = tokio::time::timeout(Duration::from_secs(1), body.frame())
        .await
        .expect("the released event must arrive")
        .unwrap()
        .unwrap()
        .into_data()
        .unwrap();
    assert_eq!(second, "data: second\n\n");
}

struct DisconnectAwareStream {
    first_sent: bool,
    dropped: Arc<Notify>,
}

impl futures_util::Stream for DisconnectAwareStream {
    type Item = Result<Bytes, Infallible>;

    fn poll_next(mut self: Pin<&mut Self>, _: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        if self.first_sent {
            Poll::Pending
        } else {
            self.first_sent = true;
            Poll::Ready(Some(Ok(Bytes::from_static(b"data: first\n\n"))))
        }
    }
}

impl Drop for DisconnectAwareStream {
    fn drop(&mut self) {
        self.dropped.notify_one();
    }
}

async fn disconnect_aware_sse(State(dropped): State<Arc<Notify>>) -> Response {
    Response::builder()
        .header("content-type", "text/event-stream")
        .body(Body::from_stream(DisconnectAwareStream {
            first_sent: false,
            dropped,
        }))
        .unwrap()
}

#[tokio::test]
async fn client_disconnect_cancels_the_upstream_response_stream() {
    let upstream_dropped = Arc::new(Notify::new());
    let upstream = spawn_server(
        Router::new()
            .route("/v1/chat/completions", any(disconnect_aware_sse))
            .with_state(Arc::clone(&upstream_dropped)),
    )
    .await;
    let gateway = spawn_gateway(upstream.addr).await;
    let request = Request::post(format!("http://{}/v1/chat/completions", gateway.addr))
        .body(Body::empty())
        .unwrap();

    let response = client().request(request).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let mut body = response.into_body();
    let first = body.frame().await.unwrap().unwrap().into_data().unwrap();
    assert_eq!(first, "data: first\n\n");
    drop(body);

    tokio::time::timeout(Duration::from_secs(1), upstream_dropped.notified())
        .await
        .expect("dropping the client response must cancel the upstream stream");
}

#[derive(Clone)]
struct RequestStreamState {
    first_chunk_seen: Arc<Notify>,
}

async fn consume_streamed_request(
    State(state): State<RequestStreamState>,
    request: Request,
) -> Response {
    let mut body = request.into_body();
    let first = body.frame().await.unwrap().unwrap().into_data().unwrap();
    state.first_chunk_seen.notify_one();
    let second = body.frame().await.unwrap().unwrap().into_data().unwrap();
    assert!(body.frame().await.is_none());
    Response::new(Body::from([first.as_ref(), second.as_ref()].concat()))
}

#[tokio::test]
async fn streams_client_request_before_it_finishes() {
    let first_chunk_seen = Arc::new(Notify::new());
    let release_second = Arc::new(Notify::new());
    let upstream = spawn_server(
        Router::new()
            .route("/v1/chat/completions", any(consume_streamed_request))
            .with_state(RequestStreamState {
                first_chunk_seen: Arc::clone(&first_chunk_seen),
            }),
    )
    .await;
    let gateway = spawn_gateway(upstream.addr).await;
    let chunks = stream::unfold(
        (0, Arc::clone(&release_second)),
        |(step, release)| async move {
            match step {
                0 => Some((
                    Ok::<_, Infallible>(Bytes::from_static(b"first-")),
                    (1, release),
                )),
                1 => {
                    release.notified().await;
                    Some((
                        Ok::<_, Infallible>(Bytes::from_static(b"second")),
                        (2, release),
                    ))
                }
                _ => None,
            }
        },
    );
    let request = Request::post(format!("http://{}/v1/chat/completions", gateway.addr))
        .body(Body::from_stream(chunks))
        .unwrap();
    let response_task = tokio::spawn(client().request(request));

    tokio::time::timeout(Duration::from_secs(1), first_chunk_seen.notified())
        .await
        .expect("the upstream must see the first chunk before the client finishes");
    release_second.notify_one();

    let response = response_task.await.unwrap().unwrap();
    assert_eq!(
        to_bytes(Body::new(response.into_body()), 1024)
            .await
            .unwrap(),
        "first-second"
    );
}

#[tokio::test]
async fn returns_typed_bad_gateway_only_for_gateway_failure() {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let unavailable = listener.local_addr().unwrap();
    drop(listener);
    let app = gateway_router(UpstreamOrigin::parse(&format!("http://{unavailable}")).unwrap());
    let request = Request::get("/v1/models").body(Body::empty()).unwrap();

    let response = app.oneshot(request).await.unwrap();

    assert_eq!(response.status(), StatusCode::BAD_GATEWAY);
    assert_eq!(response.headers()["content-type"], "application/json");
    assert_eq!(response.headers()["access-control-allow-origin"], "*");
    assert_eq!(response.headers()["access-control-expose-headers"], "*");
    assert!(!response.headers().contains_key("x-camelid-lane"));
    assert_eq!(
        to_bytes(response.into_body(), 1024).await.unwrap(),
        r#"{"error":{"message":"upstream replica is unavailable","type":"gateway_error"}}"#
    );
}

#[tokio::test]
async fn rejects_requests_with_no_bearer_token_when_auth_is_required() {
    let store = Arc::new(SqliteIdentityStore::open_in_memory().unwrap());
    let unreachable_upstream = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .unwrap()
        .local_addr()
        .unwrap();

    let gateway = spawn_gateway_with_auth(
        unreachable_upstream,
        GatewayAuth::RequireToken {
            store,
            quota: None,
            usage: None,
        },
    )
    .await;
    let request = Request::get(format!("http://{}/v1/models", gateway.addr))
        .body(Body::empty())
        .unwrap();

    let response = client().request(request).await.unwrap();

    assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    assert_eq!(response.headers()["www-authenticate"], "Bearer");
    assert_eq!(
        to_bytes(Body::new(response.into_body()), 1024)
            .await
            .unwrap(),
        r#"{"error":{"message":"missing bearer token","type":"unauthorized"}}"#
    );
}

#[tokio::test]
async fn rejects_requests_with_an_invalid_bearer_token() {
    let store = Arc::new(SqliteIdentityStore::open_in_memory().unwrap());
    let unreachable_upstream = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .unwrap()
        .local_addr()
        .unwrap();

    let gateway = spawn_gateway_with_auth(
        unreachable_upstream,
        GatewayAuth::RequireToken {
            store,
            quota: None,
            usage: None,
        },
    )
    .await;
    let request = Request::get(format!("http://{}/v1/models", gateway.addr))
        .header("authorization", "Bearer not-a-real-token")
        .body(Body::empty())
        .unwrap();

    let response = client().request(request).await.unwrap();

    assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
}

#[tokio::test]
async fn rejects_requests_with_an_expired_bearer_token() {
    let store = SqliteIdentityStore::open_in_memory().unwrap();
    let principal = store.create_user("ada").unwrap();
    let expired = store
        .issue_token(&principal, TokenLifetime::Until(identity::unix_now() - 1))
        .unwrap();
    // Deliberately unreachable: were the expired token forwarded, this would
    // answer 502 rather than 401, so the assertion cannot pass by accident.
    let unreachable_upstream = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .unwrap()
        .local_addr()
        .unwrap();

    let gateway = spawn_gateway_with_auth(
        unreachable_upstream,
        GatewayAuth::RequireToken {
            store: Arc::new(store),
            quota: None,
            usage: None,
        },
    )
    .await;
    let request = Request::get(format!("http://{}/v1/models", gateway.addr))
        .header("authorization", format!("Bearer {expired}"))
        .body(Body::empty())
        .unwrap();

    let response = client().request(request).await.unwrap();

    assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    // The distinction has to be machine-readable to be worth making. `type` is
    // the discriminator this API uses everywhere else, and the challenge is
    // RFC 6750's: `invalid_token` is the registered code covering expiry, so
    // the specificity rides in the description and in `type`.
    assert_eq!(
        response.headers()["www-authenticate"],
        r#"Bearer error="invalid_token", error_description="expired bearer token""#
    );
    assert_eq!(
        to_bytes(Body::new(response.into_body()), 1024)
            .await
            .unwrap(),
        r#"{"error":{"message":"expired bearer token","type":"token_expired"}}"#
    );
}

#[tokio::test]
async fn an_invalid_bearer_token_is_not_reported_as_an_expired_one() {
    let store = SqliteIdentityStore::open_in_memory().unwrap();
    let unreachable_upstream = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .unwrap()
        .local_addr()
        .unwrap();

    let gateway = spawn_gateway_with_auth(
        unreachable_upstream,
        GatewayAuth::RequireToken {
            store: Arc::new(store),
            quota: None,
            usage: None,
        },
    )
    .await;
    let request = Request::get(format!("http://{}/v1/models", gateway.addr))
        .header("authorization", "Bearer cme_never_issued")
        .body(Body::empty())
        .unwrap();

    let response = client().request(request).await.unwrap();

    // The other half of the contract above: an unknown credential keeps the
    // pre-existing `unauthorized` type, so a client branching on
    // `token_expired` cannot be told to go refresh a token that never existed.
    assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    assert_eq!(
        response.headers()["www-authenticate"],
        r#"Bearer error="invalid_token", error_description="invalid bearer token""#
    );
    assert_eq!(
        to_bytes(Body::new(response.into_body()), 1024)
            .await
            .unwrap(),
        r#"{"error":{"message":"invalid bearer token","type":"unauthorized"}}"#
    );
}

#[tokio::test]
async fn forwards_requests_carrying_a_token_that_has_not_expired_yet() {
    let store = SqliteIdentityStore::open_in_memory().unwrap();
    let principal = store.create_user("ada").unwrap();
    let token = store
        .issue_token(
            &principal,
            TokenLifetime::expires_in(NonZeroU64::new(3600).unwrap()),
        )
        .unwrap();

    let captured = Arc::new(Mutex::new(None));
    let upstream = spawn_server(
        Router::new()
            .fallback(any(capture_request))
            .with_state(Arc::clone(&captured)),
    )
    .await;
    let gateway = spawn_gateway_with_auth(
        upstream.addr,
        GatewayAuth::RequireToken {
            store: Arc::new(store),
            quota: None,
            usage: None,
        },
    )
    .await;
    let request = Request::get(format!("http://{}/v1/models", gateway.addr))
        .header("host", "public-gateway.example")
        .header("authorization", format!("Bearer {token}"))
        .header("x-client-test", "unexpired")
        .body(Body::empty())
        .unwrap();

    let response = client().request(request).await.unwrap();

    // 429 is what `capture_request` answers with, not a quota verdict: reaching
    // the upstream at all is the assertion here, and the status it returns is
    // arbitrary. `captured` being populated is what proves the request was
    // forwarded rather than refused at the gateway.
    assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);
    assert!(captured.lock().unwrap().is_some());
}

#[tokio::test]
async fn forwards_requests_carrying_a_valid_bearer_token() {
    let store = SqliteIdentityStore::open_in_memory().unwrap();
    let principal = store.create_user("ada").unwrap();
    let token = store.issue_token(&principal, TokenLifetime::Never).unwrap();

    let captured = Arc::new(Mutex::new(None));
    let upstream = spawn_server(
        Router::new()
            .fallback(any(capture_request))
            .with_state(Arc::clone(&captured)),
    )
    .await;
    let gateway = spawn_gateway_with_auth(
        upstream.addr,
        GatewayAuth::RequireToken {
            store: Arc::new(store),
            quota: None,
            usage: None,
        },
    )
    .await;
    let request = Request::get(format!("http://{}/v1/models", gateway.addr))
        .header("host", "public-gateway.example")
        .header("authorization", format!("Bearer {token}"))
        .header("x-client-test", "authorized")
        .body(Body::empty())
        .unwrap();

    let response = client().request(request).await.unwrap();

    assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);
    assert!(captured.lock().unwrap().is_some());
}

#[tokio::test]
async fn quota_rejects_a_request_once_the_organization_exceeds_its_limit() {
    let store = SqliteIdentityStore::open_in_memory().unwrap();
    let principal = store.create_user("ada").unwrap();
    let token = store.issue_token(&principal, TokenLifetime::Never).unwrap();

    let upstream_calls = Arc::new(AtomicUsize::new(0));
    let upstream = spawn_server(Router::new().fallback(any({
        let upstream_calls = Arc::clone(&upstream_calls);
        move || {
            let upstream_calls = Arc::clone(&upstream_calls);
            async move {
                upstream_calls.fetch_add(1, Ordering::SeqCst);
                StatusCode::NO_CONTENT
            }
        }
    })))
    .await;
    let gateway = spawn_gateway_with_auth(
        upstream.addr,
        GatewayAuth::RequireToken {
            store: Arc::new(store),
            quota: Some(Arc::new(OrgQuota::new(
                NonZeroU32::new(2).unwrap(),
                NonZeroU64::new(60).unwrap(),
            ))),
            usage: None,
        },
    )
    .await;

    let send = || {
        let addr = gateway.addr;
        let token = token.clone();
        async move {
            let request = Request::get(format!("http://{addr}/v1/models"))
                .header("authorization", format!("Bearer {token}"))
                .body(Body::empty())
                .unwrap();
            client().request(request).await.unwrap()
        }
    };

    assert_eq!(send().await.status(), StatusCode::NO_CONTENT);
    assert_eq!(send().await.status(), StatusCode::NO_CONTENT);

    let rejected = send().await;
    assert_eq!(rejected.status(), StatusCode::TOO_MANY_REQUESTS);
    let retry_after: u64 = rejected.headers()["retry-after"]
        .to_str()
        .unwrap()
        .parse()
        .unwrap();
    assert_eq!(retry_after, 60);
    assert_eq!(
        to_bytes(Body::new(rejected.into_body()), 1024)
            .await
            .unwrap(),
        r#"{"error":{"message":"organization request quota exceeded","type":"quota_exceeded"}}"#
    );
    // The rejected request never reached the upstream replica.
    assert_eq!(upstream_calls.load(Ordering::SeqCst), 2);
}

#[tokio::test]
async fn quota_tracks_organizations_independently() {
    let store = SqliteIdentityStore::open_in_memory().unwrap();
    let ada = store.create_user("ada").unwrap();
    let ada_token = store.issue_token(&ada, TokenLifetime::Never).unwrap();
    let grace = store.create_user("grace").unwrap();
    let grace_token = store.issue_token(&grace, TokenLifetime::Never).unwrap();

    let upstream =
        spawn_server(Router::new().fallback(any(|| async { StatusCode::NO_CONTENT }))).await;
    let gateway = spawn_gateway_with_auth(
        upstream.addr,
        GatewayAuth::RequireToken {
            store: Arc::new(store),
            quota: Some(Arc::new(OrgQuota::new(
                NonZeroU32::new(1).unwrap(),
                NonZeroU64::new(60).unwrap(),
            ))),
            usage: None,
        },
    )
    .await;

    let send = |token: String| {
        let addr = gateway.addr;
        async move {
            let request = Request::get(format!("http://{addr}/v1/models"))
                .header("authorization", format!("Bearer {token}"))
                .body(Body::empty())
                .unwrap();
            client().request(request).await.unwrap()
        }
    };

    assert_eq!(
        send(ada_token.clone()).await.status(),
        StatusCode::NO_CONTENT
    );
    // ada's organization has already used its one request for the window.
    assert_eq!(
        send(ada_token).await.status(),
        StatusCode::TOO_MANY_REQUESTS
    );
    // grace's organization has its own, unaffected budget.
    assert_eq!(send(grace_token).await.status(), StatusCode::NO_CONTENT);
}

#[tokio::test]
async fn quota_is_not_charged_by_requests_that_fail_authentication() {
    let store = SqliteIdentityStore::open_in_memory().unwrap();
    let principal = store.create_user("ada").unwrap();
    let token = store.issue_token(&principal, TokenLifetime::Never).unwrap();

    let upstream =
        spawn_server(Router::new().fallback(any(|| async { StatusCode::NO_CONTENT }))).await;
    let gateway = spawn_gateway_with_auth(
        upstream.addr,
        GatewayAuth::RequireToken {
            store: Arc::new(store),
            quota: Some(Arc::new(OrgQuota::new(
                NonZeroU32::new(1).unwrap(),
                NonZeroU64::new(60).unwrap(),
            ))),
            usage: None,
        },
    )
    .await;

    let unauthenticated = Request::get(format!("http://{}/v1/models", gateway.addr))
        .body(Body::empty())
        .unwrap();
    assert_eq!(
        client().request(unauthenticated).await.unwrap().status(),
        StatusCode::UNAUTHORIZED
    );

    // The organization's quota is untouched: its one request for the window
    // is still available.
    let authenticated = Request::get(format!("http://{}/v1/models", gateway.addr))
        .header("authorization", format!("Bearer {token}"))
        .body(Body::empty())
        .unwrap();
    assert_eq!(
        client().request(authenticated).await.unwrap().status(),
        StatusCode::NO_CONTENT
    );
}

#[tokio::test]
async fn quota_charges_a_request_rejected_by_admission_control() {
    let store = SqliteIdentityStore::open_in_memory().unwrap();
    let principal = store.create_user("ada").unwrap();
    let token = store.issue_token(&principal, TokenLifetime::Never).unwrap();
    let release = Arc::new(Notify::new());
    let upstream = spawn_server(
        Router::new()
            .route("/v1/chat/completions", any(delayed_sse))
            .with_state(Arc::clone(&release)),
    )
    .await;
    let gateway = spawn_gateway_with_options(
        upstream.addr,
        NonZeroUsize::new(1).unwrap(),
        Duration::from_secs(30),
        GatewayAuth::RequireToken {
            store: Arc::new(store),
            quota: Some(Arc::new(OrgQuota::new(
                NonZeroU32::new(2).unwrap(),
                NonZeroU64::new(60).unwrap(),
            ))),
            usage: None,
        },
        None,
    )
    .await;

    let request = || {
        Request::post(format!("http://{}/v1/chat/completions", gateway.addr))
            .header("authorization", format!("Bearer {token}"))
            .body(Body::empty())
            .unwrap()
    };

    let accepted = client().request(request()).await.unwrap();
    assert_eq!(accepted.status(), StatusCode::OK);

    // This request has a valid identity and passes quota, but the active
    // stream holds the only admission permit. It still consumes quota.
    assert_eq!(
        client().request(request()).await.unwrap().status(),
        StatusCode::SERVICE_UNAVAILABLE
    );
    drop(accepted);

    // Both prior valid requests were charged, including the admission 503.
    assert_eq!(
        client().request(request()).await.unwrap().status(),
        StatusCode::TOO_MANY_REQUESTS
    );
    release.notify_waiters();
}

#[tokio::test]
async fn quota_charges_a_request_when_the_upstream_is_unavailable() {
    let store = SqliteIdentityStore::open_in_memory().unwrap();
    let principal = store.create_user("ada").unwrap();
    let token = store.issue_token(&principal, TokenLifetime::Never).unwrap();
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let unavailable_upstream = listener.local_addr().unwrap();
    drop(listener);
    let gateway = spawn_gateway_with_auth(
        unavailable_upstream,
        GatewayAuth::RequireToken {
            store: Arc::new(store),
            quota: Some(Arc::new(OrgQuota::new(
                NonZeroU32::new(1).unwrap(),
                NonZeroU64::new(60).unwrap(),
            ))),
            usage: None,
        },
    )
    .await;

    let request = || {
        Request::get(format!("http://{}/v1/models", gateway.addr))
            .header("authorization", format!("Bearer {token}"))
            .body(Body::empty())
            .unwrap()
    };

    assert_eq!(
        client().request(request()).await.unwrap().status(),
        StatusCode::BAD_GATEWAY
    );
    assert_eq!(
        client().request(request()).await.unwrap().status(),
        StatusCode::TOO_MANY_REQUESTS
    );
}

#[tokio::test]
async fn rejects_non_inference_routes_without_contacting_upstream() {
    let upstream_calls = Arc::new(AtomicUsize::new(0));
    let upstream = spawn_server(Router::new().fallback(any({
        let upstream_calls = Arc::clone(&upstream_calls);
        move || {
            let upstream_calls = Arc::clone(&upstream_calls);
            async move {
                upstream_calls.fetch_add(1, Ordering::SeqCst);
                StatusCode::NO_CONTENT
            }
        }
    })))
    .await;
    let app = gateway_router(UpstreamOrigin::parse(&format!("http://{}", upstream.addr)).unwrap());

    for path in [
        "/api/models/unload",
        "/api/models/load",
        "/api/runtime/gpu",
        "/api/agent/workspace/browse",
        "/models/unload",
        "/v1%2fmodels",
        "/v1/%2e%2e/api/models/unload",
        "/v1/models%2f..%2f..%2fapi%2fmodels%2funload",
        "/",
        "/unknown",
    ] {
        let response = app
            .clone()
            .oneshot(Request::post(path).body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::NOT_FOUND, "path: {path}");
        assert_eq!(response.headers()["access-control-allow-origin"], "*");
    }

    assert_eq!(upstream_calls.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn rejects_wrong_methods_without_contacting_upstream() {
    let upstream_calls = Arc::new(AtomicUsize::new(0));
    let upstream = spawn_server(Router::new().fallback(any({
        let upstream_calls = Arc::clone(&upstream_calls);
        move || {
            let upstream_calls = Arc::clone(&upstream_calls);
            async move {
                upstream_calls.fetch_add(1, Ordering::SeqCst);
                StatusCode::NO_CONTENT
            }
        }
    })))
    .await;
    let app = gateway_router(UpstreamOrigin::parse(&format!("http://{}", upstream.addr)).unwrap());

    for (method, path) in [
        (Method::POST, "/v1/models"),
        (Method::DELETE, "/v1/models/model"),
        (Method::GET, "/v1/completions"),
        (Method::PATCH, "/v1/chat/completions"),
    ] {
        let response = app
            .clone()
            .oneshot(
                Request::builder()
                    .method(method)
                    .uri(path)
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(
            response.status(),
            StatusCode::METHOD_NOT_ALLOWED,
            "path: {path}"
        );
        assert_eq!(response.headers()["access-control-allow-origin"], "*");
    }

    assert_eq!(upstream_calls.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn head_requests_still_forward_to_the_replica() {
    let methods = Arc::new(Mutex::new(Vec::new()));
    let upstream = spawn_server(Router::new().fallback(any({
        let methods = Arc::clone(&methods);
        move |request: Request| {
            let methods = Arc::clone(&methods);
            async move {
                methods.lock().unwrap().push(request.method().clone());
                StatusCode::NO_CONTENT
            }
        }
    })))
    .await;
    let app = gateway_router(UpstreamOrigin::parse(&format!("http://{}", upstream.addr)).unwrap());

    let response = app
        .oneshot(
            Request::builder()
                .method(Method::HEAD)
                .uri("/v1/models")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::NO_CONTENT);

    assert_eq!(*methods.lock().unwrap(), [Method::HEAD]);
}

#[tokio::test]
async fn cors_preflight_is_answered_locally_without_contacting_the_replica() {
    let upstream_calls = Arc::new(AtomicUsize::new(0));
    let upstream = spawn_server(Router::new().fallback(any({
        let upstream_calls = Arc::clone(&upstream_calls);
        move || {
            let upstream_calls = Arc::clone(&upstream_calls);
            async move {
                upstream_calls.fetch_add(1, Ordering::SeqCst);
                StatusCode::NO_CONTENT
            }
        }
    })))
    .await;
    let app = gateway_router(UpstreamOrigin::parse(&format!("http://{}", upstream.addr)).unwrap());

    // A real browser preflight for a cross-origin, non-safelisted request
    // (JSON body) carries these two headers. Only their presence makes a
    // request a preflight in tower-http's CORS layer; the gateway does not
    // register an `OPTIONS` handler on any route, so a preflight must be
    // answered entirely by the CORS layer, before the router dispatches it.
    let response = app
        .oneshot(
            Request::builder()
                .method(Method::OPTIONS)
                .uri("/v1/chat/completions")
                .header("origin", "https://example.test")
                .header("access-control-request-method", "POST")
                .header("access-control-request-headers", "content-type")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(response.headers()["access-control-allow-origin"], "*");
    // `CorsLayer::permissive()` answers every preflight with the wildcard,
    // not a mirror of the requested method/headers.
    assert_eq!(response.headers()["access-control-allow-methods"], "*");
    assert_eq!(response.headers()["access-control-allow-headers"], "*");
    assert_eq!(upstream_calls.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn local_health_does_not_contact_upstream_or_consume_admission() {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let unavailable = listener.local_addr().unwrap();
    drop(listener);
    let app = router_with_max_in_flight(
        UpstreamOrigin::parse(&format!("http://{unavailable}")).unwrap(),
        NonZeroUsize::new(1).unwrap(),
    );

    for _ in 0..2 {
        let response = app
            .clone()
            .oneshot(Request::get("/healthz").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::NO_CONTENT);
    }
}

#[tokio::test]
async fn concurrency_limit_is_held_for_the_full_response_stream() {
    let release = Arc::new(Notify::new());
    let upstream = spawn_server(
        Router::new()
            .route("/v1/chat/completions", any(delayed_sse))
            .with_state(Arc::clone(&release)),
    )
    .await;
    let app = router_with_max_in_flight(
        UpstreamOrigin::parse(&format!("http://{}", upstream.addr)).unwrap(),
        NonZeroUsize::new(1).unwrap(),
    );

    let first = app
        .clone()
        .oneshot(
            Request::post("/v1/chat/completions")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(first.status(), StatusCode::OK);

    let rejected = app
        .clone()
        .oneshot(
            Request::post("/v1/chat/completions")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(rejected.status(), StatusCode::SERVICE_UNAVAILABLE);
    let retry_after: u64 = rejected.headers()["retry-after"]
        .to_str()
        .unwrap()
        .parse()
        .expect("retry-after must be a decimal integer of seconds");
    assert!(
        (1..=3).contains(&retry_after),
        "retry-after must be jittered within [1, 3], got {retry_after}"
    );
    assert_eq!(rejected.headers()["access-control-allow-origin"], "*");

    drop(first);
    let admitted = app
        .oneshot(
            Request::post("/v1/chat/completions")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(admitted.status(), StatusCode::OK);
    release.notify_waiters();
}

#[tokio::test]
async fn concurrency_limit_is_released_at_response_eof_before_body_drop() {
    let upstream =
        spawn_server(Router::new().route("/v1/chat/completions", post(|| async { "complete" })))
            .await;
    let app = router_with_max_in_flight(
        UpstreamOrigin::parse(&format!("http://{}", upstream.addr)).unwrap(),
        NonZeroUsize::new(1).unwrap(),
    );

    let first = app
        .clone()
        .oneshot(
            Request::post("/v1/chat/completions")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    let mut completed_body = first.into_body();
    while let Some(frame) = completed_body.frame().await {
        frame.unwrap();
    }

    let admitted = app
        .oneshot(
            Request::post("/v1/chat/completions")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(admitted.status(), StatusCode::OK);
    drop(completed_body);
}

#[tokio::test]
async fn concurrency_limit_is_released_for_an_immediately_complete_response() {
    let upstream = spawn_server(Router::new().route(
        "/v1/chat/completions",
        post(|| async { StatusCode::NO_CONTENT }),
    ))
    .await;
    let app = router_with_max_in_flight(
        UpstreamOrigin::parse(&format!("http://{}", upstream.addr)).unwrap(),
        NonZeroUsize::new(1).unwrap(),
    );

    let completed = app
        .clone()
        .oneshot(
            Request::post("/v1/chat/completions")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(completed.status(), StatusCode::NO_CONTENT);

    let admitted = app
        .oneshot(
            Request::post("/v1/chat/completions")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(admitted.status(), StatusCode::NO_CONTENT);
    drop(completed);
}

#[tokio::test]
async fn graceful_shutdown_waits_for_an_active_response_stream() {
    let release = Arc::new(Notify::new());
    let upstream = spawn_server(
        Router::new()
            .route("/v1/chat/completions", any(delayed_sse))
            .with_state(Arc::clone(&release)),
    )
    .await;
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let gateway_addr = listener.local_addr().unwrap();
    let app = gateway_router(UpstreamOrigin::parse(&format!("http://{}", upstream.addr)).unwrap());
    let (shutdown_tx, shutdown_rx) = oneshot::channel();
    let server = tokio::spawn(async move {
        camelid_enterprise_gateway::serve(listener, app, Duration::from_secs(30), async {
            let _ = shutdown_rx.await;
        })
        .await
        .unwrap();
    });
    let request = Request::post(format!("http://{gateway_addr}/v1/chat/completions"))
        .body(Body::empty())
        .unwrap();
    let response = client().request(request).await.unwrap();
    let mut body = response.into_body();
    let first = body.frame().await.unwrap().unwrap().into_data().unwrap();
    assert_eq!(first, "data: first\n\n");

    shutdown_tx.send(()).unwrap();
    tokio::task::yield_now().await;
    assert!(
        !server.is_finished(),
        "graceful shutdown must wait for the active response stream"
    );

    release.notify_one();
    while let Some(frame) = body.frame().await {
        frame.unwrap();
    }
    drop(body);
    server.await.unwrap();
}

#[tokio::test]
async fn stalled_connection_is_closed_after_the_maximum_duration_and_releases_its_permit() {
    let release = Arc::new(Notify::new());
    let upstream = spawn_server(
        Router::new()
            .route("/v1/chat/completions", any(delayed_sse))
            .with_state(Arc::clone(&release)),
    )
    .await;
    let gateway = spawn_gateway_with_limits(
        upstream.addr,
        NonZeroUsize::new(1).unwrap(),
        Duration::from_millis(300),
    )
    .await;

    // The first request never finishes: `delayed_sse` sends one chunk, then
    // waits on a `Notify` that this test never signals, and the client below
    // never reads past that first chunk either — exactly like a client that
    // stops reading its response. Nothing here drops the body, so the only
    // thing that can free the admission permit is the gateway's own
    // maximum-connection-duration cap.
    let first_request = Request::post(format!("http://{}/v1/chat/completions", gateway.addr))
        .body(Body::empty())
        .unwrap();
    let first_response = client().request(first_request).await.unwrap();
    assert_eq!(first_response.status(), StatusCode::OK);
    let mut first_body = first_response.into_body();
    let first_chunk = first_body
        .frame()
        .await
        .unwrap()
        .unwrap()
        .into_data()
        .unwrap();
    assert_eq!(first_chunk, "data: first\n\n");

    let second_request = Request::post(format!("http://{}/v1/chat/completions", gateway.addr))
        .body(Body::empty())
        .unwrap();
    let second_response = client().request(second_request).await.unwrap();
    assert_eq!(
        second_response.status(),
        StatusCode::SERVICE_UNAVAILABLE,
        "the single admission permit must still be pinned by the stalled first request"
    );

    tokio::time::sleep(Duration::from_millis(600)).await;

    let third_request = Request::post(format!("http://{}/v1/chat/completions", gateway.addr))
        .body(Body::empty())
        .unwrap();
    let third_response = client().request(third_request).await.unwrap();
    assert_eq!(
        third_response.status(),
        StatusCode::OK,
        "the gateway must have force-closed the stalled connection and released its permit"
    );

    drop(first_body);
}

/// Upstream handler that records the `x-camelid-request-id` header the gateway
/// forwarded (if any) and returns `204 No Content`.
async fn capture_request_id(
    State(seen): State<Arc<Mutex<Option<String>>>>,
    request: Request,
) -> Response {
    *seen.lock().unwrap() = request
        .headers()
        .get("x-camelid-request-id")
        .map(|value| value.to_str().unwrap().to_string());
    Response::builder()
        .status(StatusCode::NO_CONTENT)
        .body(Body::empty())
        .unwrap()
}

#[tokio::test]
async fn stamps_a_gateway_request_id_on_every_forwarded_request() {
    let seen = Arc::new(Mutex::new(None));
    let upstream = spawn_server(
        Router::new()
            .fallback(any(capture_request_id))
            .with_state(Arc::clone(&seen)),
    )
    .await;
    let gateway = spawn_gateway(upstream.addr).await;
    let request = Request::get(format!("http://{}/v1/models", gateway.addr))
        .body(Body::empty())
        .unwrap();

    let response = client().request(request).await.unwrap();
    assert_eq!(response.status(), StatusCode::NO_CONTENT);

    let request_id = seen
        .lock()
        .unwrap()
        .take()
        .expect("the gateway must stamp a request id on the forwarded request");
    assert!(
        request_id.starts_with("req_") && request_id.len() == "req_".len() + 32,
        "unexpected correlation id shape: {request_id}"
    );
}

#[tokio::test]
async fn overwrites_a_client_supplied_request_id() {
    let seen = Arc::new(Mutex::new(None));
    let upstream = spawn_server(
        Router::new()
            .fallback(any(capture_request_id))
            .with_state(Arc::clone(&seen)),
    )
    .await;
    let gateway = spawn_gateway(upstream.addr).await;
    let request = Request::get(format!("http://{}/v1/models", gateway.addr))
        .header("x-camelid-request-id", "req_spoofed-by-client")
        .body(Body::empty())
        .unwrap();

    client().request(request).await.unwrap();

    let request_id = seen.lock().unwrap().take().unwrap();
    assert_ne!(
        request_id, "req_spoofed-by-client",
        "the gateway must not trust a client-supplied correlation id"
    );
    assert!(request_id.starts_with("req_"));
}

#[tokio::test]
async fn audit_log_records_both_forwarded_and_rejected_requests() {
    let store = SqliteIdentityStore::open_in_memory().unwrap();
    let principal = store.create_user("ada").unwrap();
    let token = store.issue_token(&principal, TokenLifetime::Never).unwrap();
    let organization = store.organizations_for_principal(&principal).unwrap();
    assert_eq!(organization.len(), 1);

    let seen = Arc::new(Mutex::new(None));
    let upstream = spawn_server(
        Router::new()
            .fallback(any(capture_request_id))
            .with_state(Arc::clone(&seen)),
    )
    .await;

    let dir = tempfile::tempdir().unwrap();
    let audit_path = dir.path().join("audit.jsonl");
    let gateway = spawn_gateway_with_audit(
        upstream.addr,
        GatewayAuth::RequireToken {
            store: Arc::new(store),
            quota: None,
            usage: None,
        },
        gateway_log(&audit_path),
    )
    .await;

    // A request rejected before it is forwarded (no bearer token) must still be
    // audited, with a null principal.
    let rejected = Request::get(format!("http://{}/v1/models", gateway.addr))
        .body(Body::empty())
        .unwrap();
    assert_eq!(
        client().request(rejected).await.unwrap().status(),
        StatusCode::UNAUTHORIZED
    );

    // A forwarded request records the resolved principal and the same
    // correlation id the upstream received.
    let authorized = Request::get(format!("http://{}/v1/models", gateway.addr))
        .header("authorization", format!("Bearer {token}"))
        .body(Body::empty())
        .unwrap();
    assert_eq!(
        client().request(authorized).await.unwrap().status(),
        StatusCode::NO_CONTENT
    );
    let upstream_request_id = seen.lock().unwrap().take().unwrap();

    let records = read_audit_records(&audit_path, 2).await;

    let rejected_record = records
        .iter()
        .find(|record| record["status"] == 401)
        .expect("the rejected request must be audited");
    assert!(
        rejected_record["principal"].is_null(),
        "a rejected request has no resolved principal: {rejected_record}"
    );
    assert!(
        rejected_record["organization"].is_null(),
        "a rejected request has no resolved organization: {rejected_record}"
    );
    assert_eq!(rejected_record["method"], "GET");
    assert_eq!(rejected_record["path"], "/v1/models");
    assert!(rejected_record["request_id"]
        .as_str()
        .unwrap()
        .starts_with("req_"));

    let forwarded_record = records
        .iter()
        .find(|record| record["status"] == 204)
        .expect("the forwarded request must be audited");
    assert_eq!(forwarded_record["principal"], principal.to_string());
    assert_eq!(
        forwarded_record["organization"],
        organization[0].to_string()
    );
    assert_eq!(forwarded_record["request_id"], upstream_request_id);
    assert!(
        rejected_record["reason"] == "missing_token",
        "a refusal names itself: {rejected_record}"
    );
    assert!(
        forwarded_record["reason"].is_null(),
        "a forwarded request has no gateway refusal to name: {forwarded_record}"
    );
}

/// Both refusals are `401` with a null principal, so without a recorded reason
/// their audit lines are byte-identical and an operator cannot tell a client
/// using a lapsed credential from one presenting an unknown secret.
#[tokio::test]
async fn audit_log_distinguishes_an_expired_token_from_an_invalid_one() {
    let store = SqliteIdentityStore::open_in_memory().unwrap();
    let principal = store.create_user("ada").unwrap();
    let expired = store
        .issue_token(&principal, TokenLifetime::Until(identity::unix_now() - 1))
        .unwrap();
    let unreachable_upstream = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .unwrap()
        .local_addr()
        .unwrap();

    let dir = tempfile::tempdir().unwrap();
    let audit_path = dir.path().join("audit.jsonl");
    let gateway = spawn_gateway_with_audit(
        unreachable_upstream,
        GatewayAuth::RequireToken {
            store: Arc::new(store),
            quota: None,
            usage: None,
        },
        gateway_log(&audit_path),
    )
    .await;

    for token in [expired.as_str(), "cme_never_issued"] {
        let request = Request::get(format!("http://{}/v1/models", gateway.addr))
            .header("authorization", format!("Bearer {token}"))
            .body(Body::empty())
            .unwrap();
        assert_eq!(
            client().request(request).await.unwrap().status(),
            StatusCode::UNAUTHORIZED
        );
    }

    let records = read_audit_records(&audit_path, 2).await;
    let reasons: Vec<&str> = records
        .iter()
        .map(|record| {
            record["reason"]
                .as_str()
                .expect("every refusal names itself")
        })
        .collect();
    assert!(
        reasons.contains(&"expired_token") && reasons.contains(&"invalid_token"),
        "the two refusals must be distinguishable in the audit log: {reasons:?}"
    );
    for record in &records {
        assert_eq!(record["status"], 401);
        assert!(record["principal"].is_null());
    }
}

/// Under concurrent load the audit writer must produce exactly one complete,
/// independently parseable line per request, each with a unique correlation id,
/// proving the whole-record append is serialized (no torn or interleaved lines)
/// and that minted ids do not collide.
#[tokio::test]
async fn audit_log_is_not_torn_by_concurrent_requests() {
    const REQUESTS: usize = 48;
    let upstream = spawn_server(Router::new().fallback(any(|| async {
        Response::builder()
            .status(StatusCode::NO_CONTENT)
            .body(Body::empty())
            .unwrap()
    })))
    .await;
    let dir = tempfile::tempdir().unwrap();
    let audit_path = dir.path().join("audit.jsonl");
    let gateway = spawn_gateway_with_audit(
        upstream.addr,
        GatewayAuth::Disabled,
        gateway_log(&audit_path),
    )
    .await;

    let mut tasks = tokio::task::JoinSet::new();
    for _ in 0..REQUESTS {
        let addr = gateway.addr;
        tasks.spawn(async move {
            client()
                .request(
                    Request::get(format!("http://{addr}/v1/models"))
                        .body(Body::empty())
                        .unwrap(),
                )
                .await
                .unwrap()
                .status()
        });
    }
    while let Some(result) = tasks.join_next().await {
        assert_eq!(result.unwrap(), StatusCode::NO_CONTENT);
    }

    // read_audit_records parses every line; a torn line would panic here.
    let records = read_audit_records(&audit_path, REQUESTS).await;
    assert_eq!(
        records.len(),
        REQUESTS,
        "each request must append exactly one parseable audit line"
    );
    let ids: std::collections::HashSet<&str> = records
        .iter()
        .map(|record| record["request_id"].as_str().unwrap())
        .collect();
    assert_eq!(
        ids.len(),
        REQUESTS,
        "every minted correlation id must be unique"
    );
    for record in &records {
        assert_eq!(record["path"], "/v1/models");
        assert_eq!(record["status"], 204);
        assert!(record["principal"].is_null());
    }
}

#[tokio::test]
async fn usage_log_records_completed_authenticated_stream_bytes_without_changing_audit() {
    let store = SqliteIdentityStore::open_in_memory().unwrap();
    let principal = store.create_user("ada").unwrap();
    let organization = store
        .organizations_for_principal(&principal)
        .unwrap()
        .remove(0);
    let token = store.issue_token(&principal, TokenLifetime::Never).unwrap();
    let upstream =
        spawn_server(Router::new().route("/v1/chat/completions", post(usage_response))).await;
    let dir = tempfile::tempdir().unwrap();
    let usage_path = dir.path().join("usage.jsonl");
    let audit_path = dir.path().join("audit.jsonl");
    let gateway = spawn_gateway_with_options(
        upstream.addr,
        DEFAULT_MAX_IN_FLIGHT,
        Duration::from_secs(30),
        GatewayAuth::RequireToken {
            store: Arc::new(store),
            quota: None,
            usage: Some(gateway_log(&usage_path)),
        },
        Some(gateway_log(&audit_path)),
    )
    .await;

    let response = client()
        .request(
            Request::post(format!("http://{}/v1/chat/completions", gateway.addr))
                .header("authorization", format!("Bearer {token}"))
                .body(Body::from("prompt"))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        to_bytes(Body::new(response.into_body()), 1024)
            .await
            .unwrap(),
        "generated"
    );

    let usage = read_usage_records(&usage_path, 1).await;
    let record = &usage[0];
    assert_eq!(record["principal"], principal.to_string());
    assert_eq!(record["organization"], organization.to_string());
    assert_eq!(record["method"], "POST");
    assert_eq!(record["path"], "/v1/chat/completions");
    assert_eq!(record["response_head_status"], 200);
    assert_eq!(record["request_bytes"], 6);
    assert_eq!(record["response_bytes"], 9);
    assert_eq!(record["stream_outcome"], "completed");
    assert!(record["started_ts"].as_f64().unwrap() <= record["ts"].as_f64().unwrap());
    assert!(
        record["duration_ms"].as_u64().unwrap() >= 50,
        "duration must include the upstream response-head wait: {record}"
    );

    let audit = read_audit_records(&audit_path, 1).await;
    assert_eq!(audit[0]["request_id"], record["request_id"]);
    assert_eq!(audit[0]["status"], 200);
    assert!(audit[0].get("stream_outcome").is_none());
    assert!(audit[0].get("request_bytes").is_none());
}

#[tokio::test]
async fn usage_log_records_partial_response_when_the_upstream_body_errors() {
    let store = SqliteIdentityStore::open_in_memory().unwrap();
    let principal = store.create_user("ada").unwrap();
    let token = store.issue_token(&principal, TokenLifetime::Never).unwrap();
    let release = Arc::new(Notify::new());
    let upstream = spawn_server(
        Router::new()
            .route("/v1/chat/completions", post(delayed_failing_usage_response))
            .with_state(Arc::clone(&release)),
    )
    .await;
    let dir = tempfile::tempdir().unwrap();
    let usage_path = dir.path().join("usage.jsonl");
    let gateway = spawn_gateway_with_auth(
        upstream.addr,
        GatewayAuth::RequireToken {
            store: Arc::new(store),
            quota: None,
            usage: Some(gateway_log(&usage_path)),
        },
    )
    .await;

    let response = client()
        .request(
            Request::post(format!("http://{}/v1/chat/completions", gateway.addr))
                .header("authorization", format!("Bearer {token}"))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let mut body = response.into_body();
    let first = body.frame().await.unwrap().unwrap().into_data().unwrap();
    assert_eq!(first, "partial");
    release.notify_one();
    assert!(
        body.frame().await.unwrap().is_err(),
        "the client must observe the upstream stream failure"
    );
    drop(body);

    let usage = read_usage_records(&usage_path, 1).await;
    assert_eq!(usage[0]["request_bytes"], 0);
    assert_eq!(usage[0]["response_bytes"], 7);
    assert_eq!(usage[0]["stream_outcome"], "body_error");
}

#[tokio::test]
async fn usage_log_records_an_incomplete_response_stream() {
    let store = SqliteIdentityStore::open_in_memory().unwrap();
    let principal = store.create_user("ada").unwrap();
    let token = store.issue_token(&principal, TokenLifetime::Never).unwrap();
    let upstream_dropped = Arc::new(Notify::new());
    let upstream = spawn_server(
        Router::new()
            .route("/v1/chat/completions", any(disconnect_aware_sse))
            .with_state(Arc::clone(&upstream_dropped)),
    )
    .await;
    let dir = tempfile::tempdir().unwrap();
    let usage_path = dir.path().join("usage.jsonl");
    let gateway = spawn_gateway_with_auth(
        upstream.addr,
        GatewayAuth::RequireToken {
            store: Arc::new(store),
            quota: None,
            usage: Some(gateway_log(&usage_path)),
        },
    )
    .await;

    let response = client()
        .request(
            Request::post(format!("http://{}/v1/chat/completions", gateway.addr))
                .header("authorization", format!("Bearer {token}"))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    let mut body = response.into_body();
    let first = body.frame().await.unwrap().unwrap().into_data().unwrap();
    assert_eq!(first, "data: first\n\n");
    drop(body);

    tokio::time::timeout(Duration::from_secs(1), upstream_dropped.notified())
        .await
        .expect("dropping the client stream must cancel the upstream stream");
    let usage = read_usage_records(&usage_path, 1).await;
    assert_eq!(usage[0]["response_bytes"], 13);
    assert_eq!(usage[0]["stream_outcome"], "incomplete");
}

#[tokio::test]
async fn usage_log_records_a_gateway_timeout_separately_from_an_incomplete_stream() {
    let store = SqliteIdentityStore::open_in_memory().unwrap();
    let principal = store.create_user("ada").unwrap();
    let token = store.issue_token(&principal, TokenLifetime::Never).unwrap();
    let release = Arc::new(Notify::new());
    let upstream = spawn_server(
        Router::new()
            .route("/v1/chat/completions", any(delayed_sse))
            .with_state(Arc::clone(&release)),
    )
    .await;
    let dir = tempfile::tempdir().unwrap();
    let usage_path = dir.path().join("usage.jsonl");
    let gateway = spawn_gateway_with_options(
        upstream.addr,
        NonZeroUsize::new(1).unwrap(),
        Duration::from_millis(150),
        GatewayAuth::RequireToken {
            store: Arc::new(store),
            quota: None,
            usage: Some(gateway_log(&usage_path)),
        },
        None,
    )
    .await;

    let response = client()
        .request(
            Request::post(format!("http://{}/v1/chat/completions", gateway.addr))
                .header("authorization", format!("Bearer {token}"))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    let mut body = response.into_body();
    let first = body.frame().await.unwrap().unwrap().into_data().unwrap();
    assert_eq!(first, "data: first\n\n");

    let usage = read_usage_records(&usage_path, 1).await;
    assert_eq!(usage[0]["stream_outcome"], "gateway_timeout");
    drop(body);
    release.notify_waiters();
}

#[tokio::test]
async fn usage_log_records_gateway_failures_after_authentication_and_quota() {
    let store = SqliteIdentityStore::open_in_memory().unwrap();
    let principal = store.create_user("ada").unwrap();
    let token = store.issue_token(&principal, TokenLifetime::Never).unwrap();
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let unavailable_upstream = listener.local_addr().unwrap();
    drop(listener);
    let dir = tempfile::tempdir().unwrap();
    let usage_path = dir.path().join("usage.jsonl");
    let gateway = spawn_gateway_with_auth(
        unavailable_upstream,
        GatewayAuth::RequireToken {
            store: Arc::new(store),
            quota: Some(Arc::new(OrgQuota::new(
                NonZeroU32::new(2).unwrap(),
                NonZeroU64::new(60).unwrap(),
            ))),
            usage: Some(gateway_log(&usage_path)),
        },
    )
    .await;

    let response = client()
        .request(
            Request::get(format!("http://{}/v1/models", gateway.addr))
                .header("authorization", format!("Bearer {token}"))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::BAD_GATEWAY);
    let _ = to_bytes(Body::new(response.into_body()), 1024)
        .await
        .unwrap();

    let usage = read_usage_records(&usage_path, 1).await;
    assert_eq!(usage[0]["response_head_status"], 502);
    assert_eq!(usage[0]["request_bytes"], 0);
    assert!(usage[0]["response_bytes"].as_u64().unwrap() > 0);
    assert_eq!(usage[0]["stream_outcome"], "completed");
}

#[tokio::test]
async fn usage_log_records_an_admission_rejection_after_quota_admission() {
    let store = SqliteIdentityStore::open_in_memory().unwrap();
    let principal = store.create_user("ada").unwrap();
    let token = store.issue_token(&principal, TokenLifetime::Never).unwrap();
    let release = Arc::new(Notify::new());
    let upstream = spawn_server(
        Router::new()
            .route("/v1/chat/completions", any(delayed_sse))
            .with_state(Arc::clone(&release)),
    )
    .await;
    let dir = tempfile::tempdir().unwrap();
    let usage_path = dir.path().join("usage.jsonl");
    let gateway = spawn_gateway_with_options(
        upstream.addr,
        NonZeroUsize::new(1).unwrap(),
        Duration::from_secs(30),
        GatewayAuth::RequireToken {
            store: Arc::new(store),
            quota: Some(Arc::new(OrgQuota::new(
                NonZeroU32::new(2).unwrap(),
                NonZeroU64::new(60).unwrap(),
            ))),
            usage: Some(gateway_log(&usage_path)),
        },
        None,
    )
    .await;

    let request = || {
        Request::post(format!("http://{}/v1/chat/completions", gateway.addr))
            .header("authorization", format!("Bearer {token}"))
            .body(Body::empty())
            .unwrap()
    };
    let active = client().request(request()).await.unwrap();
    let rejected = client()
        .request(
            Request::post(format!("http://{}/v1/chat/completions", gateway.addr))
                .header("authorization", format!("Bearer {token}"))
                .body(Body::from("this body was not forwarded"))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(rejected.status(), StatusCode::SERVICE_UNAVAILABLE);
    let _ = to_bytes(Body::new(rejected.into_body()), 1024)
        .await
        .unwrap();
    drop(active);
    release.notify_waiters();

    let usage = read_usage_records(&usage_path, 2).await;
    let rejected = usage
        .iter()
        .find(|record| record["response_head_status"] == 503)
        .expect("the quota-admitted admission rejection must be recorded");
    assert!(
        rejected["request_bytes"].is_null(),
        "an admission rejection must not claim it measured an unforwarded body"
    );
    assert!(rejected["response_bytes"].as_u64().unwrap() > 0);
    assert_eq!(rejected["stream_outcome"], "completed");
}

#[tokio::test]
async fn usage_log_excludes_authentication_and_quota_rejections() {
    let store = SqliteIdentityStore::open_in_memory().unwrap();
    let principal = store.create_user("ada").unwrap();
    let token = store.issue_token(&principal, TokenLifetime::Never).unwrap();
    let upstream =
        spawn_server(Router::new().fallback(any(|| async { StatusCode::NO_CONTENT }))).await;
    let dir = tempfile::tempdir().unwrap();
    let usage_path = dir.path().join("usage.jsonl");
    let gateway = spawn_gateway_with_auth(
        upstream.addr,
        GatewayAuth::RequireToken {
            store: Arc::new(store),
            quota: Some(Arc::new(OrgQuota::new(
                NonZeroU32::new(1).unwrap(),
                NonZeroU64::new(60).unwrap(),
            ))),
            usage: Some(gateway_log(&usage_path)),
        },
    )
    .await;

    let unauthenticated = Request::post(format!("http://{}/v1/chat/completions", gateway.addr))
        .body(Body::from("unattributed body"))
        .unwrap();
    assert_eq!(
        client().request(unauthenticated).await.unwrap().status(),
        StatusCode::UNAUTHORIZED
    );

    let accepted = Request::get(format!("http://{}/v1/models", gateway.addr))
        .header("authorization", format!("Bearer {token}"))
        .body(Body::empty())
        .unwrap();
    assert_eq!(
        client().request(accepted).await.unwrap().status(),
        StatusCode::NO_CONTENT
    );

    let over_quota = Request::post(format!("http://{}/v1/chat/completions", gateway.addr))
        .header("authorization", format!("Bearer {token}"))
        .body(Body::from("over quota body"))
        .unwrap();
    assert_eq!(
        client().request(over_quota).await.unwrap().status(),
        StatusCode::TOO_MANY_REQUESTS
    );

    let usage = read_usage_records(&usage_path, 1).await;
    assert_eq!(usage.len(), 1);
    assert_eq!(usage[0]["response_head_status"], 204);
    assert_eq!(usage[0]["stream_outcome"], "completed");
}

#[tokio::test]
async fn usage_log_is_not_torn_by_concurrent_completed_requests() {
    const REQUESTS: usize = 48;
    let store = SqliteIdentityStore::open_in_memory().unwrap();
    let principal = store.create_user("ada").unwrap();
    let token = store.issue_token(&principal, TokenLifetime::Never).unwrap();
    let upstream =
        spawn_server(Router::new().fallback(any(|| async { StatusCode::NO_CONTENT }))).await;
    let dir = tempfile::tempdir().unwrap();
    let usage_path = dir.path().join("usage.jsonl");
    let gateway = spawn_gateway_with_auth(
        upstream.addr,
        GatewayAuth::RequireToken {
            store: Arc::new(store),
            quota: None,
            usage: Some(gateway_log(&usage_path)),
        },
    )
    .await;

    let mut tasks = tokio::task::JoinSet::new();
    for _ in 0..REQUESTS {
        let addr = gateway.addr;
        let token = token.clone();
        tasks.spawn(async move {
            client()
                .request(
                    Request::get(format!("http://{addr}/v1/models"))
                        .header("authorization", format!("Bearer {token}"))
                        .body(Body::empty())
                        .unwrap(),
                )
                .await
                .unwrap()
                .status()
        });
    }
    while let Some(result) = tasks.join_next().await {
        assert_eq!(result.unwrap(), StatusCode::NO_CONTENT);
    }

    // read_usage_records parses every line; a torn or interleaved append would
    // panic before these assertions run.
    let usage = read_usage_records(&usage_path, REQUESTS).await;
    assert_eq!(usage.len(), REQUESTS);
    let ids: std::collections::HashSet<&str> = usage
        .iter()
        .map(|record| record["request_id"].as_str().unwrap())
        .collect();
    assert_eq!(ids.len(), REQUESTS);
    for record in &usage {
        assert_eq!(record["principal"], principal.to_string());
        assert_eq!(record["response_head_status"], 204);
        assert_eq!(record["request_bytes"], 0);
        assert_eq!(record["response_bytes"], 0);
        assert_eq!(record["stream_outcome"], "completed");
    }
}

/// Usage records are written after stream completion, so this awaits their
/// asynchronous best-effort append rather than assuming the file is ready as
/// soon as the HTTP response is dropped.
async fn read_usage_records(path: &std::path::Path, expected: usize) -> Vec<serde_json::Value> {
    for _ in 0..200 {
        if let Ok(contents) = std::fs::read_to_string(path) {
            let lines: Vec<&str> = contents
                .lines()
                .filter(|line| !line.trim().is_empty())
                .collect();
            if lines.len() >= expected {
                return lines
                    .iter()
                    .map(|line| serde_json::from_str(line).unwrap())
                    .collect();
            }
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    panic!("expected {expected} usage line(s) were not written within the timeout");
}

async fn read_audit_records(path: &std::path::Path, expected: usize) -> Vec<serde_json::Value> {
    for _ in 0..200 {
        if let Ok(contents) = std::fs::read_to_string(path) {
            let lines: Vec<&str> = contents
                .lines()
                .filter(|line| !line.trim().is_empty())
                .collect();
            if lines.len() >= expected {
                return lines
                    .iter()
                    .map(|line| serde_json::from_str(line).unwrap())
                    .collect();
            }
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    panic!("expected {expected} audit line(s) were not written within the timeout");
}

/// Upstream handler that records the method and path of every request it
/// receives, so a test can assert exactly which requests the gateway forwarded.
async fn record_method_and_path(
    State(seen): State<ForwardedRequests>,
    request: Request,
) -> Response {
    seen.lock()
        .unwrap()
        .push((request.method().clone(), request.uri().path().to_string()));
    Response::builder()
        .status(StatusCode::NO_CONTENT)
        .body(Body::empty())
        .unwrap()
}

/// Every `(method, path)` an upstream test handler saw the gateway forward.
type ForwardedRequests = Arc<Mutex<Vec<(Method, String)>>>;

#[derive(Clone)]
struct CatalogUpstream {
    name: &'static str,
    calls: Arc<AtomicUsize>,
}

fn catalog_model_list_response(model_id: &str) -> Response {
    Response::builder()
        .status(StatusCode::OK)
        .header("content-type", "application/json")
        .body(Body::from(
            serde_json::json!({
                "object": "list",
                "data": [{ "id": model_id }],
            })
            .to_string(),
        ))
        .unwrap()
}

async fn catalog_upstream(State(state): State<CatalogUpstream>, request: Request) -> Response {
    if request.method() == Method::GET && request.uri().path() == "/v1/models" {
        return catalog_model_list_response(state.name);
    }
    state.calls.fetch_add(1, Ordering::SeqCst);
    let body = to_bytes(request.into_body(), 4 * 1024 * 1024)
        .await
        .unwrap();
    Response::builder()
        .status(StatusCode::OK)
        .header("x-test-catalog-upstream", state.name)
        .body(Body::from(body))
        .unwrap()
}

#[tokio::test]
async fn catalog_routes_generation_by_json_model_and_preserves_request_bytes() {
    let alpha_calls = Arc::new(AtomicUsize::new(0));
    let alpha = spawn_server(Router::new().fallback(any(catalog_upstream)).with_state(
        CatalogUpstream {
            name: "alpha",
            calls: Arc::clone(&alpha_calls),
        },
    ))
    .await;
    let bravo_calls = Arc::new(AtomicUsize::new(0));
    let bravo = spawn_server(Router::new().fallback(any(catalog_upstream)).with_state(
        CatalogUpstream {
            name: "bravo",
            calls: Arc::clone(&bravo_calls),
        },
    ))
    .await;
    let gateway = spawn_catalog_gateway(&[("alpha", alpha.addr), ("bravo", bravo.addr)]).await;

    // Query parameters are preserved for compatibility, but only the JSON
    // field selects an origin. A client therefore cannot use `?model=` to
    // redirect a request around the catalog.
    let chat_body =
        r#"{"model":"alpha","messages":[{"role":"user","content":"hello"}],"stream":false}"#;
    let chat = Request::post(format!(
        "http://{}/v1/chat/completions?model=bravo",
        gateway.addr
    ))
    .header("content-type", "application/json")
    .body(Body::from(chat_body))
    .unwrap();
    let chat_response = client().request(chat).await.unwrap();
    assert_eq!(chat_response.status(), StatusCode::OK);
    assert_eq!(chat_response.headers()["x-test-catalog-upstream"], "alpha");
    assert_eq!(
        to_bytes(Body::new(chat_response.into_body()), 4 * 1024 * 1024)
            .await
            .unwrap(),
        chat_body
    );

    // A JSON serializer may escape part of a valid backend id. Selection must
    // compare the decoded value to the verified catalog id, while forwarding
    // the original bytes unchanged.
    let escaped_chat_body = r#"{"model":"alph\u0061","messages":[]}"#;
    let escaped_chat = Request::post(format!("http://{}/v1/chat/completions", gateway.addr))
        .header("content-type", "application/json")
        .body(Body::from(escaped_chat_body))
        .unwrap();
    let escaped_chat_response = client().request(escaped_chat).await.unwrap();
    assert_eq!(escaped_chat_response.status(), StatusCode::OK);
    assert_eq!(
        escaped_chat_response.headers()["x-test-catalog-upstream"],
        "alpha"
    );
    assert_eq!(
        to_bytes(
            Body::new(escaped_chat_response.into_body()),
            4 * 1024 * 1024
        )
        .await
        .unwrap(),
        escaped_chat_body
    );

    let completion_body = r#"{"model":"bravo","prompt":"2+2=","max_tokens":4}"#;
    let completion = Request::post(format!("http://{}/v1/completions", gateway.addr))
        .header("content-type", "application/json")
        .body(Body::from(completion_body))
        .unwrap();
    let completion_response = client().request(completion).await.unwrap();
    assert_eq!(completion_response.status(), StatusCode::OK);
    assert_eq!(
        completion_response.headers()["x-test-catalog-upstream"],
        "bravo"
    );
    assert_eq!(
        to_bytes(Body::new(completion_response.into_body()), 4 * 1024 * 1024)
            .await
            .unwrap(),
        completion_body
    );
    assert_eq!(alpha_calls.load(Ordering::SeqCst), 2);
    assert_eq!(bravo_calls.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn catalog_serves_stable_model_discovery_without_contacting_replicas() {
    let alpha_calls = Arc::new(AtomicUsize::new(0));
    let alpha = spawn_server(Router::new().fallback(any(catalog_upstream)).with_state(
        CatalogUpstream {
            name: "alpha",
            calls: Arc::clone(&alpha_calls),
        },
    ))
    .await;
    let bravo_calls = Arc::new(AtomicUsize::new(0));
    let bravo = spawn_server(Router::new().fallback(any(catalog_upstream)).with_state(
        CatalogUpstream {
            name: "bravo",
            calls: Arc::clone(&bravo_calls),
        },
    ))
    .await;
    // The input order is intentionally reversed. Discovery must not inherit
    // incidental CLI ordering from a deployment manifest.
    let gateway = spawn_catalog_gateway(&[("bravo", bravo.addr), ("alpha", alpha.addr)]).await;

    let list = client()
        .request(
            Request::get(format!("http://{}/v1/models", gateway.addr))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(list.status(), StatusCode::OK);
    let list: serde_json::Value =
        serde_json::from_slice(&to_bytes(Body::new(list.into_body()), 1024).await.unwrap())
            .unwrap();
    assert_eq!(list["object"], "list");
    assert_eq!(list["data"][0]["id"], "alpha");
    assert_eq!(list["data"][1]["id"], "bravo");

    let detail = client()
        .request(
            Request::get(format!("http://{}/v1/models/alpha", gateway.addr))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(detail.status(), StatusCode::OK);
    let detail: serde_json::Value =
        serde_json::from_slice(&to_bytes(Body::new(detail.into_body()), 1024).await.unwrap())
            .unwrap();
    assert_eq!(detail["id"], "alpha");
    assert_eq!(detail["owned_by"], "camelid");

    let missing = client()
        .request(
            Request::get(format!("http://{}/v1/models/not-configured", gateway.addr))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(missing.status(), StatusCode::NOT_FOUND);
    let missing: serde_json::Value = serde_json::from_slice(
        &to_bytes(Body::new(missing.into_body()), 1024)
            .await
            .unwrap(),
    )
    .unwrap();
    assert_eq!(missing["error"]["code"], "model_not_found");
    assert_eq!(alpha_calls.load(Ordering::SeqCst), 0);
    assert_eq!(bravo_calls.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn catalog_verifies_every_configured_id_against_its_pool_before_startup() {
    let upstream = spawn_server(Router::new().route(
        "/v1/models",
        any(|| async {
            Response::builder()
                .status(StatusCode::OK)
                .header("content-type", "application/json")
                .body(Body::from(
                    r#"{"object":"list","data":[{"id":"Llama 3.2 1B Instruct"}]}"#,
                ))
                .unwrap()
        }),
    ))
    .await;
    let exact = ModelCatalog::new([(
        "Llama 3.2 1B Instruct".to_string(),
        UpstreamOrigin::parse(&format!("http://{}", upstream.addr)).unwrap(),
    )])
    .unwrap();
    exact.verify_backend_model_ids().await.unwrap();

    let alias = ModelCatalog::new([(
        "llama-3.2".to_string(),
        UpstreamOrigin::parse(&format!("http://{}", upstream.addr)).unwrap(),
    )])
    .unwrap();
    let error = alias.verify_backend_model_ids().await.unwrap_err();
    assert!(
        error
            .to_string()
            .contains("catalog model id \"llama-3.2\" is not advertised"),
        "unexpected catalog preflight error: {error}"
    );
}

#[tokio::test]
async fn catalog_rejects_unroutable_or_oversized_requests_without_contacting_replicas() {
    let calls = Arc::new(AtomicUsize::new(0));
    let upstream = spawn_server(Router::new().fallback(any(catalog_upstream)).with_state(
        CatalogUpstream {
            name: "alpha",
            calls: Arc::clone(&calls),
        },
    ))
    .await;
    let gateway = spawn_catalog_gateway(&[("alpha", upstream.addr)]).await;

    let unknown = Request::post(format!("http://{}/v1/chat/completions", gateway.addr))
        .header("content-type", "application/json")
        .body(Body::from(r#"{"model":"http://127.0.0.1:9"}"#))
        .unwrap();
    let unknown = client().request(unknown).await.unwrap();
    assert_eq!(unknown.status(), StatusCode::NOT_FOUND);
    let unknown: serde_json::Value = serde_json::from_slice(
        &to_bytes(Body::new(unknown.into_body()), 1024)
            .await
            .unwrap(),
    )
    .unwrap();
    assert_eq!(unknown["error"]["code"], "model_not_found");

    let missing = Request::post(format!("http://{}/v1/chat/completions", gateway.addr))
        .header("content-type", "application/json")
        .body(Body::from(r#"{"messages":[]}"#))
        .unwrap();
    let missing = client().request(missing).await.unwrap();
    assert_eq!(missing.status(), StatusCode::BAD_REQUEST);

    let malformed = Request::post(format!("http://{}/v1/completions", gateway.addr))
        .header("content-type", "application/json")
        .body(Body::from(r#"{"model":"alpha""#))
        .unwrap();
    let malformed = client().request(malformed).await.unwrap();
    assert_eq!(malformed.status(), StatusCode::BAD_REQUEST);

    let wrong_type = Request::post(format!("http://{}/v1/chat/completions", gateway.addr))
        .header("content-type", "text/vendor+json")
        .body(Body::from(r#"{"model":"alpha","messages":[]}"#))
        .unwrap();
    let wrong_type = client().request(wrong_type).await.unwrap();
    assert_eq!(wrong_type.status(), StatusCode::UNSUPPORTED_MEDIA_TYPE);

    let unsupported = Request::post(format!("http://{}/v1/embeddings", gateway.addr))
        .header("content-type", "application/json")
        .body(Body::from(r#"{"model":"alpha","input":"test"}"#))
        .unwrap();
    let unsupported = client().request(unsupported).await.unwrap();
    assert_eq!(unsupported.status(), StatusCode::NOT_IMPLEMENTED);
    assert_eq!(calls.load(Ordering::SeqCst), 0);

    let limited_gateway = spawn_catalog_gateway_with_options(
        &[("alpha", upstream.addr)],
        DEFAULT_MAX_IN_FLIGHT,
        ModelSelectionLimits::new(
            NonZeroUsize::new(8).unwrap(),
            NonZeroUsize::new(16).unwrap(),
        )
        .unwrap(),
        GatewayAuth::Disabled,
    )
    .await;
    let oversized = Request::post(format!(
        "http://{}/v1/chat/completions",
        limited_gateway.addr
    ))
    .header("content-type", "application/json")
    .body(Body::from("123456789"))
    .unwrap();
    let oversized = client().request(oversized).await.unwrap();
    assert_eq!(oversized.status(), StatusCode::PAYLOAD_TOO_LARGE);
    assert_eq!(calls.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn catalog_selector_work_is_bounded_while_a_request_body_stalls() {
    let upstream_calls = Arc::new(AtomicUsize::new(0));
    let upstream = spawn_server(
        Router::new()
            .route(
                "/v1/models",
                any(|| async { catalog_model_list_response("alpha") }),
            )
            .fallback(any({
                let upstream_calls = Arc::clone(&upstream_calls);
                move || {
                    let upstream_calls = Arc::clone(&upstream_calls);
                    async move {
                        upstream_calls.fetch_add(1, Ordering::SeqCst);
                        StatusCode::NO_CONTENT
                    }
                }
            })),
    )
    .await;
    let catalog = ModelCatalog::new([(
        "alpha".to_string(),
        UpstreamOrigin::parse(&format!("http://{}", upstream.addr)).unwrap(),
    )])
    .unwrap();
    let catalog = catalog.verify_backend_model_ids().await.unwrap();
    let app = router_with_model_catalog(
        catalog,
        DEFAULT_MAX_IN_FLIGHT,
        // One global selector slot, and a wait short enough that the test
        // observes the shed-load path rather than the production budget.
        ModelSelectionLimits::new(
            NonZeroUsize::new(1024).unwrap(),
            NonZeroUsize::new(2048).unwrap(),
        )
        .unwrap()
        .with_acquire_timeout(Duration::from_millis(100)),
        None,
        GatewayAuth::Disabled,
        None,
    );

    let selector_started = Arc::new(Notify::new());
    let wait_for_selector = selector_started.notified();
    let stalled_body = Body::from_stream(
        stream::once({
            let selector_started = Arc::clone(&selector_started);
            async move {
                selector_started.notify_one();
                Ok::<Bytes, Infallible>(Bytes::from_static(b"{\"model\":\"alpha\""))
            }
        })
        .chain(stream::pending::<Result<Bytes, Infallible>>()),
    );
    let first = tokio::spawn({
        let app = app.clone();
        async move {
            app.oneshot(
                Request::post("/v1/chat/completions")
                    .header("content-type", "application/json")
                    .body(stalled_body)
                    .unwrap(),
            )
            .await
            .unwrap()
        }
    });
    wait_for_selector.await;

    let rejected = app
        .clone()
        .oneshot(
            Request::post("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"model":"alpha","messages":[]}"#))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(rejected.status(), StatusCode::SERVICE_UNAVAILABLE);
    let retry_after: u64 = rejected.headers()["retry-after"]
        .to_str()
        .unwrap()
        .parse()
        .unwrap();
    assert!((1..=3).contains(&retry_after));
    assert_eq!(upstream_calls.load(Ordering::SeqCst), 0);

    first.abort();
    let _ = first.await;
    let recovered = app
        .oneshot(
            Request::post("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"model":"alpha","messages":[]}"#))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(recovered.status(), StatusCode::NO_CONTENT);
    assert_eq!(upstream_calls.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn catalog_selector_capacity_is_fair_across_authenticated_organizations() {
    let upstream_calls = Arc::new(AtomicUsize::new(0));
    let upstream = spawn_server(
        Router::new()
            .route(
                "/v1/models",
                any(|| async { catalog_model_list_response("alpha") }),
            )
            .fallback(any({
                let upstream_calls = Arc::clone(&upstream_calls);
                move || {
                    let upstream_calls = Arc::clone(&upstream_calls);
                    async move {
                        upstream_calls.fetch_add(1, Ordering::SeqCst);
                        StatusCode::NO_CONTENT
                    }
                }
            })),
    )
    .await;
    let catalog = ModelCatalog::new([(
        "alpha".to_string(),
        UpstreamOrigin::parse(&format!("http://{}", upstream.addr)).unwrap(),
    )])
    .unwrap()
    .verify_backend_model_ids()
    .await
    .unwrap();
    let store = SqliteIdentityStore::open_in_memory().unwrap();
    let tenant_a = store.create_user("tenant-a").unwrap();
    let token_a = store.issue_token(&tenant_a, TokenLifetime::Never).unwrap();
    let tenant_b = store.create_user("tenant-b").unwrap();
    let token_b = store.issue_token(&tenant_b, TokenLifetime::Never).unwrap();
    let app = router_with_model_catalog(
        catalog,
        DEFAULT_MAX_IN_FLIGHT,
        // Two global slots, so the derived per-organization default is one and
        // tenant A's stalled body exhausts A's own share without touching B's.
        ModelSelectionLimits::new(
            NonZeroUsize::new(1024).unwrap(),
            NonZeroUsize::new(4096).unwrap(),
        )
        .unwrap()
        .with_acquire_timeout(Duration::from_millis(100)),
        None,
        GatewayAuth::RequireToken {
            store: Arc::new(store),
            quota: Some(Arc::new(OrgQuota::new(
                NonZeroU32::new(1).unwrap(),
                NonZeroU64::new(60).unwrap(),
            ))),
            usage: None,
        },
        None,
    );

    let selector_started = Arc::new(Notify::new());
    let wait_for_selector = selector_started.notified();
    let stalled_body = Body::from_stream(
        stream::once({
            let selector_started = Arc::clone(&selector_started);
            async move {
                selector_started.notify_one();
                Ok::<Bytes, Infallible>(Bytes::from_static(b"{\"model\":\"alpha\""))
            }
        })
        .chain(stream::pending::<Result<Bytes, Infallible>>()),
    );
    let tenant_a_stalled = tokio::spawn({
        let app = app.clone();
        let token_a = token_a.clone();
        async move {
            app.oneshot(
                Request::post("/v1/chat/completions")
                    .header("authorization", format!("Bearer {token_a}"))
                    .header("content-type", "application/json")
                    .body(stalled_body)
                    .unwrap(),
            )
            .await
            .unwrap()
        }
    });
    wait_for_selector.await;

    let tenant_a_rejected = app
        .clone()
        .oneshot(
            Request::post("/v1/chat/completions")
                .header("authorization", format!("Bearer {token_a}"))
                .header("content-type", "application/json")
                .body(Body::from(r#"{"model":"alpha","messages":[]}"#))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(tenant_a_rejected.status(), StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(
        to_bytes(Body::new(tenant_a_rejected.into_body()), 1024)
            .await
            .unwrap(),
        r#"{"error":{"message":"gateway organization model selection limit reached","type":"gateway_error"}}"#
    );

    let tenant_b_accepted = app
        .clone()
        .oneshot(
            Request::post("/v1/chat/completions")
                .header("authorization", format!("Bearer {token_b}"))
                .header("content-type", "application/json")
                .body(Body::from(r#"{"model":"alpha","messages":[]}"#))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(tenant_b_accepted.status(), StatusCode::NO_CONTENT);
    assert_eq!(upstream_calls.load(Ordering::SeqCst), 1);

    // The incomplete selector was intentionally quota-free, but it can no
    // longer keep tenant A's organization from issuing another request once
    // its selector slot is released.
    tenant_a_stalled.abort();
    let _ = tenant_a_stalled.await;
    let tenant_a_recovered = app
        .oneshot(
            Request::post("/v1/chat/completions")
                .header("authorization", format!("Bearer {token_a}"))
                .header("content-type", "application/json")
                .body(Body::from(r#"{"model":"alpha","messages":[]}"#))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(tenant_a_recovered.status(), StatusCode::NO_CONTENT);
    assert_eq!(upstream_calls.load(Ordering::SeqCst), 2);
}

/// Selector capacity is a memory reservation held for milliseconds, so a
/// request that finds it busy has to wait for it rather than fail on contact.
///
/// This is the shape ordinary traffic has and the shape the in-process
/// single-frame bodies elsewhere in this file cannot produce: real sockets,
/// bodies large enough to span several reads, and more concurrent requests than
/// either bound allows at once. Four organizations of eight requests each
/// exceeds both the per-organization default (four, half the global capacity)
/// and the global capacity (eight) simultaneously.
///
/// With fail-fast admission this loses most of the burst -- measured at 75% of
/// sixteen same-organization 64 KiB requests, and 84% of sixty-four requests
/// spread across sixty-four distinct organizations. Every one of them is valid
/// and would have been served a moment later.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn catalog_selector_capacity_queues_a_concurrent_burst_instead_of_refusing_it() {
    const ORGANIZATIONS: usize = 4;
    const REQUESTS_PER_ORGANIZATION: usize = 8;

    let calls = Arc::new(AtomicUsize::new(0));
    let upstream = spawn_server(Router::new().fallback(any(catalog_upstream)).with_state(
        CatalogUpstream {
            name: "alpha",
            calls: Arc::clone(&calls),
        },
    ))
    .await;
    let store = SqliteIdentityStore::open_in_memory().unwrap();
    let mut tokens = Vec::new();
    for index in 0..ORGANIZATIONS {
        let principal = store.create_user(&format!("tenant-{index}")).unwrap();
        tokens.push(store.issue_token(&principal, TokenLifetime::Never).unwrap());
    }
    let gateway = spawn_catalog_gateway_with_options(
        &[("alpha", upstream.addr)],
        DEFAULT_MAX_IN_FLIGHT,
        default_selection_limits(),
        GatewayAuth::RequireToken {
            store: Arc::new(store),
            quota: None,
            usage: None,
        },
    )
    .await;

    // Far larger than a socket buffer, so materializing it spans many polls
    // and concurrent selectors genuinely overlap. A body small enough to
    // arrive in one read completes without ever yielding, which is why the
    // single-frame bodies elsewhere in this file cannot show this.
    let prompt = "x".repeat(512 * 1024);
    let mut requests = Vec::new();
    for token in &tokens {
        for _ in 0..REQUESTS_PER_ORGANIZATION {
            let token = token.clone();
            let addr = gateway.addr;
            let prompt = prompt.clone();
            requests.push(tokio::spawn(async move {
                let body = serde_json::json!({
                    "model": "alpha",
                    "messages": [{ "role": "user", "content": prompt }],
                })
                .to_string();
                let sent = client()
                    .request(
                        Request::post(format!("http://{addr}/v1/chat/completions"))
                            .header("authorization", format!("Bearer {token}"))
                            .header("content-type", "application/json")
                            .body(Body::from(body))
                            .unwrap(),
                    )
                    .await;
                match sent {
                    Ok(response) => response.status().to_string(),
                    // Answering before the body finished uploading aborts the
                    // connection, so a refusal can surface here rather than as
                    // a status. Both are the burst failing.
                    Err(error) => format!("transport failure: {error}"),
                }
            }));
        }
    }

    let mut outcomes = Vec::new();
    for request in requests {
        outcomes.push(request.await.unwrap());
    }
    let served = StatusCode::OK.to_string();
    let refused: Vec<&String> = outcomes
        .iter()
        .filter(|outcome| **outcome != served)
        .collect();
    assert!(
        refused.is_empty(),
        "every request in the burst is valid and routable, but {} of {} were not served: {refused:?}",
        refused.len(),
        outcomes.len(),
    );
    assert_eq!(
        calls.load(Ordering::SeqCst),
        ORGANIZATIONS * REQUESTS_PER_ORGANIZATION
    );
}

/// Waiting for selector capacity is only safe because holding it is bounded.
/// Without a read deadline the only limit on how long one slot stays occupied
/// is the connection cap, so a handful of dribbling clients could exhaust the
/// budget for minutes and every waiter behind them would time out.
#[tokio::test]
async fn catalog_reclaims_selector_capacity_from_a_body_that_misses_the_read_deadline() {
    let upstream_calls = Arc::new(AtomicUsize::new(0));
    let upstream = spawn_server(
        Router::new()
            .route(
                "/v1/models",
                any(|| async { catalog_model_list_response("alpha") }),
            )
            .fallback(any({
                let upstream_calls = Arc::clone(&upstream_calls);
                move || {
                    let upstream_calls = Arc::clone(&upstream_calls);
                    async move {
                        upstream_calls.fetch_add(1, Ordering::SeqCst);
                        StatusCode::NO_CONTENT
                    }
                }
            })),
    )
    .await;
    let catalog = ModelCatalog::new([(
        "alpha".to_string(),
        UpstreamOrigin::parse(&format!("http://{}", upstream.addr)).unwrap(),
    )])
    .unwrap()
    .verify_backend_model_ids()
    .await
    .unwrap();
    let app = router_with_model_catalog(
        catalog,
        DEFAULT_MAX_IN_FLIGHT,
        // One global slot, and a read deadline the stalled body below cannot
        // meet. The wait for capacity is longer than that deadline, so a
        // queued request outlives the slot it is waiting for.
        ModelSelectionLimits::new(
            NonZeroUsize::new(1024).unwrap(),
            NonZeroUsize::new(2048).unwrap(),
        )
        .unwrap()
        .with_read_timeout(Duration::from_millis(100))
        .with_acquire_timeout(Duration::from_secs(5)),
        None,
        GatewayAuth::Disabled,
        None,
    );

    let selector_started = Arc::new(Notify::new());
    let wait_for_selector = selector_started.notified();
    let stalled_body = Body::from_stream(
        stream::once({
            let selector_started = Arc::clone(&selector_started);
            async move {
                selector_started.notify_one();
                Ok::<Bytes, Infallible>(Bytes::from_static(b"{\"model\":\"alpha\""))
            }
        })
        .chain(stream::pending::<Result<Bytes, Infallible>>()),
    );
    let stalled = tokio::spawn({
        let app = app.clone();
        async move {
            app.oneshot(
                Request::post("/v1/chat/completions")
                    .header("content-type", "application/json")
                    .body(stalled_body)
                    .unwrap(),
            )
            .await
            .unwrap()
        }
    });
    wait_for_selector.await;

    // Queued behind the stalled body rather than refused: it is served once
    // the deadline reclaims the slot, without the client retrying.
    let queued = app
        .clone()
        .oneshot(
            Request::post("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"model":"alpha","messages":[]}"#))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(queued.status(), StatusCode::NO_CONTENT);
    assert_eq!(upstream_calls.load(Ordering::SeqCst), 1);

    let stalled = stalled.await.unwrap();
    assert_eq!(stalled.status(), StatusCode::REQUEST_TIMEOUT);
    let stalled: serde_json::Value = serde_json::from_slice(
        &to_bytes(Body::new(stalled.into_body()), 1024)
            .await
            .unwrap(),
    )
    .unwrap();
    assert_eq!(stalled["error"]["code"], "request_body_timeout");
    // The stalled request never named a routable model, so it never reached a
    // replica: the second call is still the only one.
    assert_eq!(upstream_calls.load(Ordering::SeqCst), 1);
}

/// A request whose declared length already exceeds the selector limit is
/// refused from its head alone, so it never occupies a slot or the bandwidth
/// behind one. Proven under exhausted capacity: if the size check ran after
/// admission it would queue and then time out as a `503` instead.
#[tokio::test]
async fn catalog_refuses_an_oversized_declared_body_before_it_spends_selector_capacity() {
    let upstream = spawn_server(
        Router::new().fallback(any(|| async { catalog_model_list_response("alpha") })),
    )
    .await;
    let catalog = ModelCatalog::new([(
        "alpha".to_string(),
        UpstreamOrigin::parse(&format!("http://{}", upstream.addr)).unwrap(),
    )])
    .unwrap()
    .verify_backend_model_ids()
    .await
    .unwrap();
    let app = router_with_model_catalog(
        catalog,
        DEFAULT_MAX_IN_FLIGHT,
        ModelSelectionLimits::new(
            NonZeroUsize::new(1024).unwrap(),
            NonZeroUsize::new(2048).unwrap(),
        )
        .unwrap()
        .with_acquire_timeout(Duration::from_millis(100)),
        None,
        GatewayAuth::Disabled,
        None,
    );

    let selector_started = Arc::new(Notify::new());
    let wait_for_selector = selector_started.notified();
    let stalled_body = Body::from_stream(
        stream::once({
            let selector_started = Arc::clone(&selector_started);
            async move {
                selector_started.notify_one();
                Ok::<Bytes, Infallible>(Bytes::from_static(b"{\"model\":\"alpha\""))
            }
        })
        .chain(stream::pending::<Result<Bytes, Infallible>>()),
    );
    let stalled = tokio::spawn({
        let app = app.clone();
        async move {
            app.oneshot(
                Request::post("/v1/chat/completions")
                    .header("content-type", "application/json")
                    .body(stalled_body)
                    .unwrap(),
            )
            .await
            .unwrap()
        }
    });
    wait_for_selector.await;

    // The control: capacity really is exhausted, so a request that has to read
    // its body is shed.
    let within_limit = app
        .clone()
        .oneshot(
            Request::post("/v1/chat/completions")
                .header("content-type", "application/json")
                .header("content-length", "31")
                .body(Body::from(r#"{"model":"alpha","messages":[]}"#))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(within_limit.status(), StatusCode::SERVICE_UNAVAILABLE);

    let oversized = app
        .clone()
        .oneshot(
            Request::post("/v1/chat/completions")
                .header("content-type", "application/json")
                .header("content-length", "4096")
                .body(Body::from("x".repeat(4096)))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(oversized.status(), StatusCode::PAYLOAD_TOO_LARGE);
    let oversized: serde_json::Value = serde_json::from_slice(
        &to_bytes(Body::new(oversized.into_body()), 1024)
            .await
            .unwrap(),
    )
    .unwrap();
    assert_eq!(oversized["error"]["code"], "request_too_large");

    stalled.abort();
    let _ = stalled.await;
}

/// serde fills a derived struct from a JSON sequence positionally, so a derived
/// selector reads `["alpha"]` as `{"model":"alpha"}` and routes it. The replica
/// deserializing the same bytes maps position zero onto the first field of its
/// own request type, so the two disagree about what the request said while the
/// gateway records a `model_id` the body never named. Only an object carries a
/// top-level `model` member, and only an object is accepted.
#[tokio::test]
async fn catalog_refuses_a_body_that_is_not_a_json_object() {
    let calls = Arc::new(AtomicUsize::new(0));
    let upstream = spawn_server(Router::new().fallback(any(catalog_upstream)).with_state(
        CatalogUpstream {
            name: "alpha",
            calls: Arc::clone(&calls),
        },
    ))
    .await;
    let gateway = spawn_catalog_gateway(&[("alpha", upstream.addr)]).await;

    for body in [r#"["alpha"]"#, r#""alpha""#, "null", "[]"] {
        let response = client()
            .request(
                Request::post(format!("http://{}/v1/chat/completions", gateway.addr))
                    .header("content-type", "application/json")
                    .body(Body::from(body))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(
            response.status(),
            StatusCode::BAD_REQUEST,
            "{body} is not a JSON object and must not select a pool"
        );
        let response: serde_json::Value = serde_json::from_slice(
            &to_bytes(Body::new(response.into_body()), 1024)
                .await
                .unwrap(),
        )
        .unwrap();
        assert_eq!(response["error"]["code"], "malformed_json");
    }
    assert_eq!(calls.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn catalog_never_retries_a_generation_after_an_upstream_connection_failure() {
    let attempts = Arc::new(AtomicUsize::new(0));
    let upstream = spawn_catalog_connection_dropper(Arc::clone(&attempts)).await;
    let gateway = spawn_catalog_gateway(&[("alpha", upstream.addr)]).await;
    let request = Request::post(format!("http://{}/v1/chat/completions", gateway.addr))
        .header("content-type", "application/json")
        .body(Body::from(r#"{"model":"alpha","messages":[]}"#))
        .unwrap();

    let response = client().request(request).await.unwrap();
    assert_eq!(response.status(), StatusCode::BAD_GATEWAY);
    assert_eq!(
        attempts.load(Ordering::SeqCst),
        1,
        "a failed generation request must not be replayed to its model pool"
    );
}

#[tokio::test]
async fn catalog_authenticates_before_discovery_and_invalid_models_do_not_spend_quota() {
    let calls = Arc::new(AtomicUsize::new(0));
    let upstream = spawn_server(Router::new().fallback(any(catalog_upstream)).with_state(
        CatalogUpstream {
            name: "alpha",
            calls: Arc::clone(&calls),
        },
    ))
    .await;
    let store = SqliteIdentityStore::open_in_memory().unwrap();
    let principal = store.create_user("ada").unwrap();
    let token = store.issue_token(&principal, TokenLifetime::Never).unwrap();
    let gateway = spawn_catalog_gateway_with_options(
        &[("alpha", upstream.addr)],
        DEFAULT_MAX_IN_FLIGHT,
        default_selection_limits(),
        GatewayAuth::RequireToken {
            store: Arc::new(store),
            quota: Some(Arc::new(OrgQuota::new(
                NonZeroU32::new(1).unwrap(),
                NonZeroU64::new(60).unwrap(),
            ))),
            usage: None,
        },
    )
    .await;

    let anonymous = Request::get(format!("http://{}/v1/models", gateway.addr))
        .body(Body::empty())
        .unwrap();
    assert_eq!(
        client().request(anonymous).await.unwrap().status(),
        StatusCode::UNAUTHORIZED
    );

    let unknown = Request::post(format!("http://{}/v1/chat/completions", gateway.addr))
        .header("authorization", format!("Bearer {token}"))
        .header("content-type", "application/json")
        .body(Body::from(r#"{"model":"not-configured"}"#))
        .unwrap();
    assert_eq!(
        client().request(unknown).await.unwrap().status(),
        StatusCode::NOT_FOUND
    );

    let request = || {
        Request::post(format!("http://{}/v1/chat/completions", gateway.addr))
            .header("authorization", format!("Bearer {token}"))
            .header("content-type", "application/json")
            .body(Body::from(r#"{"model":"alpha","messages":[]}"#))
            .unwrap()
    };
    assert_eq!(
        client().request(request()).await.unwrap().status(),
        StatusCode::OK
    );
    assert_eq!(
        client().request(request()).await.unwrap().status(),
        StatusCode::TOO_MANY_REQUESTS
    );
    assert_eq!(calls.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn catalog_records_the_selected_model_in_audit_and_usage_logs() {
    let upstream = spawn_server(
        Router::new()
            .route(
                "/v1/models",
                any(|| async { catalog_model_list_response("alpha") }),
            )
            .fallback(any(|| async { StatusCode::NO_CONTENT })),
    )
    .await;
    let store = SqliteIdentityStore::open_in_memory().unwrap();
    let principal = store.create_user("ada").unwrap();
    let token = store.issue_token(&principal, TokenLifetime::Never).unwrap();
    let dir = tempfile::tempdir().unwrap();
    let audit_path = dir.path().join("audit.jsonl");
    let usage_path = dir.path().join("usage.jsonl");
    let gateway = spawn_catalog_gateway_with_audit(
        &[("alpha", upstream.addr)],
        DEFAULT_MAX_IN_FLIGHT,
        default_selection_limits(),
        GatewayAuth::RequireToken {
            store: Arc::new(store),
            quota: None,
            usage: Some(gateway_log(&usage_path)),
        },
        Some(gateway_log(&audit_path)),
    )
    .await;

    let request = Request::post(format!("http://{}/v1/chat/completions", gateway.addr))
        .header("authorization", format!("Bearer {token}"))
        .header("content-type", "application/json")
        .body(Body::from(r#"{"model":"alpha","messages":[]}"#))
        .unwrap();
    assert_eq!(
        client().request(request).await.unwrap().status(),
        StatusCode::NO_CONTENT
    );

    let audit = read_audit_records(&audit_path, 1).await;
    let usage = read_usage_records(&usage_path, 1).await;
    assert_eq!(audit[0]["model_id"], "alpha");
    assert_eq!(usage[0]["model_id"], "alpha");
    assert_eq!(audit[0]["request_id"], usage[0]["request_id"]);
}

#[tokio::test]
async fn catalog_streams_replica_responses_after_model_selection() {
    let release_second = Arc::new(Notify::new());
    let upstream = spawn_server(
        Router::new()
            .route(
                "/v1/models",
                any(|| async { catalog_model_list_response("alpha") }),
            )
            .route("/v1/chat/completions", any(delayed_sse))
            .with_state(Arc::clone(&release_second)),
    )
    .await;
    let gateway = spawn_catalog_gateway(&[("alpha", upstream.addr)]).await;
    let request = Request::post(format!("http://{}/v1/chat/completions", gateway.addr))
        .header("content-type", "application/json")
        .body(Body::from(
            r#"{"model":"alpha","messages":[],"stream":true}"#,
        ))
        .unwrap();
    let response = client().request(request).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let mut body = response.into_body();
    let first = tokio::time::timeout(Duration::from_secs(1), body.frame())
        .await
        .expect("the first streamed event must not wait for completion")
        .unwrap()
        .unwrap()
        .into_data()
        .unwrap();
    assert_eq!(first, "data: first\n\n");
    release_second.notify_one();
    assert_eq!(
        body.frame().await.unwrap().unwrap().into_data().unwrap(),
        "data: second\n\n"
    );
}

/// Every `(method, path)` in `replica_contract::PUBLIC_ROUTES` must be forwarded
/// to the upstream (proved via `probe_path` for the parameterized route). This
/// test drives only contract requests, so what it establishes is that the
/// gateway forwards *all* of the contract, and — since the expected pairs are
/// distinct — exactly one upstream request per contract entry, with no
/// duplicates. It does not, on its own, prove that non-contract routes are
/// rejected: that comes from deriving the router from `PUBLIC_ROUTES` plus
/// `rejects_non_inference_routes_without_contacting_upstream`. Together they pin
/// the gateway allowlist to the contract — adding or removing a contract route
/// changes this test's outcome, with no separately maintained list to keep in
/// sync.
#[tokio::test]
async fn forwards_exactly_the_public_contract_routes() {
    let seen = Arc::new(Mutex::new(Vec::new()));
    let upstream = spawn_server(
        Router::new()
            .fallback(any(record_method_and_path))
            .with_state(Arc::clone(&seen)),
    )
    .await;
    let gateway = spawn_gateway(upstream.addr).await;

    let mut expected: Vec<(Method, String)> = Vec::new();
    for spec in PUBLIC_ROUTES {
        for contract_method in spec.methods {
            let method = match contract_method {
                HttpMethod::Get => Method::GET,
                HttpMethod::Post => Method::POST,
                HttpMethod::Delete => Method::DELETE,
            };
            let response = client()
                .request(
                    Request::builder()
                        .method(method.clone())
                        .uri(format!("http://{}{}", gateway.addr, spec.probe_path))
                        .body(Body::empty())
                        .unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(
                response.status(),
                StatusCode::NO_CONTENT,
                "gateway did not forward contractual route {} {}",
                method,
                spec.probe_path
            );
            expected.push((method, spec.probe_path.to_string()));
        }
    }

    let seen = seen.lock().unwrap();
    assert_eq!(
        seen.len(),
        expected.len(),
        "gateway forwarded a different number of requests than the contract declares"
    );
    for pair in &expected {
        assert!(
            seen.contains(pair),
            "upstream never received forwarded {} {}",
            pair.0,
            pair.1
        );
    }
}
