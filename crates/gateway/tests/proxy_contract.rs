use axum::body::{to_bytes, Body};
use axum::extract::{Request, State};
use axum::http::{Method, StatusCode};
use axum::response::Response;
use axum::routing::{any, post};
use axum::Router;
use bytes::Bytes;
use camelid_enterprise_gateway::{
    router as gateway_router, router_with_max_in_flight, router_with_options, GatewayAuth,
    UpstreamOrigin, DEFAULT_MAX_IN_FLIGHT,
};
use futures_util::stream;
use http_body_util::BodyExt;
use hyper_util::client::legacy::connect::HttpConnector;
use hyper_util::client::legacy::Client;
use hyper_util::rt::TokioExecutor;
use identity::SqliteIdentityStore;
use std::convert::Infallible;
use std::net::SocketAddr;
use std::num::NonZeroUsize;
use std::path::PathBuf;
use std::pin::Pin;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll};
use std::time::Duration;
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
    audit: Arc<PathBuf>,
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
    audit: Option<Arc<PathBuf>>,
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

    let gateway =
        spawn_gateway_with_auth(unreachable_upstream, GatewayAuth::RequireToken(store)).await;
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

    let gateway =
        spawn_gateway_with_auth(unreachable_upstream, GatewayAuth::RequireToken(store)).await;
    let request = Request::get(format!("http://{}/v1/models", gateway.addr))
        .header("authorization", "Bearer not-a-real-token")
        .body(Body::empty())
        .unwrap();

    let response = client().request(request).await.unwrap();

    assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
}

#[tokio::test]
async fn forwards_requests_carrying_a_valid_bearer_token() {
    let store = SqliteIdentityStore::open_in_memory().unwrap();
    let principal = store.create_user("ada").unwrap();
    let token = store.issue_token(&principal).unwrap();

    let captured = Arc::new(Mutex::new(None));
    let upstream = spawn_server(
        Router::new()
            .fallback(any(capture_request))
            .with_state(Arc::clone(&captured)),
    )
    .await;
    let gateway =
        spawn_gateway_with_auth(upstream.addr, GatewayAuth::RequireToken(Arc::new(store))).await;
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

/// The gateway forwards exactly the published contract, read from the registry
/// rather than written out again.
///
/// This is the check that was missing while the gateway kept its own axum route
/// table. Adding an eleventh route to `replica_contract::PUBLIC_ROUTES` used to
/// fail four tests in the replica crate and none here, because no gateway target
/// linked the registry at all — so a route the contract promised and the replica
/// served would have been `404`ed at the component the deployment documents as
/// the trust boundary. The drift direction was availability rather than breach,
/// which is why it could have shipped unnoticed.
///
/// Both halves matter. Every declared method/path reaching the upstream is what
/// catches a registry route the gateway cannot express — an axum pattern it does
/// not accept, or a method the mapping drops. The recorded set being *exactly*
/// the declared set is what catches the other direction: a route forwarded here
/// that the contract does not name.
#[tokio::test]
async fn the_gateway_forwards_exactly_the_published_contract() {
    let seen = Arc::new(Mutex::new(Vec::new()));
    let upstream = spawn_server(Router::new().fallback(any({
        let seen = Arc::clone(&seen);
        move |request: Request| {
            let seen = Arc::clone(&seen);
            async move {
                seen.lock().unwrap().push((
                    request.method().to_string(),
                    request.uri().path().to_string(),
                ));
                StatusCode::NO_CONTENT
            }
        }
    })))
    .await;
    let app = gateway_router(UpstreamOrigin::parse(&format!("http://{}", upstream.addr)).unwrap());

    let mut expected = Vec::new();
    for route in replica_contract::PUBLIC_ROUTES {
        for method in route.methods {
            let response = app
                .clone()
                .oneshot(
                    Request::builder()
                        .method(method.as_str())
                        .uri(route.probe_path)
                        .body(Body::empty())
                        .unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(
                response.status(),
                StatusCode::NO_CONTENT,
                "{} {} is in the published contract and the gateway did not forward it",
                method.as_str(),
                route.path
            );
            expected.push((method.as_str().to_string(), route.probe_path.to_string()));
        }
    }

    // Reached the replica, and nothing else did: `/healthz` is answered locally
    // and never appears here.
    let response = app
        .clone()
        .oneshot(Request::get("/healthz").body(Body::empty()).unwrap())
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::NO_CONTENT);

    assert_eq!(*seen.lock().unwrap(), expected);
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
    let token = store.issue_token(&principal).unwrap();

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
        GatewayAuth::RequireToken(Arc::new(store)),
        Arc::new(audit_path.clone()),
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
    assert_eq!(forwarded_record["request_id"], upstream_request_id);
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
        Arc::new(audit_path.clone()),
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

async fn read_audit_records(path: &std::path::Path, expected: usize) -> Vec<serde_json::Value> {
    for _ in 0..200 {
        if let Ok(contents) = std::fs::read_to_string(path) {
            let lines: Vec<&str> = contents.lines().filter(|line| !line.trim().is_empty()).collect();
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
