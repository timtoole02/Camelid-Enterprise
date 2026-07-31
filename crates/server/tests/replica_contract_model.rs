use axum::body::{to_bytes, Body};
use axum::http::{header::CONTENT_TYPE, Request, StatusCode};
use camelid_enterprise::{apply_deterministic, attributed_router};
use http_body_util::BodyExt;
use serde_json::{json, Value};
use std::path::PathBuf;
use std::time::Duration;
use tower::ServiceExt;

const MODEL_ENV: &str = "CAMELID_ENTERPRISE_TEST_MODEL";
// The deterministic lane serializes generation. The saturation probe below
// uses streaming requests, which post their engine job and return before the
// decode runs, so this bound is a hang detector rather than a budget for
// completed generations; the CI job's own timeout-minutes remains the real
// ceiling on total wall-clock time.
const STEP_TIMEOUT: Duration = Duration::from_secs(600);

async fn body_json(response: axum::response::Response) -> Value {
    let bytes = tokio::time::timeout(
        STEP_TIMEOUT,
        to_bytes(response.into_body(), 16 * 1024 * 1024),
    )
    .await
    .expect("replica response body exceeded 900 seconds")
    .unwrap();
    serde_json::from_slice(&bytes).unwrap()
}

fn post_json(path: &str, body: Value) -> Request<Body> {
    Request::post(path)
        .header(CONTENT_TYPE, "application/json")
        .body(Body::from(body.to_string()))
        .unwrap()
}

fn assert_attribution(response: &axum::response::Response, expected_sha: &str) {
    assert_eq!(response.headers()["x-camelid-lane"], "deterministic");
    assert_eq!(
        response.headers()["x-camelid-config-sha256"],
        &expected_sha[..12]
    );
    assert!(!response.headers()["x-camelid-host"].is_empty());
}

async fn send(app: axum::Router, request: Request<Body>) -> axum::response::Response {
    tokio::time::timeout(STEP_TIMEOUT, app.oneshot(request))
        .await
        .expect("replica contract HTTP step exceeded 900 seconds")
        .unwrap()
}

#[tokio::test]
#[ignore = "requires CAMELID_ENTERPRISE_TEST_MODEL to name a compatible local GGUF"]
async fn real_model_conforms_to_replica_http_v1() {
    let model = PathBuf::from(
        std::env::var(MODEL_ENV)
            .unwrap_or_else(|_| panic!("{MODEL_ENV} must name a compatible local GGUF")),
    )
    .canonicalize()
    .unwrap_or_else(|error| panic!("could not resolve {MODEL_ENV}: {error}"));
    let config = apply_deterministic().expect("the test process must apply the lane contract");
    let expected_sha = config.sha256.clone();
    // Nothing here may touch the engine's environment after this point. The
    // lane's guarantee is stated against a frozen configuration vector, and
    // `apply_deterministic` refuses to start when a key it excludes -- such as
    // CAMELID_QUEUE_DEPTH -- is set. Setting one after that check has passed
    // would run the engine in a configuration the lane forbids while every
    // response below still carried `expected_sha`, which asserts the opposite.
    // The queue-saturation probe therefore works against the engine's real
    // default queue depth.
    let state = camelid::api::AppState::with_configured_threads(Some(4))
        .with_default_enable_thinking(false)
        .with_models_dir(None);
    let app = attributed_router(state, config.sha256, "contract-test/host".to_string(), None);

    let load = send(
        app.clone(),
        post_json(
            "/api/models/load",
            json!({ "path": model.to_string_lossy() }),
        ),
    )
    .await;
    assert_eq!(load.status(), StatusCode::OK);
    assert_attribution(&load, &expected_sha);
    let loaded = body_json(load).await;
    let model_id = loaded["id"]
        .as_str()
        .expect("load response must name the model");

    let health = send(
        app.clone(),
        Request::get("/v1/health").body(Body::empty()).unwrap(),
    )
    .await;
    assert_eq!(health.status(), StatusCode::OK);
    assert_attribution(&health, &expected_sha);
    let health = body_json(health).await;
    assert_eq!(health["loaded_now"], true);
    assert_eq!(health["generation_ready"], true);
    assert_eq!(health["active_model_id"], model_id);

    let models = send(
        app.clone(),
        Request::get("/v1/models").body(Body::empty()).unwrap(),
    )
    .await;
    assert_eq!(models.status(), StatusCode::OK);
    let models = body_json(models).await;
    assert_eq!(models["object"], "list");
    assert_eq!(models["data"].as_array().unwrap().len(), 1);
    assert_eq!(models["data"][0]["id"], model_id);

    let request = json!({
        "model": model_id,
        "messages": [{ "role": "user", "content": "Reply briefly." }],
        "temperature": 0,
        "max_tokens": 8,
        "stream": false
    });
    let first = send(
        app.clone(),
        post_json("/v1/chat/completions", request.clone()),
    )
    .await;
    assert_eq!(first.status(), StatusCode::OK);
    assert_attribution(&first, &expected_sha);
    let first = body_json(first).await;
    assert_eq!(first["camelid_lane"], "deterministic");
    assert_eq!(first["camelid_config_sha256"], &expected_sha[..12]);

    let second = send(app.clone(), post_json("/v1/chat/completions", request)).await;
    assert_eq!(second.status(), StatusCode::OK);
    let second = body_json(second).await;
    assert_eq!(first["choices"], second["choices"]);

    let raw_request = json!({
        "model": model_id,
        "prompt": "Complete briefly:",
        "temperature": 0,
        "max_tokens": 4,
        "stream": false
    });
    let raw = send(app.clone(), post_json("/v1/completions", raw_request)).await;
    assert_eq!(raw.status(), StatusCode::OK);
    assert_attribution(&raw, &expected_sha);
    let raw = body_json(raw).await;
    assert_eq!(raw["object"], "text_completion");
    assert_eq!(raw["camelid_lane"], "deterministic");

    let stream = send(
        app.clone(),
        post_json(
            "/v1/chat/completions",
            json!({
                "model": model_id,
                "messages": [{ "role": "user", "content": "Reply briefly." }],
                "temperature": 0,
                "max_tokens": 4,
                "stream": true
            }),
        ),
    )
    .await;
    assert_eq!(stream.status(), StatusCode::OK);
    assert_attribution(&stream, &expected_sha);
    assert!(stream.headers()[CONTENT_TYPE]
        .to_str()
        .unwrap()
        .starts_with("text/event-stream"));
    let bytes = tokio::time::timeout(STEP_TIMEOUT, to_bytes(stream.into_body(), 16 * 1024 * 1024))
        .await
        .expect("replica SSE body exceeded 900 seconds")
        .unwrap();
    let stream = String::from_utf8(bytes.to_vec()).unwrap();
    let events: Vec<&str> = stream
        .lines()
        .filter_map(|line| line.strip_prefix("data: "))
        .collect();
    assert!(events.len() >= 2, "expected data plus [DONE]: {stream}");
    assert_eq!(events.last(), Some(&"[DONE]"));
    for event in &events[..events.len() - 1] {
        serde_json::from_str::<Value>(event).expect("every data event before [DONE] is JSON");
    }

    // Above the engine's default admission capacity (a bounded queue of eight
    // plus the one running job), so both the accepted and rejected branches
    // below are exercised. Streaming handlers post their decode job before
    // returning an SSE response and emit the role frame before awaiting any
    // decode event, so this proves acceptance and typed queue-full rejection
    // without waiting for completed model generations. Polling each accepted
    // body's role frame activates its cancellation guard; dropping the bodies
    // then proves queue-depth recovery without turning this admission test
    // into a hardware-duration benchmark.
    const CONCURRENT_REQUESTS: usize = 12;
    let mut requests = tokio::task::JoinSet::new();
    for index in 0..CONCURRENT_REQUESTS {
        let app = app.clone();
        let model_id = model_id.to_string();
        requests.spawn(async move {
            send(
                app,
                post_json(
                    "/v1/chat/completions",
                    json!({
                        "model": model_id,
                        "messages": [{
                            "role": "user",
                            "content": format!("queue admission probe {index}")
                        }],
                        "temperature": 0,
                        // Long enough to keep the first posted decode active
                        // while the concurrent handlers submit their jobs, but
                        // this test never waits for generation completion.
                        "max_tokens": 128,
                        "stream": true
                    }),
                ),
            )
            .await
        });
    }

    let mut accepted = 0;
    let mut rejected = 0;
    let mut accepted_bodies = Vec::new();
    while let Some(result) = requests.join_next().await {
        let response = result.unwrap();
        assert_attribution(&response, &expected_sha);
        match response.status() {
            StatusCode::OK => {
                accepted += 1;
                assert!(response.headers()[CONTENT_TYPE]
                    .to_str()
                    .unwrap()
                    .starts_with("text/event-stream"));
                let mut body = response.into_body();
                let role = tokio::time::timeout(STEP_TIMEOUT, body.frame())
                    .await
                    .expect("SSE role frame exceeded 600 seconds")
                    .expect("SSE stream ended before its role frame")
                    .expect("SSE role frame failed")
                    .into_data()
                    .expect("SSE role frame must be data");
                assert!(
                    String::from_utf8_lossy(&role).contains("assistant"),
                    "unexpected initial SSE role frame: {:?}",
                    role
                );
                accepted_bodies.push(body);
            }
            StatusCode::SERVICE_UNAVAILABLE => {
                rejected += 1;
                assert_eq!(response.headers()["retry-after"], "1");
                let body = body_json(response).await;
                assert_eq!(body["error"]["type"], "runtime_unavailable");
                assert_eq!(body["error"]["code"], "engine_queue_full");
                assert!(body["error"]["param"].is_null());
                assert_eq!(body["camelid_lane"], "deterministic");
                assert_eq!(body["camelid_config_sha256"], &expected_sha[..12]);
            }
            status => panic!("unexpected queue-stress status: {status}"),
        }
    }
    assert!(accepted > 0);
    assert!(rejected > 0);
    assert_eq!(accepted + rejected, CONCURRENT_REQUESTS);

    // Drops fire the per-stream cancellation guards. The engine observes the
    // token before its next decode step, then drains cancelled queued jobs.
    drop(accepted_bodies);

    let deadline = std::time::Instant::now() + STEP_TIMEOUT;
    loop {
        let recovered = send(
            app.clone(),
            Request::get("/v1/health").body(Body::empty()).unwrap(),
        )
        .await;
        assert_eq!(recovered.status(), StatusCode::OK);
        if body_json(recovered).await["engine_queue_depth"] == 0 {
            break;
        }
        assert!(
            std::time::Instant::now() < deadline,
            "engine queue depth did not recover after cancelling accepted streams"
        );
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
}
