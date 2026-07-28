use axum::body::{to_bytes, Body};
use axum::http::{header::CONTENT_TYPE, Request, StatusCode};
use camelid_enterprise::{apply_deterministic, attributed_router};
use serde_json::{json, Value};
use std::path::PathBuf;
use std::time::Duration;
use tower::ServiceExt;

const MODEL_ENV: &str = "CAMELID_ENTERPRISE_TEST_MODEL";
// The deterministic lane serializes generation. This test configures the
// engine's bounded queue to one slot before building AppState, so its
// saturation probe can prove typed backpressure with at most two accepted
// generations rather than queuing a long serialized batch on shared CI.
const STEP_TIMEOUT: Duration = Duration::from_secs(600);
const ENGINE_QUEUE_DEPTH_ENV: &str = "CAMELID_QUEUE_DEPTH";

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
    // EngineHandle reads this setting while AppState is built. This ignored
    // model test is run as the sole test in its binary (`--exact
    // --test-threads=1` in CI), so temporarily setting the process environment
    // cannot affect another test. Restore any caller-provided value immediately
    // after the worker has captured it.
    let prior_queue_depth = std::env::var_os(ENGINE_QUEUE_DEPTH_ENV);
    std::env::set_var(ENGINE_QUEUE_DEPTH_ENV, "1");
    let state = camelid::api::AppState::with_configured_threads(Some(4))
        .with_default_enable_thinking(false)
        .with_models_dir(None);
    match prior_queue_depth {
        Some(value) => std::env::set_var(ENGINE_QUEUE_DEPTH_ENV, value),
        None => std::env::remove_var(ENGINE_QUEUE_DEPTH_ENV),
    }
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

    // The worker is running plus one configured queue slot. Four requests with
    // a deliberately nontrivial prompt therefore prove both acceptance and
    // typed queue-full rejection while at most two requests can reach decode.
    // This is a behavioral test of bounded backpressure, not a throughput
    // benchmark; keeping the accepted set this small prevents CI's shared CPU
    // from turning a queue test into a 15-minute serialized decode workload.
    const CONCURRENT_REQUESTS: usize = 4;
    let backpressure_prompt = "backpressure ".repeat(512);
    let mut requests = tokio::task::JoinSet::new();
    for index in 0..CONCURRENT_REQUESTS {
        let app = app.clone();
        let model_id = model_id.to_string();
        let backpressure_prompt = backpressure_prompt.clone();
        requests.spawn(async move {
            send(
                app,
                post_json(
                    "/v1/chat/completions",
                    json!({
                        "model": model_id,
                        "messages": [{
                            "role": "user",
                            "content": format!("{backpressure_prompt} {index}")
                        }],
                        "temperature": 0,
                        "max_tokens": 1,
                        "stream": false
                    }),
                ),
            )
            .await
        });
    }

    let mut accepted = 0;
    let mut rejected = 0;
    while let Some(result) = requests.join_next().await {
        let response = result.unwrap();
        assert_attribution(&response, &expected_sha);
        match response.status() {
            StatusCode::OK => {
                accepted += 1;
                body_json(response).await;
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

    let recovered = send(app, Request::get("/v1/health").body(Body::empty()).unwrap()).await;
    assert_eq!(recovered.status(), StatusCode::OK);
    assert_eq!(body_json(recovered).await["engine_queue_depth"], 0);
}
