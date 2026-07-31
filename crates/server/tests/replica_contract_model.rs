//! Model-backed conformance: the surface a client meets, against real weights.
//!
//! This is the only executable check in the workspace that runs the replica with
//! a GGUF actually loaded, so what it drives matters. It drives
//! [`replica_router`] — the same composition `serve` uses — and not the bare
//! attributed router, because the bare router carries neither the served-route
//! filter nor the generation-body filter: a conformance pass over it would have
//! said nothing about the surface a client reaches, and would have kept passing
//! if either filter regressed.
//!
//! One consequence is visible in the shape of this file: there is no
//! `POST /api/models/load` step. That route is refused by the served surface, so
//! the model is loaded exactly the way a real replica loads it — in-process,
//! through the unfiltered engine router, before the served view is composed. The
//! refusal is then asserted here rather than worked around.

use axum::body::{to_bytes, Body};
use axum::http::{header::CONTENT_TYPE, Request, StatusCode};
use camelid_enterprise::{
    apply_deterministic, replica_router, Attribution, ModelIdentity, WorkerThreads,
};
use http_body_util::BodyExt;
use serde_json::{json, Value};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;
use tower::ServiceExt;

const MODEL_ENV: &str = "CAMELID_ENTERPRISE_TEST_MODEL";
/// Optional: the model file's SHA-256 as an authority outside this process
/// computed it. When set, the digest this replica publishes is checked against
/// it. That is what makes `x-camelid-model-sha256` evidence rather than a
/// restatement — the CI job that supplies this value verifies the downloaded
/// artifact with `sha256sum --check` before the test runs, so the assertion
/// closes on a number no code in this workspace produced.
///
/// Outside the `CAMELID_` namespace, unlike its `CAMELID_ENTERPRISE_TEST_MODEL`
/// sibling, and that is not an inconsistency to tidy up. Admission is
/// deny-by-default over the whole prefix and permits four names by exact match,
/// so a fifth `CAMELID_`-prefixed variable would be refused by the very scan
/// this test then calls — and admitting it would mint a new public
/// `admission_sha256` for a test harness convenience. A name outside the
/// namespace costs nothing and changes no published identity.
const MODEL_SHA_ENV: &str = "ENTERPRISE_TEST_MODEL_SHA256";
// The deterministic lane serializes generation, so the queue-saturation test
// spawns concurrent requests that decode one at a time behind a shared lock.
// Each task's timeout is measured from spawn, so a task queued behind many
// slow decodes on constrained CI hardware can wait well past its own decode
// time. This bound is a hang detector, not a performance target; the CI job's
// own timeout-minutes remains the real ceiling on total wall-clock time.
const STEP_TIMEOUT: Duration = Duration::from_secs(900);

const TEST_HOST: &str = "contract-test/host";

/// Every identity field this replica publishes, as this run resolved them.
struct Published {
    config: String,
    admission: String,
    model: String,
    workers: usize,
}

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

/// All six identity fields, on every response this harness drives.
///
/// The first three were the whole of this check before, which left the model
/// digest — the field that changes every token while changing nothing else, and
/// the only one this harness can verify against real weights — unasserted on the
/// one path CI runs with a model loaded.
fn assert_attribution(response: &axum::response::Response, expected: &Published) {
    assert_eq!(response.headers()["x-camelid-lane"], "deterministic");
    assert_eq!(
        response.headers()["x-camelid-config-sha256"],
        &expected.config[..12]
    );
    assert_eq!(
        response.headers()["x-camelid-admission-sha256"],
        &expected.admission[..12]
    );
    assert_eq!(
        response.headers()["x-camelid-model-sha256"],
        expected.model.as_str()
    );
    assert_eq!(response.headers()["x-camelid-host"], TEST_HOST);
    assert_eq!(
        response.headers()["x-camelid-worker-threads"],
        expected.workers.to_string()
    );
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

    // Read back from the pool rather than chosen, exactly as `serve` does: a
    // width this harness invented would be a published number no part of this
    // process actually runs at.
    let workers = WorkerThreads::resolved(rayon::current_num_threads());
    let model_identity = ModelIdentity::of_file(&model).expect("the model file must be readable");
    if let Ok(external) = std::env::var(MODEL_SHA_ENV) {
        // `starts_with` rather than a slice: the published form is the leading
        // twelve characters of a sixty-four character digest, and a short or
        // empty value here must fail this assertion with its message rather
        // than panic on a byte index.
        let external = external.trim().to_ascii_lowercase();
        assert!(
            external.starts_with(model_identity.short()),
            "the digest this replica is about to publish ({}) disagrees with the digest \
             {MODEL_SHA_ENV} says the file has ({external})",
            model_identity.short()
        );
    }

    let expected = Published {
        config: config.sha256.clone(),
        admission: config.admission_sha256.clone(),
        model: model_identity.short().to_string(),
        workers: workers.count(),
    };
    let identity = Attribution {
        lane: "deterministic",
        config_sha256: Arc::new(config.sha256),
        admission_sha256: Arc::new(config.admission_sha256),
        model: model_identity,
        host: Arc::new(TEST_HOST.to_string()),
        workers,
        receipts: None,
    };
    let state = camelid::api::AppState::with_configured_threads(Some(4))
        .with_default_enable_thinking(false)
        .with_models_dir(None);
    // The load happens in here, through the unfiltered engine router, before the
    // served view exists — so `app` is the stack a client meets and nothing else
    // in this file needs the control plane.
    let (app, model_id) = replica_router(
        camelid::api::router_with_state(state),
        &model,
        &model,
        identity,
    )
    .await
    .expect("the replica must load the model it was pointed at");

    // The route the load used is refused on the surface that is served, which is
    // what makes "loaded in-process" a property rather than a preference. Asserted
    // against real weights, because this is the one run where a hole here would
    // let a client swap them.
    let control_plane = send(
        app.clone(),
        post_json("/api/models/load", json!({ "path": model.to_string_lossy() })),
    )
    .await;
    assert_eq!(control_plane.status(), StatusCode::FORBIDDEN);
    assert_attribution(&control_plane, &expected);
    assert_eq!(
        body_json(control_plane).await["error"]["code"],
        "route_not_served"
    );

    let health = send(
        app.clone(),
        Request::get("/v1/health").body(Body::empty()).unwrap(),
    )
    .await;
    assert_eq!(health.status(), StatusCode::OK);
    assert_attribution(&health, &expected);
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

    // The compatibility routes are contractual as *explicit* refusals: the
    // engine's own typed 501 is the promise, and it can only be kept by letting
    // the request through the filter to the code that gives it.
    for (path, code) in [
        ("/v1/embeddings", "unsupported_embeddings"),
        ("/v1/responses", "unsupported_responses"),
        ("/v1/messages", "unsupported_messages"),
        ("/v1/rerank", "unsupported_reranking"),
        ("/v1/reranking", "unsupported_reranking"),
    ] {
        let response = send(app.clone(), post_json(path, json!({}))).await;
        assert_eq!(response.status(), StatusCode::NOT_IMPLEMENTED, "{path}");
        assert_attribution(&response, &expected);
        let body = body_json(response).await;
        assert_eq!(body["error"]["type"], "not_implemented", "{path}");
        assert_eq!(body["error"]["code"], code, "{path}");
    }

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
    assert_attribution(&first, &expected);
    let first = body_json(first).await;
    assert_eq!(first["camelid_lane"], "deterministic");
    assert_eq!(first["camelid_config_sha256"], &expected.config[..12]);
    assert_eq!(first["camelid_model_sha256"], expected.model);
    assert_eq!(first["camelid_admission_sha256"], &expected.admission[..12]);
    assert_eq!(first["camelid_host"], TEST_HOST);
    assert_eq!(first["camelid_worker_threads"], json!(expected.workers));

    let second = send(app.clone(), post_json("/v1/chat/completions", request)).await;
    assert_eq!(second.status(), StatusCode::OK);
    let second = body_json(second).await;
    assert_eq!(first["choices"], second["choices"]);

    // The body filter, against the weights it is protecting. The named path is
    // real and readable — the engine's resolver branches on `exists()`, so a
    // path that is not there would pass this for the wrong reason.
    let by_path = send(
        app.clone(),
        post_json(
            "/v1/chat/completions",
            json!({
                "model": model.to_string_lossy(),
                "messages": [{ "role": "user", "content": "Reply briefly." }],
                "max_tokens": 1
            }),
        ),
    )
    .await;
    // The replica's own weights, named by path, are one of the spellings it
    // answers to: it is rewritten to the engine's key and served.
    assert_eq!(by_path.status(), StatusCode::OK);
    let foreign = send(
        app.clone(),
        post_json(
            "/v1/chat/completions",
            json!({
                "model": "Cargo.toml",
                "messages": [{ "role": "user", "content": "Reply briefly." }],
                "max_tokens": 1
            }),
        ),
    )
    .await;
    assert_eq!(foreign.status(), StatusCode::NOT_FOUND);
    assert_attribution(&foreign, &expected);
    assert_eq!(
        body_json(foreign).await["error"]["code"],
        "model_not_served"
    );

    let raw_request = json!({
        "model": model_id,
        "prompt": "Complete briefly:",
        "temperature": 0,
        "max_tokens": 4,
        "stream": false
    });
    let raw = send(app.clone(), post_json("/v1/completions", raw_request)).await;
    assert_eq!(raw.status(), StatusCode::OK);
    assert_attribution(&raw, &expected);
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
    assert_attribution(&stream, &expected);
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
        assert_attribution(&response, &expected);
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
                assert_eq!(body["camelid_config_sha256"], &expected.config[..12]);
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
