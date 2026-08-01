//! Model-backed conformance: the surface a client meets, against real weights.
//!
//! This is the only executable check in the workspace that runs the replica with
//! a GGUF actually loaded, so what it drives matters. It drives
//! [`replica_router`] — the same in-tree composition `serve` uses — rather than
//! a test-only router assembled by this file.
//!
//! There is no `POST /api/models/load` step. The model is an owned runtime value
//! constructed before the served view, and the absent control plane is asserted
//! here rather than worked around.

use axum::body::{to_bytes, Body};
use axum::http::{header::CONTENT_TYPE, Request, StatusCode};
use camelid_enterprise::in_tree::{router as in_tree_router, LoadedModelBackend};
use camelid_enterprise::{
    apply_deterministic, load_startup_model, replica_router, Attribution, ModelIdentity,
    WorkerThreads,
};
use engine_core::runtime::{GenerationControl, IncrementalGeneration, LoadedModel};
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

fn test_model_path() -> PathBuf {
    PathBuf::from(
        std::env::var(MODEL_ENV)
            .unwrap_or_else(|_| panic!("{MODEL_ENV} must name a compatible local GGUF")),
    )
    .canonicalize()
    .unwrap_or_else(|error| panic!("could not resolve {MODEL_ENV}: {error}"))
}

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

#[derive(Clone, Debug, PartialEq, Eq)]
struct GenerationSnapshot {
    prompt_token_ids: Vec<u32>,
    generated_token_ids: Vec<u32>,
    text: String,
    finish_reason: String,
    prompt_tokens: u64,
    completion_tokens: u64,
    total_tokens: u64,
}

impl GenerationSnapshot {
    fn from_response(body: &Value, chat: bool) -> Self {
        let token_ids = |field: &str| {
            body["camelid"][field]
                .as_array()
                .unwrap_or_else(|| panic!("camelid.{field} must be an array: {body}"))
                .iter()
                .map(|token| {
                    token
                        .as_u64()
                        .and_then(|token| u32::try_from(token).ok())
                        .unwrap_or_else(|| panic!("camelid.{field} contains a non-u32: {body}"))
                })
                .collect()
        };
        let text = if chat {
            &body["choices"][0]["message"]["content"]
        } else {
            &body["choices"][0]["text"]
        };
        Self {
            prompt_token_ids: token_ids("prompt_token_ids"),
            generated_token_ids: token_ids("generated_token_ids"),
            text: text
                .as_str()
                .unwrap_or_else(|| panic!("completion text must be a string: {body}"))
                .to_string(),
            finish_reason: body["choices"][0]["finish_reason"]
                .as_str()
                .unwrap_or_else(|| panic!("finish_reason must be a string: {body}"))
                .to_string(),
            prompt_tokens: body["usage"]["prompt_tokens"]
                .as_u64()
                .unwrap_or_else(|| panic!("prompt_tokens must be a u64: {body}")),
            completion_tokens: body["usage"]["completion_tokens"]
                .as_u64()
                .unwrap_or_else(|| panic!("completion_tokens must be a u64: {body}")),
            total_tokens: body["usage"]["total_tokens"]
                .as_u64()
                .unwrap_or_else(|| panic!("total_tokens must be a u64: {body}")),
        }
    }
}

#[derive(Clone)]
struct ParityCase {
    name: &'static str,
    path: &'static str,
    request: Value,
    chat: bool,
}

fn generation_parity_cases(model_id: &str) -> Vec<ParityCase> {
    vec![
        ParityCase {
            name: "raw-short",
            path: "/v1/completions",
            request: json!({
                "model": model_id,
                "prompt": "Complete this sentence: The capital of France is",
                "temperature": 0,
                "max_tokens": 4,
                "stream": false
            }),
            chat: false,
        },
        ParityCase {
            name: "raw-unicode",
            path: "/v1/completions",
            request: json!({
                "model": model_id,
                "prompt": "Unicode fidelity: café, 東京, 🦙 —",
                "temperature": 0,
                "max_tokens": 2,
                "stream": false
            }),
            chat: false,
        },
        ParityCase {
            name: "chat-single-turn",
            path: "/v1/chat/completions",
            request: json!({
                "model": model_id,
                "messages": [{"role":"user","content":"Reply with one word: blue."}],
                "temperature": 0,
                "max_tokens": 4,
                "stream": false
            }),
            chat: true,
        },
        ParityCase {
            name: "chat-multi-turn-unicode",
            path: "/v1/chat/completions",
            request: json!({
                "model": model_id,
                "messages": [
                    {"role":"system","content":"Answer tersely."},
                    {"role":"user","content":"Say café."},
                    {"role":"assistant","content":"café"},
                    {"role":"user","content":"Now say 東京."}
                ],
                "temperature": 0,
                "max_tokens": 2,
                "stream": false
            }),
            chat: true,
        },
    ]
}

async fn capture_generation(app: axum::Router, case: &ParityCase) -> GenerationSnapshot {
    let response = send(app, post_json(case.path, case.request.clone())).await;
    let status = response.status();
    let body = body_json(response).await;
    assert_eq!(
        status,
        StatusCode::OK,
        "{} generation must succeed: {body}",
        case.name,
    );
    GenerationSnapshot::from_response(&body, case.chat)
}

#[tokio::test]
#[ignore = "requires CAMELID_ENTERPRISE_TEST_MODEL to name a compatible local GGUF"]
async fn real_model_conforms_to_replica_http_v1() {
    let model = test_model_path();
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
    let loaded_model = LoadedModel::load(&model)
        .expect("the in-tree runtime must load the model it was pointed at");
    // One slot: this test asserts the exact depth at which the bounded queue
    // refuses, which is `slots + queue depth`. Taking the runner's default width
    // would make that assertion a fact about the CI machine.
    let (app, model_id) = replica_router(loaded_model, &model, identity, 1)
        .expect("the production replica composition must accept the loaded model");

    // The route the load used is refused on the surface that is served, which is
    // what makes "loaded in-process" a property rather than a preference. Asserted
    // against real weights, because this is the one run where a hole here would
    // let a client swap them.
    let control_plane = send(
        app.clone(),
        post_json(
            "/api/models/load",
            json!({ "path": model.to_string_lossy() }),
        ),
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

/// Compare the owned runtime to the exact pinned engine revision at the token
/// boundary. The two engines are loaded sequentially: the pinned router is
/// explicitly unloaded and dropped before `engine-core` reads the same GGUF,
/// keeping this gate viable on ordinary hosted CI memory limits.
#[tokio::test]
#[ignore = "requires CAMELID_ENTERPRISE_TEST_MODEL to name a compatible local GGUF"]
async fn in_tree_generation_matches_pinned_engine() {
    let model = test_model_path();

    let pinned = camelid::api::router_with_state(
        camelid::api::AppState::with_configured_threads(Some(4))
            .with_default_enable_thinking(false)
            .with_models_dir(None),
    );
    let model_id = load_startup_model(pinned.clone(), &model)
        .await
        .expect("the pinned engine must load the parity model");
    let cases = generation_parity_cases(&model_id);
    let mut expected = Vec::with_capacity(cases.len());
    for case in &cases {
        expected.push(capture_generation(pinned.clone(), case).await);
    }

    let unload = send(
        pinned.clone(),
        post_json("/api/models/unload", json!({"id": model_id})),
    )
    .await;
    assert_eq!(
        unload.status(),
        StatusCode::NO_CONTENT,
        "the pinned model must unload before the in-tree runtime is loaded"
    );
    drop(pinned);

    let loaded = LoadedModel::load(&model).expect("the in-tree runtime must load the parity model");
    let backend = LoadedModelBackend::new(model_id, loaded)
        .expect("the pinned engine's model id is a valid in-tree discovery id");
    let in_tree = in_tree_router(Arc::new(backend), 1);

    for (case, expected) in cases.iter().zip(expected) {
        let actual = capture_generation(in_tree.clone(), case).await;
        assert_eq!(
            actual, expected,
            "{} diverged from pinned engine token generation",
            case.name
        );
    }
}

/// The isolated in-tree router is intentionally not the production surface
/// yet, so the production conformance test above cannot exercise it. This leg
/// loads the same independently verified GGUF through `engine-core` and proves
/// real model discovery, raw tokenization, embedded chat-template rendering,
/// special-token parsing, context admission, SSE framing, and the fail-closed
/// surface. A zero generation budget keeps the HTTP checks contractual rather
/// than turning them into a second multi-minute inference benchmark; numerics
/// and incremental equality are asserted immediately after load.
#[tokio::test]
#[ignore = "requires CAMELID_ENTERPRISE_TEST_MODEL to name a compatible local GGUF"]
async fn real_model_conforms_to_in_tree_generation_slice() {
    let model = test_model_path();
    let loaded = LoadedModel::load(&model).expect("the in-tree runtime must load the pinned GGUF");
    let context_length = loaded.config().context_length;
    let incremental_prompt = "Complete this sentence: The capital of France is";
    let blocking = loaded
        .complete(incremental_prompt, 2)
        .expect("blocking generation must establish the incremental oracle");
    let mut incremental_text = String::new();
    let incremental = loaded
        .complete_incremental(incremental_prompt, 2, |delta| {
            incremental_text.push_str(&delta.text);
            GenerationControl::Continue
        })
        .expect("incremental generation must complete");
    assert_eq!(
        incremental,
        IncrementalGeneration::Completed(blocking.clone())
    );
    assert_eq!(incremental_text, blocking.text);
    let model_id = model.file_name().unwrap().to_string_lossy().into_owned();
    let backend = LoadedModelBackend::new(model_id.clone(), loaded)
        .expect("the model filename is a valid discovery id");
    let app = in_tree_router(Arc::new(backend), 1);

    let health = send(
        app.clone(),
        Request::get("/v1/health").body(Body::empty()).unwrap(),
    )
    .await;
    assert_eq!(health.status(), StatusCode::OK);
    let health = body_json(health).await;
    assert_eq!(health["backend"], "engine-core");
    assert_eq!(health["generation_ready"], true);
    assert_eq!(health["active_model_id"], model_id);

    let models = send(
        app.clone(),
        Request::get("/v1/models").body(Body::empty()).unwrap(),
    )
    .await;
    assert_eq!(models.status(), StatusCode::OK);
    let models = body_json(models).await;
    assert_eq!(models["data"].as_array().unwrap().len(), 1);
    assert_eq!(models["data"][0]["id"], model_id);
    assert_eq!(models["data"][0]["meta"]["n_ctx_train"], context_length);

    let raw = send(
        app.clone(),
        post_json(
            "/v1/completions",
            json!({
                "model": model_id,
                "prompt": "Complete briefly:",
                "temperature": 0,
                "max_tokens": 0,
                "stream": false
            }),
        ),
    )
    .await;
    assert_eq!(raw.status(), StatusCode::OK);
    let raw = body_json(raw).await;
    assert_eq!(raw["object"], "text_completion");
    assert_eq!(raw["choices"][0]["text"], "");
    assert_eq!(raw["choices"][0]["finish_reason"], "length");
    assert!(raw["camelid"]["prompt_token_ids"]
        .as_array()
        .is_some_and(|tokens| !tokens.is_empty()));
    assert_eq!(raw["camelid"]["generated_token_ids"], json!([]));

    let chat = send(
        app.clone(),
        post_json(
            "/v1/chat/completions",
            json!({
                "model": model_id,
                "messages": [{"role":"user","content":"Reply briefly."}],
                "temperature": 0,
                "max_tokens": 0,
                "stream": false
            }),
        ),
    )
    .await;
    assert_eq!(chat.status(), StatusCode::OK);
    let chat = body_json(chat).await;
    assert_eq!(chat["object"], "chat.completion");
    assert_eq!(chat["choices"][0]["message"]["role"], "assistant");
    assert_eq!(chat["choices"][0]["message"]["content"], "");
    assert_eq!(chat["choices"][0]["finish_reason"], "length");
    assert!(chat["camelid"]["prompt_token_ids"]
        .as_array()
        .is_some_and(|tokens| !tokens.is_empty()));
    assert_eq!(chat["camelid"]["generated_token_ids"], json!([]));

    let overflow = send(
        app.clone(),
        post_json(
            "/v1/chat/completions",
            json!({
                "model": model_id,
                "messages": [{"role":"user","content":"Reply briefly."}],
                "max_tokens": context_length
            }),
        ),
    )
    .await;
    assert_eq!(overflow.status(), StatusCode::BAD_REQUEST);
    assert_eq!(
        body_json(overflow).await["error"]["code"],
        "context_length_exceeded"
    );

    let streaming = send(
        app.clone(),
        post_json(
            "/v1/chat/completions",
            json!({
                "model": model_id,
                "messages": [{"role":"user","content":"Reply briefly."}],
                "max_tokens": 0,
                "stream": true
            }),
        ),
    )
    .await;
    assert_eq!(streaming.status(), StatusCode::OK);
    assert!(streaming.headers()[CONTENT_TYPE]
        .to_str()
        .unwrap()
        .starts_with("text/event-stream"));
    let bytes = to_bytes(streaming.into_body(), 1024 * 1024).await.unwrap();
    let stream = String::from_utf8(bytes.to_vec()).unwrap();
    let events = stream
        .lines()
        .filter_map(|line| line.strip_prefix("data: "))
        .collect::<Vec<_>>();
    assert_eq!(events.last(), Some(&"[DONE]"));
    assert_eq!(events.len(), 3, "expected role, finish, and [DONE]: {stream}");
    let role: Value = serde_json::from_str(events[0]).unwrap();
    let finish: Value = serde_json::from_str(events[1]).unwrap();
    assert_eq!(role["object"], "chat.completion.chunk");
    assert_eq!(role["choices"][0]["delta"]["role"], "assistant");
    assert_eq!(finish["choices"][0]["delta"], json!({}));
    assert_eq!(finish["choices"][0]["finish_reason"], "length");

    // The control plane is not served at all, so its paths do not resolve. The
    // compatibility routes are registered and refuse explicitly instead, which
    // is the distinction a shared 404 would collapse: a client can tell "this
    // replica will never serve embeddings" from "this build has not heard of
    // the route". The exact codes and messages are pinned by the unit test
    // `compatibility_routes_preserve_the_pinned_explicit_refusals`; what this
    // adds is that a real loaded model does not change the answer.
    let absent = send(app.clone(), post_json("/api/models/load", json!({}))).await;
    assert_eq!(absent.status(), StatusCode::NOT_FOUND);

    let embeddings = send(app, post_json("/v1/embeddings", json!({}))).await;
    assert_eq!(embeddings.status(), StatusCode::NOT_IMPLEMENTED);
    let refusal = body_json(embeddings).await;
    assert_eq!(refusal["error"]["type"], "not_implemented");
    assert_eq!(refusal["error"]["code"], "unsupported_embeddings");
}
