//! In-tree replica HTTP adapter under construction.
//!
//! This router is deliberately separate from [`crate::attributed_router`], the
//! production router backed by the pinned Camelid dependency. It implements the
//! first self-owned contract slice — health, exact model discovery, and
//! deterministic non-streaming text completion — so parity can be proven
//! before a production cutover. It must not be merged with the pinned router:
//! axum rejects overlapping method/path registrations.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use axum::extract::rejection::JsonRejection;
use axum::extract::{Path as AxumPath, State};
use axum::http::{HeaderValue, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use engine_core::runtime::{Completion, FinishReason, LoadedModel};
use engine_core::{EngineError, Result as EngineResult};
use serde::{Deserialize, Serialize};
use tokio::sync::Semaphore;

const DEFAULT_MAX_TOKENS: usize = 16;

/// Contract paths currently implemented by [`router`]. This is a strict subset
/// of `replica_contract::PUBLIC_ROUTES`; tests keep that relationship explicit.
pub const IMPLEMENTED_ROUTE_PATHS: &[&str] = &[
    "/v1/health",
    "/v1/models",
    "/v1/models/:model",
    "/v1/completions",
];

/// Stable model facts needed by discovery and completion handlers.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ModelDescriptor {
    pub id: String,
    pub architecture: String,
    pub context_length: u32,
    pub embedding_length: u32,
    pub vocab_size: u32,
    pub file_type: Option<u32>,
    pub size_bytes: Option<u64>,
}

/// Synchronous model boundary used by the HTTP adapter.
///
/// Implementations run on Tokio's blocking pool. Production admission policy
/// belongs in the server; the engine remains synchronous and host-neutral.
pub trait CompletionBackend: Send + Sync + 'static {
    fn descriptor(&self) -> &ModelDescriptor;
    fn complete(&self, prompt: &str, max_new_tokens: usize) -> EngineResult<Completion>;
}

/// Adapter from the in-tree engine runtime to [`CompletionBackend`].
pub struct LoadedModelBackend {
    descriptor: ModelDescriptor,
    model: LoadedModel,
}

impl LoadedModelBackend {
    pub fn new(model_id: impl Into<String>, model: LoadedModel) -> Result<Self, String> {
        let model_id = model_id.into();
        if model_id.trim().is_empty() {
            return Err("model id must not be empty".to_string());
        }
        let config = model.config();
        let descriptor = ModelDescriptor {
            id: model_id,
            architecture: model.architecture().to_string(),
            context_length: config.context_length,
            embedding_length: config.embedding_length,
            vocab_size: config
                .vocab_size
                .expect("LoadedModel validates the vocabulary size"),
            file_type: config.file_type,
            size_bytes: std::fs::metadata(model.source())
                .ok()
                .map(|metadata| metadata.len()),
        };
        Ok(Self { descriptor, model })
    }
}

impl CompletionBackend for LoadedModelBackend {
    fn descriptor(&self) -> &ModelDescriptor {
        &self.descriptor
    }

    fn complete(&self, prompt: &str, max_new_tokens: usize) -> EngineResult<Completion> {
        self.model.complete(prompt, max_new_tokens)
    }
}

#[derive(Clone)]
struct ApiState {
    backend: Arc<dyn CompletionBackend>,
    /// The deterministic lane admits one whole generation at a time. This
    /// early adapter has no waiting queue: a concurrent request gets the same
    /// typed overload signal the production contract already defines.
    generation_slot: Arc<Semaphore>,
}

/// Build the isolated in-tree contract slice.
///
/// This function intentionally does not attach lane attribution and is not
/// called by the serving binary yet. The eventual cutover composes this router
/// with attribution only after model-backed parity is complete.
pub fn router(backend: Arc<dyn CompletionBackend>) -> Router {
    let state = ApiState {
        backend,
        generation_slot: Arc::new(Semaphore::new(1)),
    };
    Router::new()
        .route("/v1/health", get(health))
        .route("/v1/models", get(models))
        .route("/v1/models/:model", get(model))
        .route("/v1/completions", post(completions))
        .with_state(state)
}

#[derive(Serialize)]
struct HealthResponse {
    ok: bool,
    engine: &'static str,
    loaded_now: bool,
    generation_ready: bool,
    active_model_id: String,
    backend: &'static str,
    engine_queue_depth: usize,
}

async fn health(State(state): State<ApiState>) -> Json<HealthResponse> {
    Json(HealthResponse {
        ok: true,
        engine: "camelid",
        loaded_now: true,
        generation_ready: true,
        active_model_id: state.backend.descriptor().id.clone(),
        backend: "engine-core",
        engine_queue_depth: usize::from(state.generation_slot.available_permits() == 0),
    })
}

#[derive(Serialize)]
struct ModelListResponse {
    object: &'static str,
    data: Vec<ModelListItem>,
}

#[derive(Serialize)]
struct ModelListItem {
    id: String,
    object: &'static str,
    created: u64,
    owned_by: &'static str,
    meta: ModelListMeta,
}

#[derive(Serialize)]
struct ModelListMeta {
    n_vocab: u32,
    n_ctx_train: u32,
    n_embd: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    size: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    file_type: Option<u32>,
    architecture: String,
}

fn model_item(descriptor: &ModelDescriptor) -> ModelListItem {
    ModelListItem {
        id: descriptor.id.clone(),
        object: "model",
        created: 0,
        owned_by: "camelid",
        meta: ModelListMeta {
            n_vocab: descriptor.vocab_size,
            n_ctx_train: descriptor.context_length,
            n_embd: descriptor.embedding_length,
            size: descriptor.size_bytes,
            file_type: descriptor.file_type,
            architecture: descriptor.architecture.clone(),
        },
    }
}

async fn models(State(state): State<ApiState>) -> Json<ModelListResponse> {
    Json(ModelListResponse {
        object: "list",
        data: vec![model_item(state.backend.descriptor())],
    })
}

async fn model(AxumPath(model_id): AxumPath<String>, State(state): State<ApiState>) -> Response {
    let descriptor = state.backend.descriptor();
    if model_id == descriptor.id {
        return Json(model_item(descriptor)).into_response();
    }
    api_error(
        StatusCode::NOT_FOUND,
        "model_not_found",
        format!("model '{model_id}' is not loaded"),
        Some("model"),
    )
}

#[derive(Debug, Deserialize)]
struct CompletionRequest {
    model: Option<String>,
    prompt: Option<String>,
    stream: Option<bool>,
    max_tokens: Option<u32>,
    temperature: Option<f32>,
    #[serde(flatten)]
    unsupported_fields: BTreeMap<String, serde_json::Value>,
}

#[derive(Serialize)]
struct CompletionResponse {
    id: String,
    object: &'static str,
    created: u64,
    model: String,
    choices: Vec<CompletionChoice>,
    usage: CompletionUsage,
    camelid: CompletionDiagnostics,
}

#[derive(Serialize)]
struct CompletionChoice {
    index: u32,
    text: String,
    finish_reason: &'static str,
}

#[derive(Serialize)]
struct CompletionUsage {
    prompt_tokens: usize,
    completion_tokens: usize,
    total_tokens: usize,
}

#[derive(Serialize)]
struct CompletionDiagnostics {
    prompt_token_ids: Vec<u32>,
    generated_token_ids: Vec<u32>,
}

async fn completions(
    State(state): State<ApiState>,
    payload: Result<Json<CompletionRequest>, JsonRejection>,
) -> Response {
    let Json(request) = match payload {
        Ok(request) => request,
        Err(error) => {
            return api_error(
                StatusCode::BAD_REQUEST,
                "malformed_json",
                error.to_string(),
                None,
            )
        }
    };

    if !request.unsupported_fields.is_empty() {
        return api_error(
            StatusCode::BAD_REQUEST,
            "unsupported_parameter",
            format!(
                "unsupported completion field(s): {}",
                request
                    .unsupported_fields
                    .keys()
                    .cloned()
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
            Some("body"),
        );
    }
    let descriptor = state.backend.descriptor();
    if request
        .model
        .as_deref()
        .is_some_and(|model| model != descriptor.id)
    {
        let requested = request.model.as_deref().unwrap_or_default();
        return api_error(
            StatusCode::NOT_FOUND,
            "model_not_found",
            format!("model '{requested}' is not loaded"),
            Some("model"),
        );
    }
    let Some(prompt) = request.prompt else {
        return api_error(
            StatusCode::BAD_REQUEST,
            "missing_required_parameter",
            "prompt is required".to_string(),
            Some("prompt"),
        );
    };
    if request.stream.unwrap_or(false) {
        return api_error(
            StatusCode::NOT_IMPLEMENTED,
            "unsupported_streaming",
            "the in-tree completion adapter does not implement stream:true yet".to_string(),
            Some("stream"),
        );
    }
    if request
        .temperature
        .is_some_and(|temperature| temperature != 0.0)
    {
        return api_error(
            StatusCode::BAD_REQUEST,
            "unsupported_sampling",
            "the deterministic lane accepts only temperature:0".to_string(),
            Some("temperature"),
        );
    }

    let max_new_tokens = request
        .max_tokens
        .map_or(DEFAULT_MAX_TOKENS, |value| value as usize);
    let permit = match Arc::clone(&state.generation_slot).try_acquire_owned() {
        Ok(permit) => permit,
        Err(_) => {
            let mut response = api_error(
                StatusCode::SERVICE_UNAVAILABLE,
                "engine_queue_full",
                "the deterministic generation slot is busy".to_string(),
                None,
            );
            response
                .headers_mut()
                .insert("retry-after", HeaderValue::from_static("1"));
            return response;
        }
    };

    let backend = Arc::clone(&state.backend);
    let result = tokio::task::spawn_blocking(move || {
        let _permit = permit;
        backend.complete(&prompt, max_new_tokens)
    })
    .await;
    let completion = match result {
        Ok(Ok(completion)) => completion,
        Ok(Err(error)) => return engine_error(error),
        Err(error) => {
            return api_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "generation_task_failed",
                format!("generation task failed: {error}"),
                None,
            )
        }
    };

    let prompt_tokens = completion.prompt_tokens.len();
    let completion_tokens = completion.generated_tokens.len();
    let finish_reason = match completion.finish_reason {
        FinishReason::EndOfGeneration => "stop",
        FinishReason::Length => "length",
    };
    Json(CompletionResponse {
        id: next_completion_id(),
        object: "text_completion",
        created: unix_seconds(),
        model: descriptor.id.clone(),
        choices: vec![CompletionChoice {
            index: 0,
            text: completion.text,
            finish_reason,
        }],
        usage: CompletionUsage {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens + completion_tokens,
        },
        camelid: CompletionDiagnostics {
            prompt_token_ids: completion.prompt_tokens,
            generated_token_ids: completion.generated_tokens,
        },
    })
    .into_response()
}

fn engine_error(error: EngineError) -> Response {
    match error {
        EngineError::ShapeMismatch(message) => api_error(
            StatusCode::BAD_REQUEST,
            "context_length_exceeded",
            message,
            Some("max_tokens"),
        ),
        other => api_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            "generation_failed",
            other.to_string(),
            None,
        ),
    }
}

#[derive(Serialize)]
struct ErrorEnvelope {
    error: ErrorBody,
}

#[derive(Serialize)]
struct ErrorBody {
    message: String,
    #[serde(rename = "type")]
    error_type: &'static str,
    code: &'static str,
    param: Option<&'static str>,
}

fn api_error(
    status: StatusCode,
    code: &'static str,
    message: String,
    param: Option<&'static str>,
) -> Response {
    let error_type = match status {
        StatusCode::INTERNAL_SERVER_ERROR => "server_error",
        StatusCode::NOT_IMPLEMENTED => "not_implemented",
        StatusCode::SERVICE_UNAVAILABLE => "runtime_unavailable",
        _ => "invalid_request",
    };
    (
        status,
        Json(ErrorEnvelope {
            error: ErrorBody {
                message,
                error_type,
                code,
                param,
            },
        }),
    )
        .into_response()
}

fn unix_seconds() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or(0)
}

fn next_completion_id() -> String {
    static NEXT_ID: AtomicU64 = AtomicU64::new(1);
    format!("cmpl-in-tree-{}", NEXT_ID.fetch_add(1, Ordering::Relaxed))
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::{to_bytes, Body};
    use axum::http::header::CONTENT_TYPE;
    use axum::http::Request;
    use serde_json::{json, Value};
    use std::sync::Barrier;
    use tower::ServiceExt;

    struct FakeBackend {
        descriptor: ModelDescriptor,
    }

    impl CompletionBackend for FakeBackend {
        fn descriptor(&self) -> &ModelDescriptor {
            &self.descriptor
        }

        fn complete(&self, prompt: &str, max_new_tokens: usize) -> EngineResult<Completion> {
            if prompt == "overflow" {
                return Err(EngineError::ShapeMismatch(
                    "prompt tokens (8) plus max new tokens (1) exceed context length 8".to_string(),
                ));
            }
            let eog = prompt == "stop";
            Ok(Completion {
                prompt_tokens: vec![10, 11],
                generated_tokens: if eog {
                    vec![0]
                } else {
                    vec![20; max_new_tokens]
                },
                text: if eog {
                    "done".to_string()
                } else {
                    "x".repeat(max_new_tokens)
                },
                finish_reason: if eog {
                    FinishReason::EndOfGeneration
                } else {
                    FinishReason::Length
                },
            })
        }
    }

    struct BlockingBackend {
        descriptor: ModelDescriptor,
        started: Arc<Barrier>,
        release: Arc<Barrier>,
    }

    impl CompletionBackend for BlockingBackend {
        fn descriptor(&self) -> &ModelDescriptor {
            &self.descriptor
        }

        fn complete(&self, _prompt: &str, _max_new_tokens: usize) -> EngineResult<Completion> {
            self.started.wait();
            self.release.wait();
            Ok(Completion {
                prompt_tokens: vec![10],
                generated_tokens: vec![20],
                text: "x".to_string(),
                finish_reason: FinishReason::Length,
            })
        }
    }

    fn app() -> Router {
        router(Arc::new(FakeBackend {
            descriptor: ModelDescriptor {
                id: "test-model".to_string(),
                architecture: "llama".to_string(),
                context_length: 8,
                embedding_length: 2,
                vocab_size: 32,
                file_type: Some(7),
                size_bytes: Some(1024),
            },
        }))
    }

    async fn send(request: Request<Body>) -> (StatusCode, axum::http::HeaderMap, Value) {
        let response = app().oneshot(request).await.unwrap();
        let status = response.status();
        let headers = response.headers().clone();
        let bytes = to_bytes(response.into_body(), 1024 * 1024).await.unwrap();
        let body = serde_json::from_slice(&bytes).unwrap_or(Value::Null);
        (status, headers, body)
    }

    fn post(body: Value) -> Request<Body> {
        Request::post("/v1/completions")
            .header(CONTENT_TYPE, "application/json")
            .body(Body::from(body.to_string()))
            .unwrap()
    }

    #[test]
    fn implemented_paths_are_a_strict_contract_subset() {
        assert!(IMPLEMENTED_ROUTE_PATHS.len() < replica_contract::PUBLIC_ROUTES.len());
        for path in IMPLEMENTED_ROUTE_PATHS {
            assert!(replica_contract::PUBLIC_ROUTES
                .iter()
                .any(|route| route.path == *path));
        }
    }

    #[tokio::test]
    async fn health_and_discovery_describe_the_exact_model() {
        let (status, _, health) =
            send(Request::get("/v1/health").body(Body::empty()).unwrap()).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(health["ok"], true);
        assert_eq!(health["generation_ready"], true);
        assert_eq!(health["active_model_id"], "test-model");
        assert_eq!(health["engine_queue_depth"], 0);

        let (status, _, models) =
            send(Request::get("/v1/models").body(Body::empty()).unwrap()).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(models["object"], "list");
        assert_eq!(models["data"].as_array().unwrap().len(), 1);
        assert_eq!(models["data"][0]["id"], "test-model");
        assert_eq!(models["data"][0]["meta"]["architecture"], "llama");
        assert_eq!(models["data"][0]["meta"]["n_ctx_train"], 8);

        let (status, _, model) = send(
            Request::get("/v1/models/test-model")
                .body(Body::empty())
                .unwrap(),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(model["id"], "test-model");
    }

    #[tokio::test]
    async fn missing_model_uses_the_contract_error_shape() {
        let (status, _, body) = send(
            Request::get("/v1/models/missing")
                .body(Body::empty())
                .unwrap(),
        )
        .await;
        assert_eq!(status, StatusCode::NOT_FOUND);
        assert_eq!(body["error"]["type"], "invalid_request");
        assert_eq!(body["error"]["code"], "model_not_found");
        assert_eq!(body["error"]["param"], "model");
    }

    #[tokio::test]
    async fn completion_returns_openai_shape_usage_and_diagnostics() {
        let (status, _, body) = send(post(json!({
            "model": "test-model",
            "prompt": "hello",
            "temperature": 0,
            "max_tokens": 2,
            "stream": false
        })))
        .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body["object"], "text_completion");
        assert_eq!(body["model"], "test-model");
        assert_eq!(body["choices"][0]["text"], "xx");
        assert_eq!(body["choices"][0]["finish_reason"], "length");
        assert_eq!(body["usage"]["prompt_tokens"], 2);
        assert_eq!(body["usage"]["completion_tokens"], 2);
        assert_eq!(body["usage"]["total_tokens"], 4);
        assert_eq!(body["camelid"]["prompt_token_ids"], json!([10, 11]));
        assert_eq!(body["camelid"]["generated_token_ids"], json!([20, 20]));
    }

    #[tokio::test]
    async fn eog_maps_to_the_openai_stop_finish_reason() {
        let (status, _, body) = send(post(json!({
            "prompt": "stop",
            "temperature": 0,
            "max_tokens": 4
        })))
        .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body["choices"][0]["finish_reason"], "stop");
        assert_eq!(body["usage"]["completion_tokens"], 1);
        assert_eq!(body["camelid"]["generated_token_ids"], json!([0]));
    }

    #[tokio::test]
    async fn unsupported_request_features_fail_closed() {
        let cases = [
            (
                json!({"model":"other","prompt":"hi"}),
                StatusCode::NOT_FOUND,
                "model_not_found",
            ),
            (
                json!({"prompt":"hi","stream":true}),
                StatusCode::NOT_IMPLEMENTED,
                "unsupported_streaming",
            ),
            (
                json!({"prompt":"hi","temperature":0.5}),
                StatusCode::BAD_REQUEST,
                "unsupported_sampling",
            ),
            (
                json!({"prompt":"hi","top_p":0.9}),
                StatusCode::BAD_REQUEST,
                "unsupported_parameter",
            ),
            (
                json!({"model":"test-model"}),
                StatusCode::BAD_REQUEST,
                "missing_required_parameter",
            ),
        ];
        for (request, expected_status, expected_code) in cases {
            let (status, _, body) = send(post(request)).await;
            assert_eq!(status, expected_status);
            assert_eq!(body["error"]["code"], expected_code);
        }
    }

    #[tokio::test]
    async fn malformed_json_and_engine_context_errors_are_typed() {
        let malformed = Request::post("/v1/completions")
            .header(CONTENT_TYPE, "application/json")
            .body(Body::from("{"))
            .unwrap();
        let (status, _, body) = send(malformed).await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_eq!(body["error"]["code"], "malformed_json");

        let (status, _, body) = send(post(json!({"prompt":"overflow","max_tokens":1}))).await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_eq!(body["error"]["code"], "context_length_exceeded");
        assert_eq!(body["error"]["param"], "max_tokens");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn concurrent_generation_is_rejected_without_queueing() {
        let started = Arc::new(Barrier::new(2));
        let release = Arc::new(Barrier::new(2));
        let app = router(Arc::new(BlockingBackend {
            descriptor: ModelDescriptor {
                id: "test-model".to_string(),
                architecture: "llama".to_string(),
                context_length: 8,
                embedding_length: 2,
                vocab_size: 32,
                file_type: None,
                size_bytes: None,
            },
            started: Arc::clone(&started),
            release: Arc::clone(&release),
        }));

        let first_app = app.clone();
        let first = tokio::spawn(async move {
            first_app
                .oneshot(post(json!({"prompt":"first","max_tokens":1})))
                .await
                .unwrap()
        });
        started.wait();

        let busy = app
            .clone()
            .oneshot(post(json!({"prompt":"second","max_tokens":1})))
            .await
            .unwrap();
        assert_eq!(busy.status(), StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(busy.headers()["retry-after"], "1");
        let bytes = to_bytes(busy.into_body(), 1024 * 1024).await.unwrap();
        let body: Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(body["error"]["type"], "runtime_unavailable");
        assert_eq!(body["error"]["code"], "engine_queue_full");

        let health = app
            .oneshot(Request::get("/v1/health").body(Body::empty()).unwrap())
            .await
            .unwrap();
        let bytes = to_bytes(health.into_body(), 1024 * 1024).await.unwrap();
        let health: Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(health["engine_queue_depth"], 1);

        release.wait();
        assert_eq!(first.await.unwrap().status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn private_and_unimplemented_routes_are_not_exposed() {
        for path in ["/api/models/load", "/v1/chat/completions"] {
            let response = app()
                .oneshot(Request::post(path).body(Body::empty()).unwrap())
                .await
                .unwrap();
            assert_eq!(response.status(), StatusCode::NOT_FOUND, "{path}");
        }
    }
}
