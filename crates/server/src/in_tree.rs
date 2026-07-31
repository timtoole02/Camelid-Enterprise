//! In-tree replica HTTP adapter under construction.
//!
//! This router is deliberately separate from [`crate::attributed_router`], the
//! production router backed by the pinned Camelid dependency. It implements the
//! first self-owned contract slice — health, exact model discovery, and
//! deterministic text and chat completion, including SSE — so parity can be
//! proven before a production cutover. It must not be merged with the pinned
//! router: axum rejects overlapping method/path registrations.

mod stream;
mod worker;

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use self::worker::{GenerationWorker, PostError};
use self::stream::stream_generation;
use axum::extract::rejection::JsonRejection;
use axum::extract::{Path as AxumPath, State};
use axum::http::{HeaderValue, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use engine_core::runtime::{
    Completion, FinishReason, GenerationControl, IncrementalGeneration, LoadedModel, TokenDelta,
};
use engine_core::{EngineError, Result as EngineResult};
use minijinja::{context, Environment, ErrorKind as MiniJinjaErrorKind, UndefinedBehavior};
use serde::{Deserialize, Serialize};

const DEFAULT_MAX_TOKENS: usize = 16;
const CHAT_TEMPLATE_NAME: &str = "embedded_chat_template";

/// Contract paths currently implemented by [`router`]. This is a strict subset
/// of `replica_contract::PUBLIC_ROUTES`; tests keep that relationship explicit.
pub const IMPLEMENTED_ROUTE_PATHS: &[&str] = &[
    "/v1/health",
    "/v1/models",
    "/v1/models/:model",
    "/v1/completions",
    "/v1/chat/completions",
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

/// The embedded template and tokenizer token texts required to render chat in
/// the server without coupling HTTP schemas into the engine crate.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ChatTemplate {
    pub source: String,
    pub bos_token: String,
    pub eos_token: String,
    pub eot_token: String,
    pub eom_token: String,
    pub unk_token: String,
}

/// Synchronous model boundary used by the HTTP adapter.
///
/// Implementations run on the adapter's dedicated generation worker.
/// Production admission policy belongs in the server; the engine remains
/// synchronous and host-neutral.
pub trait CompletionBackend: Send + Sync + 'static {
    fn descriptor(&self) -> &ModelDescriptor;
    fn complete(&self, prompt: &str, max_new_tokens: usize) -> EngineResult<Completion>;
    fn complete_incremental(
        &self,
        prompt: &str,
        max_new_tokens: usize,
        on_token: &mut dyn FnMut(TokenDelta) -> GenerationControl,
    ) -> EngineResult<IncrementalGeneration>;
    fn chat_template(&self) -> Option<ChatTemplate>;
    fn complete_prompt(
        &self,
        prompt: &str,
        add_special: bool,
        parse_special: bool,
        max_new_tokens: usize,
    ) -> EngineResult<Completion>;
    fn complete_prompt_incremental(
        &self,
        prompt: &str,
        add_special: bool,
        parse_special: bool,
        max_new_tokens: usize,
        on_token: &mut dyn FnMut(TokenDelta) -> GenerationControl,
    ) -> EngineResult<IncrementalGeneration>;
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

    fn complete_incremental(
        &self,
        prompt: &str,
        max_new_tokens: usize,
        on_token: &mut dyn FnMut(TokenDelta) -> GenerationControl,
    ) -> EngineResult<IncrementalGeneration> {
        self.model
            .complete_incremental(prompt, max_new_tokens, on_token)
    }

    fn chat_template(&self) -> Option<ChatTemplate> {
        let tokenizer = self.model.tokenizer();
        tokenizer.chat_template.as_ref().map(|source| ChatTemplate {
            source: source.clone(),
            bos_token: tokenizer
                .token_text(tokenizer.special.bos)
                .unwrap_or("")
                .to_string(),
            eos_token: tokenizer
                .token_text(tokenizer.special.eos)
                .unwrap_or("")
                .to_string(),
            eot_token: tokenizer
                .token_text(tokenizer.special.eot)
                .unwrap_or("")
                .to_string(),
            eom_token: tokenizer
                .token_text(tokenizer.special.eom)
                .unwrap_or("")
                .to_string(),
            unk_token: tokenizer
                .token_text(tokenizer.special.unk)
                .unwrap_or("")
                .to_string(),
        })
    }

    fn complete_prompt(
        &self,
        prompt: &str,
        add_special: bool,
        parse_special: bool,
        max_new_tokens: usize,
    ) -> EngineResult<Completion> {
        self.model
            .complete_prompt(prompt, add_special, parse_special, max_new_tokens)
    }

    fn complete_prompt_incremental(
        &self,
        prompt: &str,
        add_special: bool,
        parse_special: bool,
        max_new_tokens: usize,
        on_token: &mut dyn FnMut(TokenDelta) -> GenerationControl,
    ) -> EngineResult<IncrementalGeneration> {
        self.model.complete_prompt_incremental(
            prompt,
            add_special,
            parse_special,
            max_new_tokens,
            on_token,
        )
    }
}

#[derive(Clone)]
struct ApiState {
    backend: Arc<dyn CompletionBackend>,
    generation_worker: GenerationWorker,
}

/// Build the isolated in-tree contract slice.
///
/// This function intentionally does not attach lane attribution and is not
/// called by the serving binary yet. The eventual cutover composes this router
/// with attribution only after model-backed parity is complete.
pub fn router(backend: Arc<dyn CompletionBackend>) -> Router {
    let state = ApiState {
        backend,
        generation_worker: GenerationWorker::spawn(),
    };
    Router::new()
        .route("/v1/health", get(health))
        .route("/v1/models", get(models))
        .route("/v1/models/:model", get(model))
        .route("/v1/completions", post(completions))
        .route("/v1/chat/completions", post(chat_completions))
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
        engine_queue_depth: state.generation_worker.depth(),
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
    let model_id = descriptor.id.clone();
    if request.stream.unwrap_or(false) {
        return stream_generation(
            &state,
            GenerationInput::RawCompletion(prompt),
            max_new_tokens,
            model_id,
            false,
        );
    }
    let completion = match run_generation(
        &state,
        GenerationInput::RawCompletion(prompt),
        max_new_tokens,
    )
    .await
    {
        Ok(completion) => completion,
        Err(response) => return response,
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
        model: model_id,
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

#[derive(Debug, Deserialize)]
struct ChatCompletionRequest {
    model: Option<String>,
    messages: Option<Vec<ChatMessage>>,
    stream: Option<bool>,
    max_tokens: Option<u32>,
    temperature: Option<f32>,
    #[serde(flatten)]
    unsupported_fields: BTreeMap<String, serde_json::Value>,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct ChatMessage {
    role: String,
    content: String,
}

#[derive(Serialize)]
struct ChatCompletionResponse {
    id: String,
    object: &'static str,
    created: u64,
    model: String,
    choices: Vec<ChatCompletionChoice>,
    usage: CompletionUsage,
    camelid: CompletionDiagnostics,
}

#[derive(Serialize)]
struct ChatCompletionChoice {
    index: u32,
    message: ChatCompletionMessage,
    finish_reason: &'static str,
}

#[derive(Serialize)]
struct ChatCompletionMessage {
    role: &'static str,
    content: String,
}

async fn chat_completions(
    State(state): State<ApiState>,
    payload: Result<Json<ChatCompletionRequest>, JsonRejection>,
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
                "unsupported chat completion field(s): {}",
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
    let Some(messages) = request.messages else {
        return api_error(
            StatusCode::BAD_REQUEST,
            "missing_required_parameter",
            "messages is required".to_string(),
            Some("messages"),
        );
    };
    if messages.is_empty() {
        return api_error(
            StatusCode::BAD_REQUEST,
            "invalid_messages",
            "messages must contain at least one chat message".to_string(),
            Some("messages"),
        );
    }
    if let Some((index, role)) = messages.iter().enumerate().find_map(|(index, message)| {
        let role = message.role.trim();
        (!matches!(role, "system" | "user" | "assistant")).then_some((index, role))
    }) {
        return api_error(
            StatusCode::BAD_REQUEST,
            "unsupported_message_role",
            format!("messages[{index}] has unsupported role '{role}'"),
            Some("messages"),
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

    let Some(template) = state.backend.chat_template() else {
        return api_error(
            StatusCode::UNPROCESSABLE_ENTITY,
            "unsupported_chat_template",
            "the loaded tokenizer has no embedded chat template; chat fails closed".to_string(),
            Some("messages"),
        );
    };
    let rendered = match render_chat_prompt(&messages, &template, &descriptor.id) {
        Ok(rendered) => rendered,
        Err(error) => {
            return api_error(
                StatusCode::UNPROCESSABLE_ENTITY,
                "unsupported_chat_template",
                format!("the embedded chat template could not render this request: {error}"),
                Some("messages"),
            )
        }
    };
    let max_new_tokens = request
        .max_tokens
        .map_or(DEFAULT_MAX_TOKENS, |value| value as usize);
    let model_id = descriptor.id.clone();
    if request.stream.unwrap_or(false) {
        return stream_generation(
            &state,
            GenerationInput::RenderedChat {
                prompt: rendered.text,
                add_special: rendered.add_special,
                parse_special: rendered.parse_special,
            },
            max_new_tokens,
            model_id,
            true,
        );
    }
    let completion = match run_generation(
        &state,
        GenerationInput::RenderedChat {
            prompt: rendered.text,
            add_special: rendered.add_special,
            parse_special: rendered.parse_special,
        },
        max_new_tokens,
    )
    .await
    {
        Ok(completion) => completion,
        Err(response) => return response,
    };

    let prompt_tokens = completion.prompt_tokens.len();
    let completion_tokens = completion.generated_tokens.len();
    let finish_reason = match completion.finish_reason {
        FinishReason::EndOfGeneration => "stop",
        FinishReason::Length => "length",
    };
    Json(ChatCompletionResponse {
        id: next_chat_completion_id(),
        object: "chat.completion",
        created: unix_seconds(),
        model: model_id,
        choices: vec![ChatCompletionChoice {
            index: 0,
            message: ChatCompletionMessage {
                role: "assistant",
                content: completion.text,
            },
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

#[derive(Serialize)]
struct ChatTemplateMessage<'a> {
    role: &'a str,
    content: &'a str,
}

struct RenderedChatPrompt {
    text: String,
    add_special: bool,
    parse_special: bool,
}

fn render_chat_prompt(
    messages: &[ChatMessage],
    template: &ChatTemplate,
    model_id: &str,
) -> Result<RenderedChatPrompt, minijinja::Error> {
    // The pinned engine deliberately uses its compact Llama 3 renderer for
    // non-Q8_0 rows. The full metadata template otherwise injects a dated
    // default system preamble, changing both prompt tokens and generation.
    // Keep that compatibility behavior until the production pin is retired;
    // the embedded template remains the authority that identifies the format.
    if is_llama3_instruct_template(&template.source)
        && !is_llama32_metadata_jinja_exact_row_model_id(model_id)
    {
        return Ok(RenderedChatPrompt {
            text: render_llama3_instruct_prompt(messages),
            add_special: true,
            parse_special: true,
        });
    }

    let rendered = render_metadata_chat_template(messages, template)?;
    let add_special = template.bos_token.is_empty() || !rendered.starts_with(&template.bos_token);
    Ok(RenderedChatPrompt {
        text: rendered,
        add_special,
        parse_special: true,
    })
}

fn is_llama3_instruct_template(template: &str) -> bool {
    template.contains("<|start_header_id|>")
        && template.contains("<|end_header_id|>")
        && template.contains("<|eot_id|>")
}

fn is_llama32_metadata_jinja_exact_row_model_id(model_id: &str) -> bool {
    let normalized = model_id
        .chars()
        .flat_map(char::to_lowercase)
        .map(|character| match character {
            '-' | '.' | ' ' => '_',
            character => character,
        })
        .collect::<String>();
    normalized.contains("llama32_1b_instruct_q8_0")
        || normalized.contains("llama_3_2_1b_instruct_q8_0")
        || normalized.contains("llama32_3b_instruct_q8_0")
        || normalized.contains("llama_3_2_3b_instruct_q8_0")
}

fn render_llama3_instruct_prompt(messages: &[ChatMessage]) -> String {
    let mut prompt = String::new();
    for message in messages {
        prompt.push_str("<|start_header_id|>");
        prompt.push_str(message.role.trim());
        prompt.push_str("<|end_header_id|>\n\n");
        prompt.push_str(&message.content);
        prompt.push_str("<|eot_id|>");
    }
    if messages
        .last()
        .is_none_or(|message| message.role.trim() != "assistant")
    {
        prompt.push_str("<|start_header_id|>assistant<|end_header_id|>\n\n");
    }
    prompt
}

fn render_metadata_chat_template(
    messages: &[ChatMessage],
    template: &ChatTemplate,
) -> Result<String, minijinja::Error> {
    let template_messages = messages
        .iter()
        .map(|message| ChatTemplateMessage {
            role: message.role.trim(),
            content: message.content.as_str(),
        })
        .collect::<Vec<_>>();
    let mut environment = Environment::new();
    environment.set_undefined_behavior(UndefinedBehavior::Strict);
    environment.add_function(
        "raise_exception",
        |message: String| -> Result<String, minijinja::Error> {
            Err(minijinja::Error::new(
                MiniJinjaErrorKind::InvalidOperation,
                message,
            ))
        },
    );
    environment.add_template(CHAT_TEMPLATE_NAME, &template.source)?;
    environment
        .get_template(CHAT_TEMPLATE_NAME)?
        .render(context! {
            messages => template_messages,
            bos_token => template.bos_token.as_str(),
            eos_token => template.eos_token.as_str(),
            eot_token => template.eot_token.as_str(),
            eom_token => template.eom_token.as_str(),
            unk_token => template.unk_token.as_str(),
            add_generation_prompt => true,
            tools => Option::<Vec<serde_json::Value>>::None,
            custom_tools => Option::<Vec<serde_json::Value>>::None,
        })
}

enum GenerationInput {
    RawCompletion(String),
    RenderedChat {
        prompt: String,
        add_special: bool,
        parse_special: bool,
    },
}

async fn run_generation(
    state: &ApiState,
    input: GenerationInput,
    max_new_tokens: usize,
) -> Result<Completion, Response> {
    let backend = Arc::clone(&state.backend);
    let result = state
        .generation_worker
        .run(move || match input {
            GenerationInput::RawCompletion(prompt) => backend.complete(&prompt, max_new_tokens),
            GenerationInput::RenderedChat {
                prompt,
                add_special,
                parse_special,
            } => backend.complete_prompt(&prompt, add_special, parse_special, max_new_tokens),
        })
        .await;

    match result {
        Ok(Ok(completion)) => Ok(completion),
        Ok(Err(error)) => Err(engine_error(error)),
        Err(error) => Err(engine_post_error(error)),
    }
}

fn engine_error_parts(error: EngineError) -> (&'static str, String) {
    match error {
        EngineError::ShapeMismatch(message) => ("context_length_exceeded", message),
        other => ("generation_failed", other.to_string()),
    }
}

fn engine_post_error(error: PostError) -> Response {
    let (code, message) = match error {
        PostError::Full => (
            "engine_queue_full",
            "the deterministic generation queue is full",
        ),
        PostError::Unavailable => (
            "generation_worker_unavailable",
            "the deterministic generation worker is unavailable",
        ),
    };
    let mut response = api_error(
        StatusCode::SERVICE_UNAVAILABLE,
        code,
        message.to_string(),
        None,
    );
    response
        .headers_mut()
        .insert("retry-after", HeaderValue::from_static("1"));
    response
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

fn next_chat_completion_id() -> String {
    static NEXT_ID: AtomicU64 = AtomicU64::new(1);
    format!(
        "chatcmpl-in-tree-{}",
        NEXT_ID.fetch_add(1, Ordering::Relaxed)
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::{to_bytes, Body};
    use axum::http::header::CONTENT_TYPE;
    use axum::http::Request;
    use serde_json::{json, Value};
    use std::sync::atomic::{AtomicBool, AtomicUsize};
    use std::sync::{Barrier, Mutex};
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

        fn complete_incremental(
            &self,
            prompt: &str,
            max_new_tokens: usize,
            on_token: &mut dyn FnMut(TokenDelta) -> GenerationControl,
        ) -> EngineResult<IncrementalGeneration> {
            let completion = self.complete(prompt, max_new_tokens)?;
            for (index, token_id) in completion.generated_tokens.iter().copied().enumerate() {
                let text = if completion.generated_tokens.len() == 1 {
                    completion.text.clone()
                } else {
                    completion
                        .text
                        .chars()
                        .nth(index)
                        .map(String::from)
                        .unwrap_or_default()
                };
                if on_token(TokenDelta { token_id, text }) == GenerationControl::Cancel {
                    return Ok(IncrementalGeneration::Cancelled);
                }
            }
            Ok(IncrementalGeneration::Completed(completion))
        }

        fn chat_template(&self) -> Option<ChatTemplate> {
            Some(ChatTemplate {
                source: concat!(
                    "{% for message in messages %}",
                    "{% if loop.first %}{{ bos_token }}{% endif %}",
                    "<|{{ message.role }}|>{{ message.content }}{{ eos_token }}",
                    "{% endfor %}",
                    "{% if add_generation_prompt %}<|assistant|>{% endif %}"
                )
                .to_string(),
                bos_token: "<s>".to_string(),
                eos_token: "</s>".to_string(),
                eot_token: String::new(),
                eom_token: String::new(),
                unk_token: "<unk>".to_string(),
            })
        }

        fn complete_prompt(
            &self,
            prompt: &str,
            _add_special: bool,
            _parse_special: bool,
            max_new_tokens: usize,
        ) -> EngineResult<Completion> {
            self.complete(prompt, max_new_tokens)
        }

        fn complete_prompt_incremental(
            &self,
            prompt: &str,
            _add_special: bool,
            _parse_special: bool,
            max_new_tokens: usize,
            on_token: &mut dyn FnMut(TokenDelta) -> GenerationControl,
        ) -> EngineResult<IncrementalGeneration> {
            self.complete_incremental(prompt, max_new_tokens, on_token)
        }
    }

    struct BlockingBackend {
        descriptor: ModelDescriptor,
        started: Arc<Barrier>,
        release: Arc<Barrier>,
        block_first: Arc<AtomicBool>,
    }

    struct TemplateBackend {
        descriptor: ModelDescriptor,
        template: Option<ChatTemplate>,
    }

    impl CompletionBackend for TemplateBackend {
        fn descriptor(&self) -> &ModelDescriptor {
            &self.descriptor
        }

        fn complete(&self, _prompt: &str, _max_new_tokens: usize) -> EngineResult<Completion> {
            unreachable!("template refusal tests must stop before generation")
        }

        fn complete_incremental(
            &self,
            _prompt: &str,
            _max_new_tokens: usize,
            _on_token: &mut dyn FnMut(TokenDelta) -> GenerationControl,
        ) -> EngineResult<IncrementalGeneration> {
            unreachable!("template refusal tests must stop before generation")
        }

        fn chat_template(&self) -> Option<ChatTemplate> {
            self.template.clone()
        }

        fn complete_prompt(
            &self,
            _prompt: &str,
            _add_special: bool,
            _parse_special: bool,
            _max_new_tokens: usize,
        ) -> EngineResult<Completion> {
            unreachable!("template refusal tests must stop before generation")
        }

        fn complete_prompt_incremental(
            &self,
            _prompt: &str,
            _add_special: bool,
            _parse_special: bool,
            _max_new_tokens: usize,
            _on_token: &mut dyn FnMut(TokenDelta) -> GenerationControl,
        ) -> EngineResult<IncrementalGeneration> {
            unreachable!("template refusal tests must stop before generation")
        }
    }

    impl CompletionBackend for BlockingBackend {
        fn descriptor(&self) -> &ModelDescriptor {
            &self.descriptor
        }

        fn complete(&self, _prompt: &str, _max_new_tokens: usize) -> EngineResult<Completion> {
            if self.block_first.swap(false, Ordering::SeqCst) {
                self.started.wait();
                self.release.wait();
            }
            Ok(Completion {
                prompt_tokens: vec![10],
                generated_tokens: vec![20],
                text: "x".to_string(),
                finish_reason: FinishReason::Length,
            })
        }

        fn complete_incremental(
            &self,
            prompt: &str,
            max_new_tokens: usize,
            on_token: &mut dyn FnMut(TokenDelta) -> GenerationControl,
        ) -> EngineResult<IncrementalGeneration> {
            let completion = self.complete(prompt, max_new_tokens)?;
            if on_token(TokenDelta {
                token_id: 20,
                text: "x".to_string(),
            }) == GenerationControl::Cancel
            {
                return Ok(IncrementalGeneration::Cancelled);
            }
            Ok(IncrementalGeneration::Completed(completion))
        }

        fn chat_template(&self) -> Option<ChatTemplate> {
            None
        }

        fn complete_prompt(
            &self,
            prompt: &str,
            _add_special: bool,
            _parse_special: bool,
            max_new_tokens: usize,
        ) -> EngineResult<Completion> {
            self.complete(prompt, max_new_tokens)
        }

        fn complete_prompt_incremental(
            &self,
            prompt: &str,
            _add_special: bool,
            _parse_special: bool,
            max_new_tokens: usize,
            on_token: &mut dyn FnMut(TokenDelta) -> GenerationControl,
        ) -> EngineResult<IncrementalGeneration> {
            self.complete_incremental(prompt, max_new_tokens, on_token)
        }
    }

    struct StreamingBackend {
        descriptor: ModelDescriptor,
        produced: Arc<AtomicUsize>,
    }

    impl CompletionBackend for StreamingBackend {
        fn descriptor(&self) -> &ModelDescriptor {
            &self.descriptor
        }

        fn complete(&self, _prompt: &str, _max_new_tokens: usize) -> EngineResult<Completion> {
            unreachable!("the streaming test must use complete_incremental")
        }

        fn complete_incremental(
            &self,
            _prompt: &str,
            max_new_tokens: usize,
            on_token: &mut dyn FnMut(TokenDelta) -> GenerationControl,
        ) -> EngineResult<IncrementalGeneration> {
            let mut generated_count = 0;
            for _ in 0..max_new_tokens {
                self.produced.fetch_add(1, Ordering::SeqCst);
                generated_count += 1;
                if on_token(TokenDelta {
                    token_id: 20,
                    text: "x".to_string(),
                }) == GenerationControl::Cancel
                {
                    return Ok(IncrementalGeneration::Cancelled);
                }
            }
            Ok(IncrementalGeneration::Completed(Completion {
                prompt_tokens: vec![10],
                generated_tokens: vec![20; generated_count],
                text: "x".repeat(generated_count),
                finish_reason: FinishReason::Length,
            }))
        }

        fn chat_template(&self) -> Option<ChatTemplate> {
            None
        }

        fn complete_prompt(
            &self,
            _prompt: &str,
            _add_special: bool,
            _parse_special: bool,
            _max_new_tokens: usize,
        ) -> EngineResult<Completion> {
            unreachable!("the streaming test uses raw completion")
        }

        fn complete_prompt_incremental(
            &self,
            _prompt: &str,
            _add_special: bool,
            _parse_special: bool,
            _max_new_tokens: usize,
            _on_token: &mut dyn FnMut(TokenDelta) -> GenerationControl,
        ) -> EngineResult<IncrementalGeneration> {
            unreachable!("the streaming test uses raw completion")
        }
    }

    fn app() -> Router {
        router(Arc::new(FakeBackend {
            descriptor: test_descriptor(),
        }))
    }

    fn test_descriptor() -> ModelDescriptor {
        ModelDescriptor {
            id: "test-model".to_string(),
            architecture: "llama".to_string(),
            context_length: 8,
            embedding_length: 2,
            vocab_size: 32,
            file_type: Some(7),
            size_bytes: Some(1024),
        }
    }

    async fn send(request: Request<Body>) -> (StatusCode, axum::http::HeaderMap, Value) {
        let response = app().oneshot(request).await.unwrap();
        let status = response.status();
        let headers = response.headers().clone();
        let bytes = to_bytes(response.into_body(), 1024 * 1024).await.unwrap();
        let body = serde_json::from_slice(&bytes).unwrap_or(Value::Null);
        (status, headers, body)
    }

    async fn health_depth(app: Router) -> usize {
        let response = app
            .oneshot(Request::get("/v1/health").body(Body::empty()).unwrap())
            .await
            .unwrap();
        let body = to_bytes(response.into_body(), 1024 * 1024).await.unwrap();
        serde_json::from_slice::<Value>(&body).unwrap()["engine_queue_depth"]
            .as_u64()
            .unwrap() as usize
    }

    fn post(body: Value) -> Request<Body> {
        Request::post("/v1/completions")
            .header(CONTENT_TYPE, "application/json")
            .body(Body::from(body.to_string()))
            .unwrap()
    }

    fn post_chat(body: Value) -> Request<Body> {
        Request::post("/v1/chat/completions")
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
    async fn chat_completion_uses_the_embedded_template_and_openai_shape() {
        struct RecordingBackend {
            descriptor: ModelDescriptor,
            prompt: Arc<Mutex<Option<(String, bool, bool)>>>,
        }

        impl CompletionBackend for RecordingBackend {
            fn descriptor(&self) -> &ModelDescriptor {
                &self.descriptor
            }

            fn complete(&self, _prompt: &str, _max_new_tokens: usize) -> EngineResult<Completion> {
                unreachable!("the chat route must use complete_prompt")
            }

            fn complete_incremental(
                &self,
                _prompt: &str,
                _max_new_tokens: usize,
                _on_token: &mut dyn FnMut(TokenDelta) -> GenerationControl,
            ) -> EngineResult<IncrementalGeneration> {
                unreachable!("the chat route must use complete_prompt_incremental")
            }

            fn chat_template(&self) -> Option<ChatTemplate> {
                FakeBackend {
                    descriptor: self.descriptor.clone(),
                }
                .chat_template()
            }

            fn complete_prompt(
                &self,
                prompt: &str,
                add_special: bool,
                parse_special: bool,
                _max_new_tokens: usize,
            ) -> EngineResult<Completion> {
                *self.prompt.lock().unwrap() =
                    Some((prompt.to_string(), add_special, parse_special));
                Ok(Completion {
                    prompt_tokens: vec![1, 2, 3],
                    generated_tokens: vec![4],
                    text: "answer".to_string(),
                    finish_reason: FinishReason::EndOfGeneration,
                })
            }

            fn complete_prompt_incremental(
                &self,
                prompt: &str,
                add_special: bool,
                parse_special: bool,
                max_new_tokens: usize,
                on_token: &mut dyn FnMut(TokenDelta) -> GenerationControl,
            ) -> EngineResult<IncrementalGeneration> {
                let completion = self.complete_prompt(
                    prompt,
                    add_special,
                    parse_special,
                    max_new_tokens,
                )?;
                if on_token(TokenDelta {
                    token_id: 4,
                    text: completion.text.clone(),
                }) == GenerationControl::Cancel
                {
                    return Ok(IncrementalGeneration::Cancelled);
                }
                Ok(IncrementalGeneration::Completed(completion))
            }
        }

        let captured = Arc::new(Mutex::new(None));
        let backend = Arc::new(RecordingBackend {
            descriptor: ModelDescriptor {
                id: "test-model".to_string(),
                architecture: "llama".to_string(),
                context_length: 8,
                embedding_length: 2,
                vocab_size: 32,
                file_type: None,
                size_bytes: None,
            },
            prompt: Arc::clone(&captured),
        });
        let response = router(backend)
            .oneshot(post_chat(json!({
                "model": "test-model",
                "messages": [
                    {"role":"system","content":"rules"},
                    {"role":"user","content":"hello"}
                ],
                "temperature": 0,
                "max_tokens": 2
            })))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let bytes = to_bytes(response.into_body(), 1024 * 1024).await.unwrap();
        let body: Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(body["object"], "chat.completion");
        assert_eq!(body["model"], "test-model");
        assert_eq!(body["choices"][0]["message"]["role"], "assistant");
        assert_eq!(body["choices"][0]["message"]["content"], "answer");
        assert_eq!(body["choices"][0]["finish_reason"], "stop");
        assert_eq!(body["usage"]["prompt_tokens"], 3);
        assert_eq!(body["usage"]["completion_tokens"], 1);
        assert_eq!(body["usage"]["total_tokens"], 4);
        assert!(body["id"]
            .as_str()
            .unwrap()
            .starts_with("chatcmpl-in-tree-"));

        assert_eq!(
            captured.lock().unwrap().as_ref(),
            Some(&(
                "<s><|system|>rules</s><|user|>hello</s><|assistant|>".to_string(),
                false,
                true,
            ))
        );
    }

    #[test]
    fn llama3_chat_uses_the_pinned_compact_prompt_shape() {
        let template = ChatTemplate {
            source: concat!(
                "{{ bos_token }}",
                "<|start_header_id|>system<|end_header_id|>",
                "default system preamble<|eot_id|>"
            )
            .to_string(),
            bos_token: "<|begin_of_text|>".to_string(),
            eos_token: "<|end_of_text|>".to_string(),
            eot_token: "<|eot_id|>".to_string(),
            eom_token: "<|eom_id|>".to_string(),
            unk_token: String::new(),
        };
        let messages = vec![ChatMessage {
            role: "user".to_string(),
            content: "Reply with one word: blue.".to_string(),
        }];

        let rendered = render_chat_prompt(&messages, &template, "Llama-3.2-1B-Q4_K_M").unwrap();

        assert_eq!(
            rendered.text,
            concat!(
                "<|start_header_id|>user<|end_header_id|>\n\n",
                "Reply with one word: blue.<|eot_id|>",
                "<|start_header_id|>assistant<|end_header_id|>\n\n"
            )
        );
        assert!(rendered.add_special);
        assert!(rendered.parse_special);
        assert!(!rendered.text.contains("default system preamble"));

        let exact_q8 =
            render_chat_prompt(&messages, &template, "Llama-3.2-1B-Instruct-Q8_0").unwrap();
        assert_eq!(
            exact_q8.text,
            concat!(
                "<|begin_of_text|>",
                "<|start_header_id|>system<|end_header_id|>",
                "default system preamble<|eot_id|>"
            )
        );
        assert!(!exact_q8.add_special);
        assert!(exact_q8.parse_special);
    }

    #[tokio::test]
    async fn unsupported_chat_shapes_fail_closed() {
        let cases = [
            (
                json!({"messages":[{"role":"user","content":"hi"}],"temperature":0.5}),
                StatusCode::BAD_REQUEST,
                "unsupported_sampling",
            ),
            (
                json!({"messages":[{"role":"tool","content":"hi"}]}),
                StatusCode::BAD_REQUEST,
                "unsupported_message_role",
            ),
            (
                json!({"messages":[{"role":"user","content":"hi"}],"tools":[]}),
                StatusCode::BAD_REQUEST,
                "unsupported_parameter",
            ),
            (
                json!({"messages":[]}),
                StatusCode::BAD_REQUEST,
                "invalid_messages",
            ),
            (
                json!({"model":"other","messages":[{"role":"user","content":"hi"}]}),
                StatusCode::NOT_FOUND,
                "model_not_found",
            ),
        ];
        for (request, expected_status, expected_code) in cases {
            let (status, _, body) = send(post_chat(request)).await;
            assert_eq!(status, expected_status);
            assert_eq!(body["error"]["code"], expected_code);
        }
    }

    #[tokio::test]
    async fn missing_and_invalid_chat_templates_are_typed_refusals() {
        for template in [
            None,
            Some(ChatTemplate {
                source: "{{ undefined_template_value }}".to_string(),
                bos_token: String::new(),
                eos_token: String::new(),
                eot_token: String::new(),
                eom_token: String::new(),
                unk_token: String::new(),
            }),
        ] {
            let app = router(Arc::new(TemplateBackend {
                descriptor: test_descriptor(),
                template,
            }));
            let response = app
                .oneshot(post_chat(json!({
                    "messages":[{"role":"user","content":"hi"}]
                })))
                .await
                .unwrap();
            assert_eq!(response.status(), StatusCode::UNPROCESSABLE_ENTITY);
            let bytes = to_bytes(response.into_body(), 1024 * 1024).await.unwrap();
            let body: Value = serde_json::from_slice(&bytes).unwrap();
            assert_eq!(body["error"]["code"], "unsupported_chat_template");
            assert_eq!(body["error"]["param"], "messages");
        }
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

    fn data_events(body: &str) -> Vec<&str> {
        body.lines()
            .filter_map(|line| line.strip_prefix("data: "))
            .collect()
    }

    #[tokio::test]
    async fn raw_and_chat_streams_emit_openai_chunks_and_done() {
        let raw = app()
            .oneshot(post(json!({
                "model":"test-model",
                "prompt":"hi",
                "max_tokens":2,
                "stream":true
            })))
            .await
            .unwrap();
        assert_eq!(raw.status(), StatusCode::OK);
        assert!(raw.headers()[CONTENT_TYPE]
            .to_str()
            .unwrap()
            .starts_with("text/event-stream"));
        let bytes = to_bytes(raw.into_body(), 1024 * 1024).await.unwrap();
        let raw = String::from_utf8(bytes.to_vec()).unwrap();
        let events = data_events(&raw);
        assert_eq!(events.last(), Some(&"[DONE]"));
        let chunks = events[..events.len() - 1]
            .iter()
            .map(|event| serde_json::from_str::<Value>(event).unwrap())
            .collect::<Vec<_>>();
        assert_eq!(chunks.len(), 3);
        assert!(chunks
            .iter()
            .all(|chunk| chunk["object"] == "text_completion"));
        assert_eq!(chunks[0]["choices"][0]["text"], "x");
        assert_eq!(chunks[1]["choices"][0]["text"], "x");
        assert_eq!(chunks[2]["choices"][0]["text"], "");
        assert_eq!(chunks[2]["choices"][0]["finish_reason"], "length");

        let chat = app()
            .oneshot(post_chat(json!({
                "model":"test-model",
                "messages":[{"role":"user","content":"hi"}],
                "max_tokens":2,
                "stream":true
            })))
            .await
            .unwrap();
        assert_eq!(chat.status(), StatusCode::OK);
        assert!(chat.headers()[CONTENT_TYPE]
            .to_str()
            .unwrap()
            .starts_with("text/event-stream"));
        let bytes = to_bytes(chat.into_body(), 1024 * 1024).await.unwrap();
        let chat = String::from_utf8(bytes.to_vec()).unwrap();
        let events = data_events(&chat);
        assert_eq!(events.last(), Some(&"[DONE]"));
        let chunks = events[..events.len() - 1]
            .iter()
            .map(|event| serde_json::from_str::<Value>(event).unwrap())
            .collect::<Vec<_>>();
        assert_eq!(chunks.len(), 4);
        assert!(chunks
            .iter()
            .all(|chunk| chunk["object"] == "chat.completion.chunk"));
        assert_eq!(chunks[0]["choices"][0]["delta"]["role"], "assistant");
        assert_eq!(chunks[1]["choices"][0]["delta"]["content"], "x");
        assert_eq!(chunks[2]["choices"][0]["delta"]["content"], "x");
        assert_eq!(chunks[3]["choices"][0]["delta"], json!({}));
        assert_eq!(chunks[3]["choices"][0]["finish_reason"], "length");
    }

    #[tokio::test]
    async fn an_engine_error_after_sse_headers_is_a_typed_terminal_event() {
        let response = app()
            .oneshot(post(json!({
                "prompt":"overflow",
                "max_tokens":1,
                "stream":true
            })))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let bytes = to_bytes(response.into_body(), 1024 * 1024).await.unwrap();
        let body = String::from_utf8(bytes.to_vec()).unwrap();
        assert!(body.contains("event: error"));
        let events = data_events(&body);
        assert_eq!(events.last(), Some(&"[DONE]"));
        let error: Value = serde_json::from_str(events[0]).unwrap();
        assert_eq!(error["error"]["type"], "server_error");
        assert_eq!(error["error"]["code"], "context_length_exceeded");
    }

    #[tokio::test]
    async fn dropping_a_backpressured_stream_cancels_generation() {
        let produced = Arc::new(AtomicUsize::new(0));
        let app = router(Arc::new(StreamingBackend {
            descriptor: test_descriptor(),
            produced: Arc::clone(&produced),
        }));
        let response = app
            .clone()
            .oneshot(post(json!({"prompt":"hi","max_tokens":100,"stream":true})))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);

        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(2);
        while produced.load(Ordering::SeqCst) < 33 {
            assert!(
                std::time::Instant::now() < deadline,
                "producer never reached the bounded event queue"
            );
            tokio::time::sleep(std::time::Duration::from_millis(5)).await;
        }
        assert_eq!(produced.load(Ordering::SeqCst), 33);

        drop(response);
        while health_depth(app.clone()).await != 0 {
            assert!(
                std::time::Instant::now() < deadline,
                "generation worker did not recover after stream drop"
            );
            tokio::time::sleep(std::time::Duration::from_millis(5)).await;
        }
        assert!(produced.load(Ordering::SeqCst) < 100);
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
    async fn generation_queue_is_bounded_and_reports_its_depth() {
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
            block_first: Arc::new(AtomicBool::new(true)),
        }));

        let first_app = app.clone();
        let first = tokio::spawn(async move {
            first_app
                .oneshot(post(json!({"prompt":"first","max_tokens":1})))
                .await
                .unwrap()
        });
        started.wait();

        let mut queued = tokio::task::JoinSet::new();
        for index in 0..8 {
            let app = app.clone();
            queued.spawn(async move {
                app.oneshot(post(json!({
                    "prompt":format!("queued-{index}"),
                    "max_tokens":1
                })))
                .await
                .unwrap()
            });
        }
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(2);
        while health_depth(app.clone()).await != 9 {
            assert!(
                std::time::Instant::now() < deadline,
                "all eight waiting jobs were not admitted"
            );
            tokio::task::yield_now().await;
        }

        let busy = app
            .clone()
            .oneshot(post(json!({"prompt":"overflow","max_tokens":1})))
            .await
            .unwrap();
        assert_eq!(busy.status(), StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(busy.headers()["retry-after"], "1");
        let bytes = to_bytes(busy.into_body(), 1024 * 1024).await.unwrap();
        let body: Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(body["error"]["type"], "runtime_unavailable");
        assert_eq!(body["error"]["code"], "engine_queue_full");

        assert_eq!(health_depth(app.clone()).await, 9);

        release.wait();
        assert_eq!(first.await.unwrap().status(), StatusCode::OK);
        while let Some(response) = queued.join_next().await {
            assert_eq!(response.unwrap().status(), StatusCode::OK);
        }
        assert_eq!(health_depth(app).await, 0);
    }

    #[tokio::test]
    async fn private_and_unimplemented_routes_are_not_exposed() {
        for path in ["/api/models/load", "/v1/embeddings"] {
            let response = app()
                .oneshot(Request::post(path).body(Body::empty()).unwrap())
                .await
                .unwrap();
            assert_eq!(response.status(), StatusCode::NOT_FOUND, "{path}");
        }
    }
}
