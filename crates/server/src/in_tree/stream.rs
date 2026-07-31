//! OpenAI-shaped SSE transport over the synchronous incremental backend.

use std::convert::Infallible;
use std::sync::Arc;

use axum::response::sse::{Event, Sse};
use axum::response::{IntoResponse, Response};
use engine_core::runtime::{FinishReason, GenerationControl, IncrementalGeneration, TokenDelta};
use serde::Serialize;

use super::{
    engine_error_parts, engine_post_error, next_chat_completion_id, next_completion_id, ApiState,
    CompletionBackend, GenerationInput,
};

const EVENT_BUFFER: usize = 32;

#[derive(Serialize)]
struct ChatCompletionStreamChunk {
    id: String,
    object: &'static str,
    created: u64,
    model: String,
    choices: Vec<ChatCompletionStreamChoice>,
}

#[derive(Serialize)]
struct ChatCompletionStreamChoice {
    index: u32,
    delta: ChatCompletionDelta,
    finish_reason: Option<&'static str>,
}

#[derive(Serialize)]
struct ChatCompletionDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    role: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<String>,
}

#[derive(Serialize)]
struct CompletionStreamChunk {
    id: String,
    object: &'static str,
    created: u64,
    model: String,
    choices: Vec<CompletionStreamChoice>,
}

#[derive(Serialize)]
struct CompletionStreamChoice {
    index: u32,
    text: String,
    finish_reason: Option<&'static str>,
}

enum StreamGenerationEvent {
    Delta(String),
    Finished { finish_reason: &'static str },
    Failed { code: &'static str, message: String },
}

pub(super) fn stream_generation(
    state: &ApiState,
    input: GenerationInput,
    max_new_tokens: usize,
    model_id: String,
    chat: bool,
) -> Response {
    let stream_id = if chat {
        next_chat_completion_id()
    } else {
        next_completion_id()
    };
    let (event_sender, mut event_receiver) = tokio::sync::mpsc::channel(EVENT_BUFFER);
    let backend = Arc::clone(&state.backend);
    if let Err(error) = state.generation_worker.post(Box::new(move || {
        run_stream_generation(backend, input, max_new_tokens, event_sender);
    })) {
        return engine_post_error(error);
    }

    let events = async_stream::stream! {
        if chat {
            yield sse_json_event(&ChatCompletionStreamChunk {
                id: stream_id.clone(),
                object: "chat.completion.chunk",
                created: 0,
                model: model_id.clone(),
                choices: vec![ChatCompletionStreamChoice {
                    index: 0,
                    delta: ChatCompletionDelta {
                        role: Some("assistant"),
                        content: None,
                    },
                    finish_reason: None,
                }],
            });
        }

        while let Some(event) = event_receiver.recv().await {
            match event {
                StreamGenerationEvent::Delta(delta) if chat => {
                    yield sse_json_event(&ChatCompletionStreamChunk {
                        id: stream_id.clone(),
                        object: "chat.completion.chunk",
                        created: 0,
                        model: model_id.clone(),
                        choices: vec![ChatCompletionStreamChoice {
                            index: 0,
                            delta: ChatCompletionDelta {
                                role: None,
                                content: Some(delta),
                            },
                            finish_reason: None,
                        }],
                    });
                }
                StreamGenerationEvent::Delta(delta) => {
                    yield sse_json_event(&CompletionStreamChunk {
                        id: stream_id.clone(),
                        object: "text_completion",
                        created: 0,
                        model: model_id.clone(),
                        choices: vec![CompletionStreamChoice {
                            index: 0,
                            text: delta,
                            finish_reason: None,
                        }],
                    });
                }
                StreamGenerationEvent::Finished { finish_reason } => {
                    if chat {
                        yield sse_json_event(&ChatCompletionStreamChunk {
                            id: stream_id.clone(),
                            object: "chat.completion.chunk",
                            created: 0,
                            model: model_id.clone(),
                            choices: vec![ChatCompletionStreamChoice {
                                index: 0,
                                delta: ChatCompletionDelta {
                                    role: None,
                                    content: None,
                                },
                                finish_reason: Some(finish_reason),
                            }],
                        });
                    } else {
                        yield sse_json_event(&CompletionStreamChunk {
                            id: stream_id.clone(),
                            object: "text_completion",
                            created: 0,
                            model: model_id.clone(),
                            choices: vec![CompletionStreamChoice {
                                index: 0,
                                text: String::new(),
                                finish_reason: Some(finish_reason),
                            }],
                        });
                    }
                    yield Ok(Event::default().data("[DONE]"));
                    return;
                }
                StreamGenerationEvent::Failed { code, message } => {
                    yield stream_error_event(code, message);
                    yield Ok(Event::default().data("[DONE]"));
                    return;
                }
            }
        }

        yield stream_error_event(
            "generation_worker_failed",
            "generation worker ended before completing the stream".to_string(),
        );
        yield Ok(Event::default().data("[DONE]"));
    };

    Sse::new(events).into_response()
}

fn run_stream_generation(
    backend: Arc<dyn CompletionBackend>,
    input: GenerationInput,
    max_new_tokens: usize,
    event_sender: tokio::sync::mpsc::Sender<StreamGenerationEvent>,
) {
    if event_sender.is_closed() {
        return;
    }
    let mut on_token = |delta: TokenDelta| {
        if event_sender.is_closed() {
            return GenerationControl::Cancel;
        }
        if delta.text.is_empty() {
            return GenerationControl::Continue;
        }
        if event_sender
            .blocking_send(StreamGenerationEvent::Delta(delta.text))
            .is_ok()
        {
            GenerationControl::Continue
        } else {
            GenerationControl::Cancel
        }
    };
    let result = match input {
        GenerationInput::RawCompletion(prompt) => {
            backend.complete_incremental(&prompt, max_new_tokens, &mut on_token)
        }
        GenerationInput::RenderedChat {
            prompt,
            add_special,
            parse_special,
        } => backend.complete_prompt_incremental(
            &prompt,
            add_special,
            parse_special,
            max_new_tokens,
            &mut on_token,
        ),
    };

    match result {
        Ok(IncrementalGeneration::Completed(completion)) => {
            let finish_reason = match completion.finish_reason {
                FinishReason::EndOfGeneration => "stop",
                FinishReason::Length => "length",
            };
            let _ = event_sender.blocking_send(StreamGenerationEvent::Finished { finish_reason });
        }
        Ok(IncrementalGeneration::Cancelled) => {}
        Err(error) => {
            let (code, message) = engine_error_parts(error);
            let _ = event_sender.blocking_send(StreamGenerationEvent::Failed { code, message });
        }
    }
}

fn sse_json_event(value: &impl Serialize) -> Result<Event, Infallible> {
    Ok(Event::default()
        .data(serde_json::to_string(value).expect("stream response structs serialize")))
}

fn stream_error_event(code: &'static str, message: String) -> Result<Event, Infallible> {
    Ok(Event::default().event("error").data(
        serde_json::json!({
            "error": {
                "type": "server_error",
                "code": code,
                "message": message,
                "param": null
            }
        })
        .to_string(),
    ))
}
