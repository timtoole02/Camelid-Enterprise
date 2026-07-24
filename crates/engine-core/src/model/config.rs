//! Model configuration read from GGUF metadata.
//!
//! [`LlamaModelConfig`] is the flat, architecture-agnostic scalar config the
//! dense decoder forward pass reads. It is built entirely from the file's
//! `<arch>.*` metadata keys; nothing is inferred from tensor data except the
//! vocabulary size, which falls back to the token-embedding shape when the
//! metadata omits it. [`DenseLlamaDims`] derives the integer geometry the
//! shape validators and the forward pass consume.

use serde::Serialize;

use crate::gguf::GgufFile;
use crate::{EngineError, Result};

/// Flat scalar configuration for a dense decoder, produced from GGUF metadata.
///
/// This is the single source of truth the forward pass reads. It carries only
/// the scalars the dense Llama-family spine needs. Architectures with their own
/// layer plans or metadata shapes — mixture-of-experts routing, per-layer
/// dimension arrays, hybrid linear attention — are not configured here; a model
/// that needs them fails closed rather than being partially configured. The
/// engine builds the dense Llama, Mistral, Qwen2, and Qwen3 spine.
#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct LlamaModelConfig {
    pub context_length: u32,
    pub embedding_length: u32,
    pub block_count: u32,
    pub feed_forward_length: u32,
    pub attention_head_count: u32,
    pub attention_head_count_kv: u32,
    pub rope_dimension_count: Option<u32>,
    pub rope_freq_base: Option<f32>,
    pub rope_scaling_type: Option<String>,
    pub rope_scaling_factor: Option<f32>,
    pub rope_scaling_original_context_length: Option<u32>,
    pub rope_scaling_low_freq_factor: Option<f32>,
    pub rope_scaling_high_freq_factor: Option<f32>,
    pub rms_norm_epsilon: f32,
    pub vocab_size: Option<u32>,
    pub file_type: Option<u32>,
    /// Explicit per-head dimension from `<arch>.attention.key_length`, when the
    /// GGUF sets it. Qwen3 sizes where `head_dim != embedding_length/head_count`
    /// rely on this; `None` falls back to `embedding_length / head_count`.
    pub attention_key_length: Option<u32>,
    /// RoPE uses NEOX "split-half" pairing (dim `d` rotated with `d + rope_dim/2`)
    /// rather than the default "adjacent even/odd" pairing. Conversions that do
    /// not permute the Q/K projection weights (qwen3/qwen35/phi3) require this;
    /// permuted LLaMA-family conversions use adjacent even/odd.
    pub rope_neox_pairing: bool,
}

/// Whether `architecture` is one of the dense-decoder families this engine
/// attempts. This mirrors the accepted set in [`LlamaModelConfig::from_gguf`].
/// It is a pure classification check; an implemented architecture is only
/// attemptable, never automatically supported.
pub fn is_implemented_architecture(architecture: &str) -> bool {
    // The dense Llama-family spine. Architectures with their own layer plans or
    // metadata shapes (MoE routing, Gemma per-layer arrays, hybrid linear
    // attention, fused-QKV splitting) are not built here and fail closed rather
    // than binding a divergent partial config.
    matches!(architecture, "llama" | "mistral" | "qwen2" | "qwen3")
}

impl LlamaModelConfig {
    pub fn from_gguf(gguf: &GgufFile) -> Result<Self> {
        let architecture = match gguf.architecture() {
            Some(architecture @ ("llama" | "mistral" | "qwen2" | "qwen3")) => architecture,
            // Block-diffusion Gemma spellings are not autoregressive decoders;
            // this runtime has no diffusion decode loop, so fail closed rather
            // than mis-binding the shared backbone tensors.
            Some(other) if other.to_ascii_lowercase().contains("diffusion") => {
                return Err(EngineError::UnsupportedModelArchitecture(format!(
                    "{other}: block-diffusion architecture is not an autoregressive decoder"
                )))
            }
            Some(other) => {
                return Err(EngineError::UnsupportedModelArchitecture(other.to_string()))
            }
            None => {
                return Err(EngineError::InvalidModelMetadata(
                    "required metadata general.architecture is missing".to_string(),
                ))
            }
        };

        let attention_head_count =
            required_u32(gguf, &architecture_key(architecture, "attention.head_count"))?;
        let attention_head_count_kv =
            llama_attention_head_count_kv(gguf, architecture, attention_head_count);
        Ok(Self {
            context_length: required_u32(gguf, &architecture_key(architecture, "context_length"))?,
            embedding_length: required_u32(
                gguf,
                &architecture_key(architecture, "embedding_length"),
            )?,
            block_count: required_u32(gguf, &architecture_key(architecture, "block_count"))?,
            feed_forward_length: required_u32(
                gguf,
                &architecture_key(architecture, "feed_forward_length"),
            )?,
            attention_head_count,
            attention_head_count_kv,
            rope_dimension_count: gguf
                .metadata_u32(&architecture_key(architecture, "rope.dimension_count")),
            rope_freq_base: gguf.metadata_f32(&architecture_key(architecture, "rope.freq_base")),
            rope_scaling_type: gguf
                .metadata_string(&architecture_key(architecture, "rope.scaling.type"))
                .map(str::to_string),
            rope_scaling_factor: gguf
                .metadata_f32(&architecture_key(architecture, "rope.scaling.factor")),
            rope_scaling_original_context_length: gguf.metadata_u32(&architecture_key(
                architecture,
                "rope.scaling.original_context_length",
            )),
            rope_scaling_low_freq_factor: gguf.metadata_f32(&architecture_key(
                architecture,
                "rope.scaling.low_freq_factor",
            )),
            rope_scaling_high_freq_factor: gguf.metadata_f32(&architecture_key(
                architecture,
                "rope.scaling.high_freq_factor",
            )),
            // The rms epsilon defaults to 1e-5 only when the key is absent; the
            // forward path takes eps from here, never a hardcoded constant.
            rms_norm_epsilon: gguf
                .metadata_f32(&architecture_key(
                    architecture,
                    "attention.layer_norm_rms_epsilon",
                ))
                .unwrap_or(1e-5),
            vocab_size: gguf
                .metadata_u32(&architecture_key(architecture, "vocab_size"))
                .or_else(|| {
                    infer_vocab_size_from_token_embedding(
                        gguf,
                        "token_embd.weight",
                        required_u32(gguf, &architecture_key(architecture, "embedding_length"))
                            .ok()?,
                    )
                }),
            file_type: gguf.metadata_u32("general.file_type"),
            attention_key_length: gguf
                .metadata_u32(&architecture_key(architecture, "attention.key_length")),
            rope_neox_pairing: arch_uses_neox_rope_pairing(architecture),
        })
    }
}

/// NEOX split-half RoPE pairing for the unpermuted conversions (qwen3/qwen35/
/// phi3); everything else keeps adjacent even/odd (permuted LLaMA-style
/// conversions).
fn arch_uses_neox_rope_pairing(architecture: &str) -> bool {
    matches!(architecture, "qwen3" | "qwen35" | "phi3")
}

pub(crate) fn architecture_key(architecture: &str, suffix: &str) -> String {
    format!("{architecture}.{suffix}")
}

pub(crate) fn required_u32(gguf: &GgufFile, key: &str) -> Result<u32> {
    gguf.metadata_u32(key).ok_or_else(|| {
        EngineError::InvalidModelMetadata(format!("required metadata {key} is missing or not u32"))
    })
}

fn llama_attention_head_count_kv(
    gguf: &GgufFile,
    architecture: &str,
    attention_head_count: u32,
) -> u32 {
    gguf.metadata_u32(&architecture_key(architecture, "attention.head_count_kv"))
        .unwrap_or(attention_head_count)
}

/// Infer the vocabulary size from the two-dimensional token-embedding tensor
/// when the metadata omits `<arch>.vocab_size`: the vocab is whichever
/// dimension is not the embedding width.
fn infer_vocab_size_from_token_embedding(
    gguf: &GgufFile,
    tensor_name: &str,
    embedding_length: u32,
) -> Option<u32> {
    let embedding_length = u64::from(embedding_length);
    let tensor = gguf.tensors.iter().find(|t| t.name == tensor_name)?;
    if tensor.dimensions.len() != 2 {
        return None;
    }
    let dims = tensor.dimensions.as_slice();
    let inferred = if dims[0] == embedding_length {
        dims[1]
    } else if dims[1] == embedding_length {
        dims[0]
    } else {
        return None;
    };
    inferred.try_into().ok()
}

/// Derived integer dimensions for a dense decoder, used by both descriptor-
/// and runtime-level shape validation and by the forward pass.
#[derive(Debug, Clone, Copy)]
pub struct DenseLlamaDims {
    pub embedding_length: usize,
    pub block_count: usize,
    pub feed_forward_length: usize,
    pub attention_head_count_kv: usize,
    pub head_dim: usize,
    /// Query projection width = `attention_head_count * head_dim`. Equals
    /// `embedding_length` only when `head_dim == embedding_length/head_count`;
    /// an explicit larger head_dim makes `q_width > embedding_length`.
    pub q_width: usize,
    pub kv_width: usize,
    pub vocab_size: usize,
}

impl DenseLlamaDims {
    pub fn from_config(config: &LlamaModelConfig) -> Result<Self> {
        let embedding_length = config.embedding_length as usize;
        let attention_head_count = config.attention_head_count as usize;
        if attention_head_count == 0 || !embedding_length.is_multiple_of(attention_head_count) {
            return Err(EngineError::InvalidModelMetadata(format!(
                "embedding length {embedding_length} is not divisible by attention head count {attention_head_count}"
            )));
        }

        let attention_head_count_kv = config.attention_head_count_kv as usize;
        if attention_head_count_kv == 0 {
            return Err(EngineError::InvalidModelMetadata(
                "attention kv head count must be greater than zero".to_string(),
            ));
        }
        if !attention_head_count.is_multiple_of(attention_head_count_kv) {
            return Err(EngineError::InvalidModelMetadata(format!(
                "attention head count {attention_head_count} must be a multiple of kv head count {attention_head_count_kv}"
            )));
        }

        let vocab_size = config.vocab_size.ok_or_else(|| {
            EngineError::InvalidModelMetadata(
                "required vocabulary size is missing for dense tensor validation".to_string(),
            )
        })? as usize;
        if vocab_size == 0 {
            return Err(EngineError::InvalidModelMetadata(
                "vocabulary size must be greater than zero".to_string(),
            ));
        }

        // Prefer the GGUF's explicit per-head dim when present; fall back to
        // embedding/head_count for rows that do not carry it.
        let head_dim = match config.attention_key_length {
            Some(key_length) if key_length > 0 => key_length as usize,
            _ => embedding_length / attention_head_count,
        };
        Ok(Self {
            embedding_length,
            block_count: config.block_count as usize,
            feed_forward_length: config.feed_forward_length as usize,
            attention_head_count_kv,
            head_dim,
            q_width: attention_head_count * head_dim,
            kv_width: attention_head_count_kv * head_dim,
            vocab_size,
        })
    }
}
