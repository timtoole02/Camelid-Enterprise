//! Owned, platform-neutral model lifecycle for the in-tree engine.
//!
//! [`LoadedModel`] is the boundary between GGUF storage and serving code. It
//! performs the complete load pipeline once, owns the tokenizer/config/weights,
//! and creates an isolated decoder for each generation. HTTP schemas, queues,
//! and lane policy deliberately remain outside this crate.

use std::path::{Path, PathBuf};

use crate::forward::Decoder;
use crate::gguf::read_metadata;
use crate::model::{
    expand_fused_dense_tensors, LlamaModelConfig, LlamaTensorBinding, LlamaWeights,
};
use crate::tensor::{q8_0_dot_rows, Q8DotRows, TensorStore};
use crate::tokenizer::{TokenId, Tokenizer};
use crate::{EngineError, Result};

/// Why a greedy generation stopped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FinishReason {
    /// The model emitted one of the tokenizer's end-of-generation tokens.
    EndOfGeneration,
    /// The caller-provided maximum number of new tokens was reached.
    Length,
}

/// A raw-text completion produced by the in-tree engine.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Completion {
    /// Tokenized input, including tokenizer-configured BOS/EOS markers.
    pub prompt_tokens: Vec<TokenId>,
    /// Emitted ids. An end-of-generation id is retained when one caused the stop.
    pub generated_tokens: Vec<TokenId>,
    /// Generated ids decoded with special/control tokens removed.
    pub text: String,
    pub finish_reason: FinishReason,
}

/// One loaded dense-decoder model using the in-tree engine.
///
/// The model is immutable after construction. A fresh [`Decoder`] (and
/// therefore a fresh KV cache) is created for each call to [`Self::generate`],
/// which keeps request state out of the model lifecycle boundary.
pub struct LoadedModel {
    source: PathBuf,
    name: Option<String>,
    architecture: String,
    config: LlamaModelConfig,
    tokenizer: Tokenizer,
    weights: LlamaWeights,
    q8_dot: Q8DotRows,
}

impl LoadedModel {
    /// Load a model using the portable, order-stable Q8_0 dot implementation.
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        Self::load_with_q8_dot(path, q8_0_dot_rows)
    }

    /// Load a model with a caller-supplied Q8_0 kernel.
    ///
    /// Platform crates use this seam for an accelerated kernel proven
    /// bit-identical to the portable reference. All parsing, validation, and
    /// model ownership remain platform-neutral.
    pub fn load_with_q8_dot(path: impl AsRef<Path>, q8_dot: Q8DotRows) -> Result<Self> {
        let path = path.as_ref();
        let gguf = read_metadata(path)?;
        let name = gguf.model_name().map(str::to_owned);
        let architecture = gguf
            .architecture()
            .ok_or_else(|| {
                EngineError::InvalidModelMetadata(
                    "required metadata general.architecture is missing".to_string(),
                )
            })?
            .to_owned();
        let config = LlamaModelConfig::from_gguf(&gguf)?;
        let tokenizer = Tokenizer::from_gguf(&gguf)?;
        expand_fused_dense_tensors(&gguf, &config)?;
        let binding = LlamaTensorBinding::bind(&gguf, &config)?;
        let store = TensorStore::open(path, &gguf);
        let weights = LlamaWeights::load(&store, &binding)?;

        Self::from_components(
            path.to_path_buf(),
            name,
            architecture,
            config,
            tokenizer,
            weights,
            q8_dot,
        )
    }

    fn from_components(
        source: PathBuf,
        name: Option<String>,
        architecture: String,
        config: LlamaModelConfig,
        tokenizer: Tokenizer,
        weights: LlamaWeights,
        q8_dot: Q8DotRows,
    ) -> Result<Self> {
        weights.validate_dense_shapes(&config)?;
        let vocab_size = config.vocab_size.ok_or_else(|| {
            EngineError::InvalidModelMetadata(
                "vocabulary size is required by the loaded model runtime".to_string(),
            )
        })? as usize;
        if tokenizer.tokens.len() != vocab_size {
            return Err(EngineError::InvalidTokenizerMetadata(format!(
                "tokenizer vocabulary has {} entries but the model expects {vocab_size}",
                tokenizer.tokens.len()
            )));
        }
        Ok(Self {
            source,
            name,
            architecture,
            config,
            tokenizer,
            weights,
            q8_dot,
        })
    }

    pub fn source(&self) -> &Path {
        &self.source
    }

    pub fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }

    pub fn architecture(&self) -> &str {
        &self.architecture
    }

    pub fn config(&self) -> &LlamaModelConfig {
        &self.config
    }

    pub fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }

    /// Encode a raw completion prompt with the model's tokenizer policy.
    pub fn encode_completion(&self, prompt: &str) -> Result<Vec<TokenId>> {
        self.tokenizer.encode(prompt, true, false)
    }

    /// Generate from already-tokenized input.
    ///
    /// The prompt plus requested budget must fit the declared context window.
    /// This check happens before allocating a KV cache or beginning inference,
    /// so an oversized request fails closed without partial work.
    pub fn generate(&self, prompt_tokens: &[TokenId], max_new_tokens: usize) -> Result<Completion> {
        if prompt_tokens.is_empty() {
            return Err(EngineError::ShapeMismatch(
                "generation requires a non-empty prompt".to_string(),
            ));
        }
        let requested = prompt_tokens
            .len()
            .checked_add(max_new_tokens)
            .ok_or_else(|| {
                EngineError::ShapeMismatch(
                    "prompt and generation token counts overflow usize".to_string(),
                )
            })?;
        let context_length = self.config.context_length as usize;
        if requested > context_length {
            return Err(EngineError::ShapeMismatch(format!(
                "prompt tokens ({}) plus max new tokens ({max_new_tokens}) exceed context length {context_length}",
                prompt_tokens.len()
            )));
        }

        let mut decoder = Decoder::with_q8_dot(&self.config, &self.weights, self.q8_dot)?;
        let generated_tokens = decoder.generate_until(prompt_tokens, max_new_tokens, |token| {
            self.tokenizer.special.eog.contains(&token)
        })?;
        let finish_reason = if generated_tokens
            .last()
            .is_some_and(|token| self.tokenizer.special.eog.contains(token))
        {
            FinishReason::EndOfGeneration
        } else {
            FinishReason::Length
        };
        let text = self.tokenizer.decode(&generated_tokens, true)?;
        Ok(Completion {
            prompt_tokens: prompt_tokens.to_vec(),
            generated_tokens,
            text,
            finish_reason,
        })
    }

    /// Tokenize and greedily complete a raw-text prompt.
    pub fn complete(&self, prompt: &str, max_new_tokens: usize) -> Result<Completion> {
        let prompt_tokens = self.encode_completion(prompt)?;
        self.generate(&prompt_tokens, max_new_tokens)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::{BTreeSet, HashMap};

    use super::*;
    use crate::model::LlamaLayerWeights;
    use crate::tensor::CpuTensor;
    use crate::tokenizer::{
        BpePreTokenizer, BpeRegistry, SpecialTokens, Token, TokenKind, TokenizerConfig,
        TokenizerModel,
    };

    fn tensor(name: &str, dims: Vec<usize>, data: Vec<f32>) -> CpuTensor {
        CpuTensor::from_f32(name, dims, data).unwrap()
    }

    fn zero_matrix(name: &str) -> CpuTensor {
        tensor(name, vec![2, 2], vec![0.0; 4])
    }

    fn synthetic_model() -> LoadedModel {
        let config = LlamaModelConfig {
            context_length: 8,
            embedding_length: 2,
            block_count: 1,
            feed_forward_length: 2,
            attention_head_count: 1,
            attention_head_count_kv: 1,
            rope_dimension_count: Some(2),
            rope_freq_base: Some(10_000.0),
            rope_scaling_type: None,
            rope_scaling_factor: None,
            rope_scaling_original_context_length: None,
            rope_scaling_low_freq_factor: None,
            rope_scaling_high_freq_factor: None,
            rms_norm_epsilon: 1e-5,
            vocab_size: Some(2),
            file_type: Some(0),
            attention_key_length: None,
            rope_neox_pairing: false,
        };
        let layer = LlamaLayerWeights {
            attention_norm: tensor("blk.0.attn_norm.weight", vec![2], vec![1.0; 2]),
            attention_q: zero_matrix("blk.0.attn_q.weight"),
            attention_k: zero_matrix("blk.0.attn_k.weight"),
            attention_v: zero_matrix("blk.0.attn_v.weight"),
            attention_output: zero_matrix("blk.0.attn_output.weight"),
            attention_q_norm: None,
            attention_k_norm: None,
            ffn_norm: tensor("blk.0.ffn_norm.weight", vec![2], vec![1.0; 2]),
            ffn_gate: zero_matrix("blk.0.ffn_gate.weight"),
            ffn_up: zero_matrix("blk.0.ffn_up.weight"),
            ffn_down: zero_matrix("blk.0.ffn_down.weight"),
        };
        let weights = LlamaWeights {
            token_embedding: tensor("token_embd.weight", vec![2, 2], vec![0.0; 4]),
            output_norm: tensor("output_norm.weight", vec![2], vec![1.0; 2]),
            output: Some(zero_matrix("output.weight")),
            rope_freqs: None,
            layers: vec![layer],
        };
        let tokens = vec![
            Token {
                id: 0,
                text: "<eog>".to_string(),
                score: 0.0,
                kind: TokenKind::Control,
            },
            Token {
                id: 1,
                text: "▁a".to_string(),
                score: 0.0,
                kind: TokenKind::Normal,
            },
        ];
        let tokenizer = Tokenizer {
            model: TokenizerModel::LlamaSpm,
            bpe_pre_tokenizer: BpePreTokenizer::default(),
            token_to_id: HashMap::from([("<eog>".to_string(), 0), ("▁a".to_string(), 1)]),
            tokens,
            byte_token_to_id: HashMap::new(),
            bpe_ranks: HashMap::new(),
            bpe_registry: BpeRegistry::default(),
            special: SpecialTokens {
                eos: Some(0),
                eog: BTreeSet::from([0]),
                ..SpecialTokens::default()
            },
            config: TokenizerConfig {
                add_bos: false,
                add_eos: false,
                add_sep: false,
                add_space_prefix: true,
                remove_extra_whitespaces: false,
            },
            chat_template: None,
        };

        LoadedModel::from_components(
            PathBuf::from("synthetic.gguf"),
            Some("synthetic".to_string()),
            "llama".to_string(),
            config,
            tokenizer,
            weights,
            q8_0_dot_rows,
        )
        .unwrap()
    }

    #[test]
    fn completion_stops_on_eog_and_hides_the_control_token() {
        let completion = synthetic_model().complete("a", 4).unwrap();
        assert_eq!(completion.prompt_tokens, vec![1]);
        assert_eq!(completion.generated_tokens, vec![0]);
        assert_eq!(completion.text, "");
        assert_eq!(completion.finish_reason, FinishReason::EndOfGeneration);
    }

    #[test]
    fn generation_rejects_a_budget_beyond_the_context_window() {
        let error = synthetic_model().complete("a", 8).unwrap_err();
        assert!(error
            .to_string()
            .contains("prompt tokens (1) plus max new tokens (8) exceed context length 8"));
    }

    #[test]
    fn generation_with_zero_budget_finishes_by_length_without_decoding() {
        let completion = synthetic_model().complete("a", 0).unwrap();
        assert!(completion.generated_tokens.is_empty());
        assert_eq!(completion.text, "");
        assert_eq!(completion.finish_reason, FinishReason::Length);
    }
}
