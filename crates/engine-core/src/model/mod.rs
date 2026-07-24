//! Model configuration and weight binding for the dense decoder spine.
//!
//! The load pipeline is a fixed sequence that sits entirely above the GGUF
//! reader and tensor store and performs no tensor arithmetic:
//!
//! 1. [`LlamaModelConfig::from_gguf`] reads the `<arch>.*` metadata into a flat
//!    scalar config;
//! 2. [`expand_fused_dense_tensors`] rejects fused projection layouts (a no-op
//!    for already-split models) and must run before binding;
//! 3. [`LlamaTensorBinding::bind`] resolves every tensor by name and validates
//!    its descriptor shape;
//! 4. [`LlamaWeights::load`] materializes the descriptors into `CpuTensor`s,
//!    with 2-D Q8_0 linears held as plain blocks.
//!
//! Scope is the dense Llama/Mistral/Qwen2/Qwen3/phi3 family. Mixture-of-experts,
//! Gemma 4, and Qwen3.5 lanes are excluded; a model that needs them fails
//! closed during config or binding rather than being partially built.

pub mod binding;
pub mod config;
pub mod weights;

pub use binding::{
    expand_fused_dense_tensors, LlamaFfnTensors, LlamaLayerTensors, LlamaTensorBinding,
};
pub use config::{is_implemented_architecture, DenseLlamaDims, LlamaModelConfig};
pub use weights::{LlamaLayerWeights, LlamaWeights};

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::path::{Path, PathBuf};

    use super::*;
    use crate::gguf::{read_metadata, GgufFile, MetadataValue, TensorDescriptor, TensorType};
    use crate::tensor::TensorStore;

    fn u32_meta(value: u32) -> MetadataValue {
        MetadataValue::U32(value)
    }

    /// Build a synthetic header with the given metadata and a single
    /// `token_embd.weight` descriptor whose dims drive vocab inference.
    fn synthetic_gguf(
        metadata: Vec<(&str, MetadataValue)>,
        token_embd_dims: Vec<u64>,
    ) -> GgufFile {
        let metadata: BTreeMap<String, MetadataValue> = metadata
            .into_iter()
            .map(|(k, v)| (k.to_string(), v))
            .collect();
        let tensors = vec![TensorDescriptor {
            name: "token_embd.weight".to_string(),
            dimensions: token_embd_dims,
            tensor_type: TensorType::Q8_0,
            relative_offset: 0,
            absolute_offset: 0,
            n_bytes: 0,
        }];
        GgufFile {
            path: PathBuf::from("synthetic.gguf"),
            version: 3,
            tensor_count: tensors.len() as i64,
            metadata_count: metadata.len() as i64,
            alignment: 32,
            data_start_offset: 0,
            metadata,
            tensors,
        }
    }

    #[test]
    fn from_gguf_applies_defaults_and_infers_vocab() {
        // No head_count_kv, no rms epsilon, no vocab_size: all must fall back.
        // token_embd dims [64, 100] with embedding 64 => vocab inferred 100.
        let gguf = synthetic_gguf(
            vec![
                ("general.architecture", MetadataValue::String("llama".into())),
                ("llama.context_length", u32_meta(512)),
                ("llama.embedding_length", u32_meta(64)),
                ("llama.block_count", u32_meta(2)),
                ("llama.feed_forward_length", u32_meta(128)),
                ("llama.attention.head_count", u32_meta(8)),
            ],
            vec![64, 100],
        );
        let config = LlamaModelConfig::from_gguf(&gguf).unwrap();
        assert_eq!(config.rms_norm_epsilon, 1e-5);
        assert_eq!(config.attention_head_count_kv, 8, "kv head falls back to head count");
        assert_eq!(config.vocab_size, Some(100), "vocab inferred from token_embd");
        assert_eq!(config.rope_freq_base, None, "rope freq base is not defaulted here");
        assert!(!config.rope_neox_pairing, "llama uses adjacent even/odd pairing");
    }

    #[test]
    fn from_gguf_honors_explicit_values() {
        let gguf = synthetic_gguf(
            vec![
                ("general.architecture", MetadataValue::String("qwen3".into())),
                ("qwen3.context_length", u32_meta(4096)),
                ("qwen3.embedding_length", u32_meta(64)),
                ("qwen3.block_count", u32_meta(2)),
                ("qwen3.feed_forward_length", u32_meta(128)),
                ("qwen3.attention.head_count", u32_meta(8)),
                ("qwen3.attention.head_count_kv", u32_meta(2)),
                ("qwen3.attention.layer_norm_rms_epsilon", MetadataValue::F32(1e-6)),
                ("qwen3.vocab_size", u32_meta(777)),
                ("qwen3.rope.freq_base", MetadataValue::F32(1_000_000.0)),
            ],
            vec![64, 777],
        );
        let config = LlamaModelConfig::from_gguf(&gguf).unwrap();
        assert_eq!(config.rms_norm_epsilon, 1e-6);
        assert_eq!(config.attention_head_count_kv, 2);
        assert_eq!(config.vocab_size, Some(777));
        assert_eq!(config.rope_freq_base, Some(1_000_000.0));
        assert!(config.rope_neox_pairing, "qwen3 uses NEOX split-half pairing");
    }

    #[test]
    fn from_gguf_rejects_unknown_architecture() {
        let gguf = synthetic_gguf(
            vec![("general.architecture", MetadataValue::String("mamba".into()))],
            vec![64, 100],
        );
        let err = LlamaModelConfig::from_gguf(&gguf).unwrap_err();
        assert!(
            matches!(err, crate::EngineError::UnsupportedModelArchitecture(_)),
            "{err}"
        );
    }

    #[test]
    fn dense_dims_reject_indivisible_head_count() {
        let gguf = synthetic_gguf(
            vec![
                ("general.architecture", MetadataValue::String("llama".into())),
                ("llama.context_length", u32_meta(512)),
                ("llama.embedding_length", u32_meta(100)),
                ("llama.block_count", u32_meta(2)),
                ("llama.feed_forward_length", u32_meta(128)),
                ("llama.attention.head_count", u32_meta(8)),
            ],
            vec![100, 50],
        );
        let config = LlamaModelConfig::from_gguf(&gguf).unwrap();
        let err = DenseLlamaDims::from_config(&config).unwrap_err();
        assert!(
            matches!(err, crate::EngineError::InvalidModelMetadata(_)),
            "{err}"
        );
    }

    #[test]
    fn implemented_architecture_classification() {
        assert!(is_implemented_architecture("llama"));
        assert!(is_implemented_architecture("qwen3"));
        assert!(!is_implemented_architecture("mamba"));
    }

    /// Full load of a real model, gated on a local GGUF path. Set
    /// CAMELID_ENTERPRISE_TEST_MODEL to enable; the pinned expectations below
    /// are for Llama-3.2-1B-Instruct-Q8_0.
    #[test]
    fn loads_a_real_model_when_available() {
        let Ok(path) = std::env::var("CAMELID_ENTERPRISE_TEST_MODEL") else {
            return;
        };
        let path = Path::new(&path);
        let gguf = read_metadata(path).unwrap();
        let config = LlamaModelConfig::from_gguf(&gguf).unwrap();

        assert_eq!(config.block_count, 16);
        assert_eq!(config.attention_head_count, 32);
        assert_eq!(config.attention_head_count_kv, 8);
        assert_eq!(config.embedding_length, 2048);
        assert_eq!(config.feed_forward_length, 8192);
        assert_eq!(config.rope_freq_base, Some(500000.0));
        assert_eq!(config.rms_norm_epsilon, 1e-5);
        assert_eq!(config.vocab_size, Some(128256));

        let dims = DenseLlamaDims::from_config(&config).unwrap();
        assert_eq!(dims.head_dim, 64);

        // Already-split model: fused expansion is a no-op.
        expand_fused_dense_tensors(&gguf, &config).unwrap();

        let binding = LlamaTensorBinding::bind(&gguf, &config).unwrap();
        assert!(binding.output_is_tied_embedding, "Llama-3.2-1B ties the embedding");
        assert_eq!(binding.layers.len(), 16);

        let store = TensorStore::open(path, &gguf);
        let weights = LlamaWeights::load(&store, &binding).unwrap();
        weights.validate_dense_shapes(&config).unwrap();

        // A sampled Q8_0 layer linear must materialize as plain Q8_0 blocks
        // with the block count implied by its element count.
        let sampled = &weights.layers[0].attention_q;
        let blocks = sampled
            .q8_0_blocks
            .as_ref()
            .expect("attention_q should be plain Q8_0 blocks");
        let expected_blocks = sampled.shape.element_count().unwrap() / 32;
        assert_eq!(blocks.len(), expected_blocks);
        assert!(sampled.data.is_empty(), "block-backed linear keeps data empty");
    }
}
