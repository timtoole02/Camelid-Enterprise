//! The dense decoder forward pass.
//!
//! This module builds the order-stable CPU decode spine on the engine-core
//! primitives: rotary position embedding ([`rope`]), the position-major f16
//! key/value cache ([`kv_cache`]), and the per-layer attention + FFN compute
//! with the per-token decode loop and greedy sampling ([`compute`]).
//!
//! All arithmetic is portable and serial. The only acceleration seam is the
//! Q8_0 row dot inside [`compute`], which routes through
//! [`crate::tensor::q8_0_dot_rows`] at a single call site so a platform crate
//! can inject a hardware kernel without touching the spine.

pub mod compute;
pub mod kv_cache;
pub mod rope;

pub use compute::{argmax, Decoder};
pub use kv_cache::{f16_to_f32_kv, f32_to_f16_kv, LlamaKvCache, LlamaKvCachePlan};
pub use rope::{apply_rope, RopePairing};

#[cfg(test)]
mod integration_tests {
    use std::path::Path;

    use super::compute::{argmax, Decoder};
    use crate::gguf::read_metadata;
    use crate::model::{
        expand_fused_dense_tensors, LlamaModelConfig, LlamaTensorBinding, LlamaWeights,
    };
    use crate::tensor::TensorStore;

    /// End-to-end forward pass on a real model, gated on a local GGUF path via
    /// `CAMELID_ENTERPRISE_TEST_MODEL`. Runs a short prompt through prefill +
    /// decode and checks the logits are well-formed and the argmax is stable
    /// across two independent runs (deterministic spine).
    #[test]
    fn forward_pass_runs_and_is_deterministic() {
        let Ok(path) = std::env::var("CAMELID_ENTERPRISE_TEST_MODEL") else {
            return;
        };
        let path = Path::new(&path);
        let gguf = read_metadata(path).unwrap();
        let config = LlamaModelConfig::from_gguf(&gguf).unwrap();
        expand_fused_dense_tensors(&gguf, &config).unwrap();
        let binding = LlamaTensorBinding::bind(&gguf, &config).unwrap();
        let store = TensorStore::open(path, &gguf);
        let weights = LlamaWeights::load(&store, &binding).unwrap();
        weights.validate_dense_shapes(&config).unwrap();

        // A short token sequence within any vocab (BOS-ish small ids).
        let prompt: [u32; 4] = [1, 15043, 1734, 29901];

        let run = |cfg: &LlamaModelConfig, w: &LlamaWeights| -> (u32, Vec<f32>) {
            let mut decoder = Decoder::new(cfg, w).unwrap();
            // Prefill all but the last token (no logits).
            for &token in &prompt[..prompt.len() - 1] {
                assert!(decoder.forward_token(token, false).unwrap().is_none());
            }
            // Decode the last token with logits.
            let logits = decoder
                .forward_token(prompt[prompt.len() - 1], true)
                .unwrap()
                .expect("logits on the final token");
            assert_eq!(decoder.position(), prompt.len());
            let vocab = config.vocab_size.unwrap() as usize;
            assert_eq!(logits.shape.dims, vec![1, vocab]);
            assert!(
                logits.data.iter().all(|v| v.is_finite()),
                "all logits must be finite"
            );
            (argmax(&logits).unwrap(), logits.data)
        };

        let (token_a, logits_a) = run(&config, &weights);
        let (token_b, logits_b) = run(&config, &weights);
        assert_eq!(token_a, token_b, "argmax must be deterministic");
        assert_eq!(
            logits_a.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            logits_b.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            "logits must be bit-identical across runs"
        );

        // One more decode step continues deterministically.
        let mut decoder = Decoder::new(&config, &weights).unwrap();
        for &token in &prompt[..prompt.len() - 1] {
            decoder.forward_token(token, false).unwrap();
        }
        let first = decoder.forward_token(prompt[prompt.len() - 1], true).unwrap().unwrap();
        let next = argmax(&first).unwrap();
        let second = decoder.forward_token(next, true).unwrap().unwrap();
        assert_eq!(second.shape.dims, vec![1, config.vocab_size.unwrap() as usize]);
    }

    /// Correctness gate: the greedy continuation must equal the known-good
    /// token sequence, token for token. The prompt ids are the exact
    /// tokenization of a chat-framed "2+2=" request and the expected ids are the
    /// known-good greedy continuation for this model — so a match proves the
    /// whole forward pass (embedding, attention, RoPE, KV, FFN, lm_head, argmax)
    /// is numerically faithful, not merely self-consistent.
    #[test]
    fn greedy_generation_matches_known_good_tokens() {
        let Ok(path) = std::env::var("CAMELID_ENTERPRISE_TEST_MODEL") else {
            return;
        };
        let path = Path::new(&path);
        let gguf = read_metadata(path).unwrap();
        let config = LlamaModelConfig::from_gguf(&gguf).unwrap();
        expand_fused_dense_tensors(&gguf, &config).unwrap();
        let binding = LlamaTensorBinding::bind(&gguf, &config).unwrap();
        let store = TensorStore::open(path, &gguf);
        let weights = LlamaWeights::load(&store, &binding).unwrap();

        let prompt: [u32; 14] = [
            128000, 128006, 882, 128007, 271, 17, 10, 17, 28, 128009, 128006, 78191, 128007, 271,
        ];
        let expected: [u32; 4] = [17, 489, 220, 17];

        let mut decoder = Decoder::new(&config, &weights).unwrap();
        let generated = decoder.generate(&prompt, expected.len()).unwrap();
        assert_eq!(
            generated, expected,
            "greedy continuation must match the known-good tokens for this model"
        );
    }
}
