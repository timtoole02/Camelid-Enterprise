//! The NEON dot accelerates engine-core's forward pass without changing its
//! output.
//!
//! This is the platform-separation architecture end to end: engine-core owns
//! the portable forward pass and exposes a dot seam; engine-macos supplies a
//! bit-identical NEON kernel through it. Injecting the kernel changes speed,
//! never results — the greedy continuation is the same known-good token
//! sequence the portable path produces.
//!
//! Gated on a local model: set CAMELID_ENTERPRISE_TEST_MODEL to a Llama-3.2-1B
//! Q8_0 GGUF path to run it.

#![cfg(target_arch = "aarch64")]

use engine_core::forward::Decoder;
use engine_core::gguf::read_metadata;
use engine_core::model::{
    expand_fused_dense_tensors, LlamaModelConfig, LlamaTensorBinding, LlamaWeights,
};
use engine_core::tensor::TensorStore;
use std::path::Path;
use std::time::Instant;

#[test]
fn neon_accelerated_forward_matches_known_good_tokens() {
    let Ok(model_path) = std::env::var("CAMELID_ENTERPRISE_TEST_MODEL") else {
        return;
    };
    let path = Path::new(&model_path);

    let gguf = read_metadata(path).unwrap();
    let config = LlamaModelConfig::from_gguf(&gguf).unwrap();
    expand_fused_dense_tensors(&gguf, &config).unwrap();
    let binding = LlamaTensorBinding::bind(&gguf, &config).unwrap();
    let store = TensorStore::open(path, &gguf);
    let weights = LlamaWeights::load(&store, &binding).unwrap();

    // The exact tokenization of a chat-framed "2+2=" request and its known-good
    // greedy continuation for this model.
    let prompt: [u32; 14] = [
        128000, 128006, 882, 128007, 271, 17, 10, 17, 28, 128009, 128006, 78191, 128007, 271,
    ];
    let expected: [u32; 4] = [17, 489, 220, 17];

    // Drive engine-core's forward with the NEON dot injected through the seam.
    let mut decoder =
        Decoder::with_q8_dot(&config, &weights, engine_macos::q8_0_dot_rows).unwrap();

    let started = Instant::now();
    let generated = decoder.generate(&prompt, expected.len()).unwrap();
    let elapsed = started.elapsed();

    assert_eq!(
        generated, expected,
        "NEON-accelerated forward must produce the same tokens as the portable path"
    );

    let tokens = prompt.len() + expected.len();
    eprintln!(
        "[neon-forward] {tokens} tokens in {:.2}s ({:.1} ms/token)",
        elapsed.as_secs_f64(),
        elapsed.as_secs_f64() * 1000.0 / tokens as f64
    );
}
