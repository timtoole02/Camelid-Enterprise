//! The AVX2 dot accelerates engine-core's forward pass without changing one bit
//! of its output.
//!
//! This is the platform-separation architecture end to end: engine-core owns the
//! portable forward pass and exposes a dot seam; engine-windows supplies a
//! bit-identical AVX2 kernel through it. Two decoders are built over the *same*
//! loaded weights — one on engine-core's portable dot, one on this crate's
//! kernel — driven in lockstep, and the whole `[1, vocab]` logits vector is
//! compared `to_bits()` at every step where logits exist.
//!
//! ## Why the logits and not the tokens
//!
//! The token comparison at the end is the **outer** gate; it is not the gate.
//! `argmax` collapses a 128256-wide f32 vector to one index, so any numeric
//! defect that does not cross an argmax boundary is invisible to it — and most
//! do not. An upstream campaign planted a real one-position numeric defect and
//! measured which observable caught it across nine items at 513-2400 tokens:
//! token identity caught it 0/9, numeric (KV/hidden) equivalence caught it 9/9.
//! Token identity stays here because it is the claim engine-macos's NEON gate
//! also makes, on the same prompt and the same continuation, and two ports
//! asserting one continuation is the cross-platform determinism claim. But it is
//! the weaker of the two things this file asserts, and it is written second on
//! purpose.
//!
//! Bit-identity is asserted here because it is achievable for *this* kernel, not
//! because it is the general acceleration contract. `engine_windows::POSTURE`
//! declares that, and a path admitted under a published tolerance would carry a
//! different posture and need a different gate — this one would correctly refuse
//! it rather than quietly widen.
//!
//! ## What this does NOT prove
//!
//! It never exercises this crate's AVX2 **quantizer**. `linear_row` quantizes
//! the activation row with engine-core's own portable `quantize_q8_0_blocks`
//! before dotting it against every weight row, and the seam replaces the dot and
//! nothing else. The quantizer's only coverage is the in-crate fuzz in
//! `src/q8_dot.rs`.
//!
//! It covers Q8_0 weights only — a model whose linears are K-quants sends
//! `linear_row` down a different arm and the seam is never called, which is why
//! the guards below check for Q8_0 blocks by name rather than letting that
//! surface as a confusing shape error. It says nothing about a batched prefill:
//! there is none, tokens are fed one at a time. And with
//! `CAMELID_ENTERPRISE_TEST_MODEL` unset it proves nothing at all — no CI job
//! sets it. As on macOS, end-to-end equality is a local claim; the CI-enforced
//! claim is the in-crate bit-identity fuzz.
//!
//! Gated on `target_arch`, not `target_os`: the kernel is selected by
//! instruction set, so this is the honest gate, and the file compiles away
//! entirely on the aarch64-pc-windows-msvc cross-check — where
//! `engine_windows::q8_0_dot_rows` *is* the portable reference and the
//! comparison would be the reference against itself.

#![cfg(target_arch = "x86_64")]

use engine_core::forward::{argmax, Decoder};
use engine_core::gguf::read_metadata;
use engine_core::model::{
    expand_fused_dense_tensors, LlamaModelConfig, LlamaTensorBinding, LlamaWeights,
};
use engine_core::tensor::{
    quantize_q8_0_blocks, CpuTensor, Q8_0Block, TensorStore, Q8_0_BLOCK_VALUES,
};
use std::path::Path;
use std::time::Instant;

/// The exact tokenization of a chat-framed "2+2=" request and its known-good
/// greedy continuation for this model. Shared verbatim with engine-macos's NEON
/// gate and with engine-core's own portable
/// `greedy_generation_matches_known_good_tokens`; if a model swap ever breaks
/// these, all three fail together, which is the intended signal.
const PROMPT: [u32; 14] = [
    128000, 128006, 882, 128007, 271, 17, 10, 17, 28, 128009, 128006, 78191, 128007, 271,
];
const EXPECTED: [u32; 4] = [17, 489, 220, 17];

/// The activation width this model projects from, used by the control's premise
/// check so it runs a shape the forward pass actually issues.
const EMBEDDING_WIDTH: usize = 2048;

/// The negative control: this crate's kernel, off by exactly one ULP.
///
/// `^ 1` rather than `f32::next_up`, and the reason is totality rather than the
/// manifest's `rust-version` — that pin is already unattainable, since this
/// workspace calls `next_up` and `is_multiple_of` (1.86 and 1.87) and no job
/// builds at the pinned version. The XOR is defined on every bit pattern and
/// changes every one of them, where `next_up` is the identity on NaN and on
/// `+inf`; it cannot manufacture a NaN or an infinity from a finite input
/// (reaching `0x7F800000` from a finite value would require flipping an exponent
/// bit); and it moves `-0.0` as readily as anything else, which matters here
/// because the sign of zero is what this kernel's reduction is pinned on.
///
/// **Every** call is perturbed, not one. Keys and values round through f16 on
/// their way into the KV cache, so a single-site perturbation would usually be
/// absorbed before it reached a logit and the control would be the vacuous half
/// of this file. The honest consequence, stated rather than overclaimed: this
/// demonstrates the comparison has power against a pervasive defect. It does not
/// establish a detection floor for a lone one-site defect.
///
/// It cannot be optimized into the identity: the value is returned and compared,
/// and the kernel is reached through a `fn`-pointer field on `Decoder`, an
/// indirect call across a crate boundary.
#[inline(never)]
fn one_ulp_perturbed_dot(weight: &[Q8_0Block], input: &[Q8_0Block]) -> f32 {
    f32::from_bits(engine_windows::q8_0_dot_rows(weight, input).to_bits() ^ 1)
}

/// Compare two logits vectors bit for bit, reporting the first divergence.
///
/// `to_bits()` and not `==`, for two independent reasons. `-0.0 == 0.0` is true,
/// and the `-0.0` accumulator seed is this kernel's documented observable — the
/// one property a `0.0`-seeded port reproduces wrong while passing everything
/// else. And a NaN logit compares unequal to itself, so `==` would report a
/// divergence between a vector and itself.
///
/// The differing *count* is reported as well as the first index, because it is
/// the first thing a debugger needs: one differing index means a single lm_head
/// row is wrong, while most of the vocabulary differing means the hidden state
/// diverged upstream and the head is only reporting it.
fn assert_logits_bit_identical(want: &CpuTensor, got: &CpuTensor, step: usize) {
    let first = want
        .data
        .iter()
        .zip(&got.data)
        .enumerate()
        .find(|(_, (a, b))| a.to_bits() != b.to_bits());
    let Some((index, (reference, accelerated))) = first else {
        return;
    };
    let differing = want
        .data
        .iter()
        .zip(&got.data)
        .filter(|(a, b)| a.to_bits() != b.to_bits())
        .count();
    panic!(
        "step {step}: the AVX2-accelerated logits are not bit-identical to the portable \
         reference. {differing} of {} indices differ; the first is {index}, where the \
         reference has {reference} (0x{:08x}) and the accelerated path has {accelerated} \
         (0x{:08x}).",
        want.data.len(),
        reference.to_bits(),
        accelerated.to_bits(),
    );
}

/// Drive one token through both decoders and assert they agree bit for bit.
///
/// Returns the **accelerated** path's logits, so the outer token gate literally
/// speaks for the kernel under test rather than restating the reference's
/// answer. That choice is dominated by the assertion one line above it: the two
/// vectors were just proven identical.
///
/// The position check catches a KV desync *as* a desync. Without it, two
/// decoders that had drifted out of step would report a logits mismatch and
/// leave the reader to work out that the numbers were never comparable.
fn lockstep_logits(
    reference: &mut Decoder<'_>,
    accelerated: &mut Decoder<'_>,
    token: u32,
    step: usize,
    vocab: usize,
) -> CpuTensor {
    let want = reference
        .forward_token(token, true)
        .unwrap_or_else(|err| panic!("step {step}: the portable forward failed: {err}"))
        .unwrap_or_else(|| panic!("step {step}: logits were requested and not returned"));
    let got = accelerated
        .forward_token(token, true)
        .unwrap_or_else(|err| panic!("step {step}: the AVX2 forward failed: {err}"))
        .unwrap_or_else(|| panic!("step {step}: logits were requested and not returned"));

    assert_eq!(
        reference.position(),
        accelerated.position(),
        "step {step}: the two decoders' KV caches are at different positions, so their \
         logits were never comparable"
    );
    assert_eq!(want.shape.dims, vec![1, vocab], "step {step}: reference shape");
    assert_eq!(got.shape.dims, vec![1, vocab], "step {step}: accelerated shape");
    assert_logits_bit_identical(&want, &got, step);
    got
}

#[test]
fn avx2_accelerated_forward_is_bit_identical_in_the_logits() {
    let Ok(model_path) = std::env::var("CAMELID_ENTERPRISE_TEST_MODEL") else {
        return;
    };
    let path = Path::new(&model_path);

    // Vacuity guard, using this crate's published vocabulary because
    // `x86::avx2_enabled` is private to the q8_dot module. Without AVX2 both
    // decoders run the portable reference and the comparison below compares it
    // against itself — green, and proving nothing.
    assert!(
        engine_windows::probe().simd.contains(&"avx2"),
        "this x86-64 host does not advertise AVX2, so the accelerated decoder below is the \
         portable reference and this test compares it against itself"
    );

    // The control's premise, checked before the model is even opened: the
    // perturbation is not an identity function, and it is exactly one ULP. A
    // future edit that neutered the wrapper would fail here, on its own named
    // assertion, rather than silently making the whole gate green by
    // construction.
    let row = |seed: u32| -> Vec<Q8_0Block> {
        let values: Vec<f32> = (0..EMBEDDING_WIDTH)
            .map(|i| {
                let step = (i as u32).wrapping_mul(2654435761).wrapping_add(seed);
                (step % 977) as f32 * 0.01 - 4.0
            })
            .collect();
        quantize_q8_0_blocks(&values)
    };
    let (weight_row, input_row) = (row(1), row(7));
    assert_eq!(weight_row.len(), EMBEDDING_WIDTH / Q8_0_BLOCK_VALUES);
    let exact = engine_windows::q8_0_dot_rows(&weight_row, &input_row);
    let perturbed = one_ulp_perturbed_dot(&weight_row, &input_row);
    assert_ne!(
        perturbed.to_bits(),
        exact.to_bits(),
        "the negative control is an identity function, so it can prove nothing"
    );
    assert_eq!(
        (i64::from(perturbed.to_bits()) - i64::from(exact.to_bits())).abs(),
        1,
        "the negative control must be exactly one ULP, or the gate's measured power is not \
         the power it claims"
    );

    let gguf = read_metadata(path).unwrap();
    let config = LlamaModelConfig::from_gguf(&gguf).unwrap();
    expand_fused_dense_tensors(&gguf, &config).unwrap();
    let binding = LlamaTensorBinding::bind(&gguf, &config).unwrap();
    let store = TensorStore::open(path, &gguf);
    let weights = LlamaWeights::load(&store, &binding).unwrap();
    weights.validate_dense_shapes(&config).unwrap();
    let vocab = config.vocab_size.unwrap() as usize;

    // The seam has to actually be on the path. A non-Q8_0 GGUF sends
    // `linear_row` down its f32 arm, `q8_dot` is never called, and every
    // comparison below would pass by construction — the comparison itself has no
    // way to notice.
    assert!(
        weights.layers[0].attention_q.q8_0_blocks.is_some(),
        "this model's attention projections are not Q8_0, so the dot seam is never reached \
         and this test would be green without executing the kernel"
    );
    assert!(
        weights.output_projection().q8_0_blocks.is_some(),
        "this model's output projection is not Q8_0, so the logits are produced without the \
         dot seam"
    );

    // Two decoders over ONE weight set. `Decoder<'w>` borrows the weights
    // shared, so this costs one 1.4 GB materialization rather than two; only the
    // KV caches are per-decoder, and those grow lazily with the sequence.
    let mut reference = Decoder::new(&config, &weights).unwrap();
    let mut accelerated =
        Decoder::with_q8_dot(&config, &weights, engine_windows::q8_0_dot_rows).unwrap();

    let started = Instant::now();
    let mut steps = 0_usize;
    let mut baseline: Option<Vec<f32>> = None;
    let mut logits: Option<CpuTensor> = None;

    // Prefill, with logits computed at every position rather than only the last.
    // That is numerically inert — the logits branch consumes `hidden`, touches
    // no decoder state, and `advance_position` runs identically either way — and
    // it raises the number of compared vectors from 4 to 17 for a modest wall
    // cost. A defect that only shows during prefill has nowhere to hide.
    for (index, &token) in PROMPT.iter().enumerate() {
        let step = lockstep_logits(&mut reference, &mut accelerated, token, index, vocab);
        if index == 0 {
            baseline = Some(step.data.clone());
        }
        logits = Some(step);
        steps += 1;
    }

    // Decode. The last emitted token is never fed back, so no logits exist for
    // it — which is why the step count below is one short of prompt + generated.
    let mut generated = Vec::with_capacity(EXPECTED.len());
    loop {
        let held = logits.as_ref().expect("prefill leaves logits held");
        let next = argmax(held).unwrap();
        generated.push(next);
        if generated.len() == EXPECTED.len() {
            break;
        }
        logits = Some(lockstep_logits(
            &mut reference,
            &mut accelerated,
            next,
            steps,
            vocab,
        ));
        steps += 1;
    }

    // The loop really compared every position it drove. Without this a
    // restructuring that broke early would shrink the gate silently.
    assert_eq!(
        steps,
        PROMPT.len() + EXPECTED.len() - 1,
        "the lockstep loop compared fewer positions than it drove"
    );
    assert_eq!(reference.position(), steps);

    // The outer gate. Same continuation engine-macos asserts for NEON and
    // engine-core asserts for the portable path, taken from the accelerated
    // decoder's own logits.
    assert_eq!(
        generated, EXPECTED,
        "AVX2-accelerated forward must produce the same tokens as the portable path"
    );

    // The negative control, on the baseline the prefill already captured — one
    // extra forward call rather than a second full prefill. It doubles as the
    // seam-liveness proof: were the injected kernel never reached, the perturbed
    // decoder would be bit-identical too and this assertion would fail.
    let baseline = baseline.expect("the prefill captured a step-0 baseline");
    let mut control = Decoder::with_q8_dot(&config, &weights, one_ulp_perturbed_dot).unwrap();
    let control_logits = control
        .forward_token(PROMPT[0], true)
        .expect(
            "the perturbed forward failed. At one ULP this should be an ordinary forward \
             pass; a softmax-normalization error here is the control tripping a numeric \
             guard rather than the control's claim failing",
        )
        .expect("logits were requested and not returned");
    let control_differing = baseline
        .iter()
        .zip(&control_logits.data)
        .filter(|(a, b)| a.to_bits() != b.to_bits())
        .count();
    assert_ne!(
        control_differing, 0,
        "a kernel perturbed by one ULP on every call produced bit-identical logits. The \
         comparison above therefore has no power and this gate is green by construction — \
         either the seam is not reached, or the comparison is not comparing what it reads."
    );

    let elapsed = started.elapsed();
    eprintln!(
        "[avx2-forward] {steps} logits vectors compared bit for bit ({vocab} f32 each, \
         {} pairs total) in {:.2}s; negative control diverged at {control_differing} of \
         {vocab} indices",
        steps * vocab,
        elapsed.as_secs_f64(),
    );
}
