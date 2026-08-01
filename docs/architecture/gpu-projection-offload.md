# Scoping: GPU offload for the linear projections

**Status: scoping, not a decision.** Nothing here is built. Claims about the
current tree name the file or symbol that carries them; numbers measured on this
hardware say so; everything else is labelled an estimate.

**Date:** 2026-08-01
**Measured on:** Apple M4, 10 cores, 16 GB, `Llama-3.2-1B-Instruct-Q8_0.gguf`
**Code this would touch:** `crates/engine-core/src/forward/compute.rs`,
`crates/engine-core/src/tensor/q8_dot.rs`, `crates/engine-macos`,
`crates/engine-linux`, `crates/engine-windows`

---

## The question

After #31 the replica serves ~46 tok/s on the model above. The Camelid engine's
Metal GPU-resident path serves ~69 tok/s on the same model and box, measured in
the same session with the binaries interleaved. That gap — about **1.5x** — is
what this would close, and the two cheap alternatives are already ruled out:

- the parallel threshold is at its optimum (256 → 41.1, **1024 → 42.4**, 4096 →
  38.5 tok/s, interleaved in one thermal state), so dispatch tuning is spent;
- the macOS Q8_0 kernel is already NEON + dotprod (`crates/engine-macos/src/q8_dot.rs`,
  `sdot`), reached on the serve path via `load_with_q8_dot`.

So the remaining headroom is the GPU. At 46 tok/s the replica moves 60.7 GB/s of
weights, 51% of this host's 120 GB/s; the Metal path reaches 76%.

## The constraint that shapes everything

The deterministic lane's published result is that the same greedy request yields
the identical token stream. A GPU backend that changes one bit changes that
result. So the first question is not "how fast" but **"can a GPU kernel be
bit-identical to the portable reference"**, and for the Q8_0 projection the
answer is yes — for reasons specific enough to be worth writing down.

The reference (`crates/engine-core/src/tensor/q8_dot.rs`) is:

```rust
weight.iter().zip(input).map(|(w, i)| {
    let int_sum = block_int_dot(&w.quants, &i.quants);
    int_sum as f32 * w.scale * i.scale
}).sum()
```

Three properties make that reproducible on a GPU:

1. **The block dot is integer.** 32 products of `i8 × i8` summed in `i32` is
   exact and order-independent, so a GPU may sum it in any order.
2. **The `i32 → f32` conversion is exact.** Worst case is
   `32 × 128 × 128 = 524,288`, well inside f32's exact-integer range of 2²⁴ =
   16,777,216. No rounding happens at the conversion, on any hardware.
3. **The f32 accumulation is sequential from `0.0`.** `.sum()` folds
   left-to-right. A GPU thread that owns one output row and walks its blocks in
   index order performs the identical sequence of roundings.

Note what property 3 requires: **one thread per output row, not a tree reduction
across a row.** A row split across lanes and combined pairwise would reorder the
f32 fold and change bits. This is the same line #31 drew on the CPU — parallelise
across rows, never within one — and it survives onto the GPU unchanged.

**The one real hazard is FMA contraction.** The reference computes
`int_sum as f32 * w.scale * i.scale` and then a separate `+`. Metal Shading
Language contracts `a * b + c` into a fused multiply-add by default, which
rounds once instead of twice and produces different bits. The kernel must be
compiled with contraction off (`-ffp-contract=off`, or `fma`-free arithmetic
written explicitly). This is a compile flag, not a design problem, but it is
silent when wrong: it produces *plausible* output that diverges slowly. It needs
a test, not a comment.

### What the sibling engine's experience says

Camelid's own CUDA resident lane is **not** token-identical to its CPU oracle
below the sliding window — 10/15 legs, disclosed rather than adjudicated, with
worst relative L2 1.89% and smooth growth with depth. The disclosed cause is
design, not GPUs: that lane *"quantizes activations to Q8_0 per GEMV and stores
KV as f16 while the oracle is f32 throughout."*

Both halves of that are informative for this port:

- **Activation quantisation is not a difference here.** `linear_row` already
  quantises the input row to Q8_0 (`quantize_q8_0_blocks`) before every
  projection. Enterprise's reference *is* the Q8-activation path, so a GPU kernel
  matching it inherits no divergence from this.
- **KV precision would be.** `LlamaKvCache` stores `keys: Vec<f32>` and
  `values: Vec<f32>`. Any future move of attention onto the GPU must keep f32 KV
  or accept exactly the divergence Camelid disclosed.

## What blocks it today

`Q8DotRows` is `fn(&[Q8_0Block], &[Q8_0Block]) -> f32` — **one call per output
row**. A GPU cannot be driven through it: the vocabulary head alone is 128,256
rows, so a per-row seam means 128,256 dispatches per token. The seam has to widen
to a whole projection before any GPU work can begin.

That widening pays for itself on CPU independently: it removes ~128k indirect
calls per token on the head and lets the kernel hoist its setup. **Estimated**
5–15%; not measured.

## Phases

**Phase 0 — widen the seam.** Replace the per-row `Q8DotRows` with a
per-projection kernel (weight blocks, input blocks, `&mut [f32]` out). One
function in `engine-core`, one implementation in `engine-macos`, portable
fallback unchanged. Bit-identity is trivially preserved because the same scalar
kernel runs in the same order. CPU-only change, shippable and measurable on its
own. **Estimate: small.** This is the precondition for everything below and the
only phase I would start without further decisions.

**Phase 1 — Metal projection kernel.** A new `engine-macos` path that uploads
the Q8_0 weight blocks once at load (Apple Silicon is unified-memory, so
"upload" is page-aligned residency, not a copy — Camelid's `CAMELID_METAL_NOCOPY`
does exactly this) and dispatches one thread per output row per projection.
Contraction off. Verified by a bit-identity test against the portable reference
over the real weights, plus the existing model-backed parity gate.
**Estimate: the bulk of the work.** Device/buffer/pipeline lifecycle, MSL source,
and the residency question of when weights are uploaded relative to
`LoadedModel::load`.

**Phase 2 — measure, then decide how much further to go.** Projections are most
of the FLOPs but not all of the time; norms, RoPE and attention stay on CPU in
Phase 1, and each round trip costs. Whether attention follows depends on what
Phase 1 measures, and it is the phase where bit-identity gets genuinely hard —
softmax is an f32 reduction whose order would have to be pinned the same way.

**Phase 3 — CUDA, for Windows and Linux.** `crates/engine-linux` and
`crates/engine-windows` exist and today supply only host probes. The same
kernel shape applies; the toolchain, build story and CI story do not. Note CI
builds Windows but deliberately does not run the workspace suite there
(`.github/workflows/ci.yml`, `windows-check`), so a CUDA path arrives with less
test coverage than the Metal one.

## Decisions this needs before Phase 1

1. **Does the GPU lane have to be bit-identical, or may it be a second declared
   posture?** Bit-identity looks achievable and is assumed above, but it
   constrains the kernel to one-thread-per-row and forbids the tree reductions a
   GPU would otherwise prefer. Accepting a *separate* posture would allow a
   faster kernel at the cost of a second identity to publish and defend.
2. **Where does residency live?** Uploading at `LoadedModel::load` makes the
   runtime own a device handle and ends `engine-core`'s current property of being
   platform-free. Uploading lazily at first projection keeps that property and
   costs a branch per call.
3. **macOS first, or both backends together?** Metal alone leaves Windows and
   Linux on the CPU path, which is defensible — but it makes the replica's
   performance host-dependent in a way the identity surface does not currently
   describe.

## Interaction with ADR 0003

This is the trigger [ADR 0003](../adr/0003-identity-after-the-engine-cutover.md)
names: *"the first change that gives `engine-core` more than one execution
path for the same model on the same host."* A GPU backend means a replica can serve
the same weights two ways, which is the first execution posture worth putting in
a configuration vector since the cutover. The identity correction ADR 0003 defers
should land with Phase 1, not after it — decision 1 above is the same decision.

## What this scope does not cover

- Batching. Continuous batching changes numerics by construction and belongs to
  the `throughput` lane, not here. This document is about making one stream
  faster, not many streams cheaper.
- Prefill. Phase 1 accelerates the projection used by both prefill and decode,
  but a batched prefill kernel — processing many positions per dispatch — is a
  separate design with its own bit-identity question.
- Any quantisation other than Q8_0. The wire formats `CpuTensor` carries
  (`q4_k_wire_bytes`, `q5_k_wire_bytes`, `q6_k_wire_bytes`, `tq2_0_wire_bytes`,
  `iq4_xs_wire_bytes`) are out of scope; the seam is Q8_0's.
- Any throughput claim. The 1.5x above is the gap to a *different engine's*
  Metal path on one model on one host, not a prediction of what this port would
  reach.
