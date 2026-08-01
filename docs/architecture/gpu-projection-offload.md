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

> **The project has since chosen not to require it** (see *Decision 1* below):
> the GPU path is a second declared posture. This section is retained because it
> is what makes that a *choice* rather than a concession — bit-identity was
> available and was traded for kernel freedom — and because Phase 0 still
> preserves it, and it remains the fallback if the second posture proves more
> expensive to publish than it is worth.

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
does exactly this) and dispatches a projection at a time. Under Decision 1 the
reduction shape is unconstrained — pick whatever is fastest — subject only to
being *fixed and data-independent*, with no float atomics, so the posture stays
reproducible run to run.

Verified by: a **restart-reproducibility** test (same request twice across a full
process restart, byte-identical), a comparison against the CPU posture on a
committed prompt pack to quantify the divergence the posture is declaring, and
the existing model-backed parity gate still passing **on the CPU posture**.
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

## Decision 1: a second declared posture — DECIDED

**The GPU path is a second declared posture, not a bit-identical replacement.**

The kernel is therefore free of the one-thread-per-row constraint above. It may
use whatever reduction shape is fastest — threadgroup tree reductions, SIMD-group
sums, wider tiles — and the FMA-contraction hazard stops being a correctness
issue and becomes a performance choice. That is the point of the decision, and it
is what makes a GPU kernel worth writing rather than a GPU-shaped CPU kernel.

### What the decision does NOT relax

**Reproducibility within the posture.** "Not identical to the CPU path" is not
"not deterministic". The lane's published result is that the same greedy request
yields the identical token stream on every run, *including across process
restarts* — that claim is per-posture and survives this decision intact. The GPU
kernel must therefore avoid the sources of **run-to-run** variance, which are
different from the sources of CPU-difference:

- **no float atomics.** An `atomic_fetch_add` over f32 accumulates in whatever
  order the threadgroups happen to retire, which varies run to run. This is the
  single most common way a GPU kernel becomes irreproducible, and it is easy to
  reach for.
- **a fixed, data-independent reduction order.** A tree reduction is fine — it
  just has to be the *same* tree every time, so threadgroup size and tile shape
  must not depend on runtime conditions such as device occupancy.
- **no dependence on dispatch order** between projections that write shared
  state.

A kernel meeting those is reproducible without being CPU-identical, which is
exactly the posture being declared.

### What it costs

Two things now need building that bit-identity would have given for free:

1. **A second identity to publish and defend.** Two replicas serving the same
   weights on different postures produce different tokens, and a client must be
   able to tell them apart *before* it compares outputs. This is no longer
   optional — see the ADR 0003 section below.
2. **An acceptance criterion for the GPU posture.** The model-backed parity gate
   (`in_tree_generation_matches_pinned_engine`) asserts token-identity against
   the pinned oracle. That gate now belongs to the **CPU posture**. The GPU
   posture needs its own, and "token-identical to the oracle" is no longer
   available as the bar. The sibling engine's answer is an evidence bundle with
   the divergence measured and disclosed rather than adjudicated — worth copying
   in shape, since the alternative is a lane whose quality is unstated.

**Open: what the GPU posture's acceptance bar actually is.** Candidates: token
identity against the *CPU posture* over a committed prompt pack at fixed depths;
a disclosed divergence budget in the manner of the sibling engine's bundles; or
top-1 agreement above a stated threshold. This wants deciding before Phase 1
ships, not before it starts — Phase 1 can be measured against the CPU posture
while the bar is chosen.

## Remaining decisions before Phase 1

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
a configuration vector since the cutover.

**Decision 1 promotes that from "should" to "must".** While both paths were going
to be bit-identical, a replica that failed to say which one it ran produced the
same tokens either way, so the stale digest was an honesty problem rather than a
correctness one. A second *declared* posture is only declared if something
declares it: two replicas serving the same weights now emit different tokens, and
a client that cannot tell them apart before comparing outputs will read a posture
difference as a bug in one of them.

So the identity correction ADR 0003 defers is no longer deferrable past Phase 1.
Concretely, the versioned change must land in the same release: publish an engine
identity, re-scope `config_sha256` so it distinguishes the two postures, and
retire the fourteen `CAMELID_*` keys that describe neither. ADR 0003's own list
of every published copy of both digests is the checklist for that change.

## What this scope does not cover

- Batching. Continuous batching changes numerics by construction and belongs to
  the `throughput` lane, not here. This document is about making one stream
  faster, not many streams cheaper.
- Prefill. Phase 1 accelerates the projection used by both prefill and decode,
  but a batched prefill kernel — processing many positions per dispatch — is a
  separate design, and one whose reduction shape would have to be pinned against
  the *position count* to stay reproducible.
- Any quantisation other than Q8_0. The wire formats `CpuTensor` carries
  (`q4_k_wire_bytes`, `q5_k_wire_bytes`, `q6_k_wire_bytes`, `tq2_0_wire_bytes`,
  `iq4_xs_wire_bytes`) are out of scope; the seam is Q8_0's.
- Any throughput claim. The 1.5x above is the gap to a *different engine's*
  Metal path on one model on one host, not a prediction of what this port would
  reach.
