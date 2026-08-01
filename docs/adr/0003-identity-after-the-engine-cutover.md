# 0003 — What the identity surface still describes after the engine cutover

> **Status: accepted.** This record does not change a published value. It states
> what two of them mean **today**, which is not what [ADR
> 0002](0002-replica-identity-surface.md) said they meant when the replica ran a
> different engine, and it defers the correction with a stated trigger.

**Date:** 2026-08-01
**Applies to:** the `deterministic` lane, at and after the in-tree cutover (#28)
**Code:** `crates/server/src/lane.rs`, `crates/server/src/main.rs`, `crates/engine-core/`
**Supersedes nothing.** It narrows two claims in ADR 0002; that record's
reasoning was correct for the engine it was written against.

---

## Context

ADR 0002 established what a replica publishes about itself, and its central
claim is that `config_sha256` lets a client "treat two replicas carrying the
same digest as interchangeable". That claim was built on a specific mechanism:
the lane writes a canonical vector of `CAMELID_*` environment keys, the engine
reads them, and the digest of the vector therefore describes how the engine will
execute.

The engine that read those keys is gone. PR #28 replaced it with
`crates/engine-core`, which the workspace owns, and moved the original to a
dev-only parity oracle (`docs/architecture/engine-dependency.md`). The identity
surface was not revisited in the same change.

## What is no longer true

**The configuration vector configures nothing in production.**
`apply_deterministic` (`crates/server/src/lane.rs`) writes the fourteen
`CANONICAL` keys — `CAMELID_DETERMINISTIC`, `CAMELID_NO_GPU_SAMPLE`,
`CAMELID_METAL_RESIDENT_DECODE`, `CAMELID_METAL_F32Y`, `CAMELID_METAL_WIRE`,
`CAMELID_METAL_WIRE_NSG8`, `CAMELID_METAL_ATTN2`,
`CAMELID_METAL_RESIDENT_PREFILL`, `CAMELID_METAL_MM`, `CAMELID_METAL_LINEAR`,
`CAMELID_METAL_Q8`, `CAMELID_METAL_Q8_RETAINED`, `CAMELID_HYBRID_Q8_RETAINED`,
`CAMELID_METAL_NOCOPY` — into the process environment, and hashes them into

```
30d77c2608036f8475372ace9ec125ffc5fa16d8d63f0355a08c32c69f4449b7
```

published on every response as `x-camelid-config-sha256` and
`camelid_config_sha256`.

**Evidence:** `crates/engine-core` reads no environment variable on any serving
path. Every `std::env::var` in it naming a `CAMELID_*` key is inside a
`#[cfg(test)]` module — `model/mod.rs:166` and `tokenizer.rs:1560,1562,1828`,
all test-gated, all reading `CAMELID_ENTERPRISE_TEST_MODEL` or the two
Llama-3 tokenizer fixtures. Nor does the parity oracle see the writes:
`apply_deterministic` is called from `crates/server/src/main.rs` and from
`crates/server/tests/lane_environment.rs`, and from nowhere else — in
particular not from `crates/server/tests/replica_contract_model.rs`, where the
pinned engine actually runs.

So the digest is a hash of settings that no code in the serving process reads.
It is stable, and two replicas carrying it really are interchangeable — but that
follows from the engine being one implementation compiled into one binary, not
from the vector, and the digest would be equally stable if every key in it were
deleted.

**The published worker width sizes a pool nothing reads.**
`resolve_worker_pool` (`crates/server/src/main.rs`) builds a global `rayon`
pool from `--threads` and publishes `rayon::current_num_threads()` as
`x-camelid-worker-threads`. `crates/engine-core` contains no `rayon`, no
`par_iter`, no `ThreadPool` and no `std::thread::spawn`; its forward pass is
single-threaded. The only remaining consumers of `rayon` in the workspace are
the sizing call itself and the width read back from it.

## What is still true

**Admission is real, and `admission_sha256` still means what it says.** The scan
in `refuse_unpermitted` runs against the operator's environment before the lane
writes anything, and it fails a start closed. It is a claim about what this
build would refuse, not about what the engine reads, so removing the engine did
not weaken it. Its published value moved to
`318fb6d65c0fb2cd3630594b08cc70a1bc3ae0bca7b8bd15c121458e651959f6` in #29 when
`CAMELID_ENTERPRISE_MAX_CONCURRENCY` was admitted.

**The model digest is the load-bearing identity.** `x-camelid-model-sha256` is
taken over the GGUF before the port is bound and re-verified after load
(`crates/server/src/lib.rs`, `replica_router`). It names the weights, which is
the input that actually moves every token.

**Output is genuinely reproducible.** It is reproducible because
`engine-core`'s forward pass is single-threaded with a fixed reduction order and
greedy selection, not because of anything the vector says. #29's worker pool
does not change this: each generation owns its decoder and KV cache over
read-only weights, so a request emits the same tokens at any width.

## The gap this leaves

Nothing published identifies **which build of the in-tree engine produced a
token**. Before the cutover the engine was pinned by revision and that pin was
published as `ENGINE_PIN`; that field now names the *oracle*, not the engine
that serves. Two Enterprise builds with different `engine-core` code publish
identical `config_sha256`, identical `admission_sha256` and identical
`ENGINE_PIN`, and are indistinguishable from outside. That is a larger hole than
the stale vector, and it is the one a client asking "what produced this" most
needs closed.

## Decision

Record the defect now; do not redefine the identity surface yet.

Changing what `config_sha256` covers, or retiring it for an engine identity, is
a versioned identity-contract decision in the sense ADR 0002 uses: every
published copy moves, and clients comparing replicas across the change get a
false negative. It is worth doing once.

It is not worth doing twice, and doing it now would be doing it twice. The
in-tree engine is CPU-only today (`engine-core` has no GPU backend, and the
macOS crate supplies one Q8_0 kernel through an explicit seam). A GPU backend is
planned for macOS/Metal and for Windows and Linux. That work will introduce the
first execution posture this replica can actually vary on — resident versus
portable kernels, device present or not, per-host kernel selection — which is
precisely the material a configuration vector should describe. Designing the
vector before that variance exists would produce a second vector describing
nothing.

**Trigger:** the first change that gives `engine-core` more than one execution
path for the same model on the same host. At that point, in one versioned
change: publish an engine identity, re-scope or retire `config_sha256`, and
decide whether `--threads` names anything.

Until then, the correction is documentary: prose that says the vector configures
the forward pass is changed to say what it does.

## What this record does not close

- It does not make `config_sha256` meaningful. It states that it is not, and
  leaves it published and unchanged.
- It does not remove the `CANONICAL` writes. They are inert in production and
  cost one pass over fourteen keys at startup; removing them is part of the
  versioned change, not a tidy-up to be done separately.
- It does not remove the `rayon` dependency or the `--threads` flag. The flag
  still refuses a zero width and still publishes the width it resolved; it just
  does not size anything that serves a request.
- It does not claim the engine reads no environment variable *ever*. It claims
  no serving path in `crates/engine-core` reads one, with the test-gated
  exceptions named above.
- It makes no claim about the gateway, whose configuration is its own process's
  concern.
