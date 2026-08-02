# 0004 — Publishing the engine identity and the execution posture

> **Status: accepted.** Supersedes §1 of [ADR 0002](0002-replica-identity-surface.md)
> — the preimage of `config_sha256` — and closes the correction
> [ADR 0003](0003-identity-after-the-engine-cutover.md) deferred. Everything else
> in 0002 stands.

**Date:** 2026-08-01
**Code:** `crates/server/build.rs`, `crates/server/src/lane.rs`, `crates/server/src/attribution.rs`

---

## Context

ADR 0003 recorded two defects and deliberately did not fix them: `config_sha256`
hashed fourteen `CAMELID_*` keys that no serving path reads, and **nothing
published identified which build of `engine-core` produced a token**. It named
the second as the larger hole, and set a trigger: the first change giving the
engine more than one execution path for the same model on the same host.

That trigger is now in sight rather than hypothetical — a GPU backend is scoped
(`docs/architecture/gpu-projection-offload.md`) and has been decided to be a
**second declared posture** rather than a bit-identical replacement. Two
replicas on the same weights will emit different tokens, so a client that cannot
tell the postures apart before comparing outputs would read a posture difference
as a bug.

## Decision

**Three published values, not one.**

1. `config_sha256` is now SHA-256 over `posture=<EXECUTION_POSTURE>\n` then
   `engine_pin=<ENGINE_PIN>`. Today:
   `b62869e991172aadb0204c526ff41fd7486434320884bda323e36cff6e13b00d`, published
   as `b62869e99117`. The canonical environment vector is gone from the preimage.
2. `x-camelid-posture` / `camelid_posture` / receipt `posture` — which
   forward-pass implementation served the token. One value today, `cpu`.
3. `x-camelid-engine-sha256` / `camelid_engine_sha256` / receipt `engine_sha256`
   — SHA-256 over every `.rs` under `crates/engine-core/src`, path and bytes, in
   sorted order, computed in `build.rs`.

### Why the posture field exists before a second posture does

Adding a **field** to a published identity is the versioned change; adding a
**value** to a field that already exists is not. Introducing `posture` now, with
one value, means the GPU backend publishes `metal` without moving anyone's
contract a second time. Doing it the other way round would spend the cascade
twice.

### Why the engine digest is beside `config_sha256` and not inside it

It is a genuine identity but a **fast-moving** one: it changes with every edit to
the engine's source. Folding it into `config_sha256` — a value pinned by a test
and quoted in three published documents — would move that digest on every
ordinary engine change, and a digest that churns is one nobody keeps accurate.
Published separately, `config_sha256` moves only when the posture set or the
oracle pin does, both deliberate acts, and a client is told *which* of the three
moved. This is the same reasoning 0002 used to split admission from config.

### Why a source digest rather than a git revision

It is correct from a tarball, from a dirty tree, and in CI, and it moves when
and only when the engine's source moves. A git revision is wrong in the first
two cases and moves for changes that cannot affect a token.

### What happened to the canonical vector

`CANONICAL` is **kept**, and is now an admission concern only. The lane still
writes those fourteen keys, because that is what lets `refuse_unpermitted`
refuse an operator who pre-sets one *even at the canonical value*, with a message
that says why. Deleting the list would leave those keys refused by the general
deny-by-default rule with a less useful message and would buy nothing. Editing
it now changes what this replica **refuses** (`admission_sha256`) and no longer
changes what it **is** (`config_sha256`).

## What this record does not close

- It does not make the deterministic lane's output depend on `posture`. Today
  there is one posture; the field is a promise about how the next one will be
  announced, not evidence that one exists.
- It does not give the GPU posture an acceptance criterion. The model-backed
  parity gate asserts token-identity against the pinned oracle and belongs to
  the **CPU** posture. What bar a non-bit-identical posture must clear is open,
  and is called out in the GPU scoping document. The *shape* that bar has to
  take — and the measurements from another campaign that make each of its terms
  non-optional, including that token-identity alone measured 0/9 against a real
  defect — is recorded in `docs/architecture/tolerance-bound-shape.md`. The
  numbers still belong to whoever measures the backend.
- It does not touch `ENGINE_PIN`, which still names the oracle. That field's
  name remains misleading and is left alone deliberately: renaming it is a
  cascade of its own and buys nothing while the pin is still the parity baseline.
- It does not revisit host identity or worker width, both of which 0002 settled.
