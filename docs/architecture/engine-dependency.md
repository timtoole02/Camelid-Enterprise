# The pinned engine dependency

**Status:** scoped, not started. Records what "un-pin the engine" actually
requires, because the phrase is short and the work is not.

`crates/server` depends on the inference engine by git revision:

```toml
camelid = { git = "https://github.com/timtoole02/Camelid", rev = "b4e3a9056567ed8145fc4fa29850d6f1f261ac2b" }
```

This is the last external dependency in a product whose thesis is that an
organization runs the whole thing on its own hardware.

---

## 1. What the pin actually is

- `camelid` v0.3.1, revision `b4e3a905`, dated 2026-07-23.
- Locked in `Cargo.lock` as
  `git+https://github.com/timtoole02/Camelid?rev=b4e3a905…#b4e3a905…`.
- The source repository is public, active, and not archived. Upstream has
  already moved past the pin (`c780eda`, 2026-07-27).

The pin is **deliberate**, not neglect. `crates/server/Cargo.toml` says so:
the deterministic lane's behavior is stated against exactly this engine
version, so the revision is part of the contract. Floating the dependency would
break the property the lane exists to provide.

## 2. What the server uses from it

Two items. That is the whole import surface:

| Symbol | Used at |
|---|---|
| `camelid::api::AppState` | `server/src/lib.rs:21`, `main.rs:86`, `contract.rs:330`, `tests/replica_contract_model.rs:63` |
| `camelid::api::router_with_state` | `server/src/lib.rs:32`, `contract.rs:415` |

Two symbols is misleadingly small. `router_with_state` returns the entire HTTP
surface, and `AppState` is the entire engine runtime behind it.

## 3. What those two symbols actually pull in

**Contractual (public, forwarded by the gateway) — 10 routes:**
`/v1/health`, `/v1/models`, `/v1/models/:model`, `/v1/completions`,
`/v1/chat/completions`, `/v1/embeddings`, `/v1/responses`, `/v1/messages`,
`/v1/rerank`, `/v1/reranking`.

**Private (replica-local, never exposed) — including:** `/health`,
`/api/capabilities`, `/api/runtime/gpu`, `/api/telemetry/stream`,
`/execution-plan`, `/api/execution-plan`, `/api/models/load`,
`/api/models/inspect`, `/tokenize`, `/models`, `/metrics`, `/infill`,
`/rerank`, plus six `/api/agent/workspace/...` thread and session routes.

So the dependency supplies: an OpenAI-compatible API, a model registry and
load lifecycle, streaming, embeddings, reranking, tokenization, a metrics
endpoint, GPU runtime control, a telemetry stream, and an agent workspace
subsystem.

## 4. What is already in-tree

`crates/engine-core` is roughly 9,400 lines (about 8,700 excluding blanks) and
depends on **`serde` alone** — it does no HTTP at all. It is a numerics
library, and a substantial one:

| Area | Lines |
|---|---|
| `tensor/` (blocks, mod, store, q8_dot) | ~3,970 |
| `tokenizer.rs` | 1,723 |
| `gguf.rs` | 832 |
| `forward/` (compute, rope, kv_cache, mod) | ~1,510 |
| `model/` (binding, weights, config, mod) | ~1,230 |

`forward::compute` exposes `Decoder::generate(prompt, max_new)` and `argmax`,
so greedy decoding exists. Because the deterministic lane *is* greedy, the
absence of temperature and top-p sampling is not a gap for the lane this
product ships. Per-OS kernels live in `engine-{macos,linux,windows}`.

**The numerics are largely solved. The serving layer does not exist.**

## 5. The gap, stated plainly

Missing entirely, in rough order of difficulty:

- the HTTP layer — every route above
- OpenAI request and response schemas
- model registry, load and lifecycle management
- SSE streaming for completions and chat
- chat templating
- embeddings models — a different model family, not a decode loop
- reranking models — likewise
- the agent workspace subsystem

This is a program of work, not a change. Anyone estimating it from "the server
only imports two symbols" will be wrong by an order of magnitude.

## 6. Risk, honestly assessed

Not all of this is equally urgent, and conflating the parts is how it stays
unaddressed.

| Risk | Assessment |
|---|---|
| **Supply-chain integrity** | **Low.** The revision is a git SHA recorded in `Cargo.lock`. That is content-addressed and tamper-evident: the pin cannot be swapped underneath the build without the lock changing. |
| **Availability** | **The real one.** One repository, one account. If it is made private, renamed, or deleted, nobody can build this product — including operators who already deployed it. There is no vendored copy and no mirror. |
| **Air-gapped builds** | **Real.** A network fetch is required to build. For a product sold on running inside your own trust boundary, "you need GitHub to compile it" is a contradiction worth removing. |
| **Divergence from upstream** | **Not a risk — a feature.** The pin is the contract anchor. Upstream moving is expected and irrelevant until someone deliberately re-pins. |

The important consequence: **the availability and air-gap risks do not require
un-pinning to fix.** They are fixed by vendoring or mirroring the pinned
revision, which is a small, contained change. Un-pinning is a separate,
much larger effort motivated by ownership rather than by risk.

That distinction is the main point of this document. Doing the cheap thing
first buys most of the safety at a fraction of the cost.

## 7. If and when the migration happens

There is a viable incremental path, and the repo already built the safety net
for it without meaning to.

`crates/server/src/contract.rs` verifies the pinned router against
`replica_contract::PUBLIC_ROUTES`, and axum's `Router::merge` allows in-tree
routes and the pinned router to coexist in one service. So the surface can be
migrated **one route at a time**, with the existing contract test proving at
every step that the union still satisfies the published contract exactly — no
route silently lost, none silently added.

A defensible order, easiest and most load-bearing first:

1. `/v1/health` — trivial, and proves the hybrid-router harness works.
2. `/v1/models`, `/v1/models/:model` — needs a real model registry, so this
   converges with the Phase 5 catalog service rather than duplicating it.
3. `/v1/completions` — the first route needing `engine-core` end to end.
   Greedy only, which the deterministic lane already requires.
4. `/v1/chat/completions` — adds chat templating and SSE streaming.
5. `/v1/embeddings`, `/v1/rerank`, `/v1/reranking` — different model families.
   Decide deliberately whether they are re-implemented, left pinned, or
   removed from the contract; they may not belong in this product's surface.
6. Private `/api` routes — decide what is genuinely needed. The agent
   workspace in particular should be justified before it is reimplemented.

## 8. Recommendation

**Do not start the migration now.** Do these instead, in order:

1. **Mirror or vendor the pinned revision.** Small, contained, and it removes
   the only risk here that can strand an operator who already deployed.
2. **Leave the pin in place.** It is a correct contract anchor and there is no
   defect motivating its removal.
3. **Revisit un-pinning when a route needs to change** — when the contract has
   to gain or alter behavior the pinned engine cannot provide. That is the
   point at which owning the surface starts paying for itself, and step 7's
   route-at-a-time path becomes the plan.
