# The pinned engine dependency

**Status:** migration started behind an internal runtime boundary and an
isolated HTTP adapter; the serving binary still uses the pin. Records what
"un-pin the engine" actually requires, because the phrase is short and the
work is not.

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
`/rerank`, plus ten `/api/agent/workspace/...` routes, eight of which are
thread and session paths.

So the dependency supplies: an OpenAI-compatible API, a model registry and
load lifecycle, streaming, embeddings, reranking, tokenization, a metrics
endpoint, GPU runtime control, a telemetry stream, and an agent workspace
subsystem.

## 4. What is already in-tree

`crates/engine-core` is a substantial numerics and model-lifecycle library and
depends on **`serde` alone** — it does no HTTP at all:

| Area | Lines |
|---|---|
| `tensor/` (blocks, mod, store, q8_dot) | ~3,970 |
| `tokenizer.rs` | 1,723 |
| `gguf.rs` | 832 |
| `forward/` (compute, rope, kv_cache, mod) | ~1,510 |
| `model/` (binding, weights, config, mod) | ~1,230 |
| `runtime.rs` (owned load + raw completion) | ~340 |

`runtime::LoadedModel` now owns the complete in-tree load path: GGUF parsing,
tokenizer construction, configuration, tensor binding, materialized weights,
context admission, and EOG-aware greedy raw completion. It creates a fresh
decoder and KV cache per generation and accepts the per-platform Q8_0 kernel at
one explicit seam. Because the deterministic lane *is* greedy, the absence of
temperature and top-p sampling is not a gap for the lane this product ships.
Per-OS kernels live in `engine-{macos,linux,windows}`.

**The numerics, owned completion lifecycle, and first isolated serving slices
now exist.** `crates/server/src/in_tree.rs` implements health, exact one-model
discovery, and deterministic non-streaming `/v1/completions` and
`/v1/chat/completions` behind a backend trait backed by `LoadedModel`. Chat uses
the GGUF-embedded template with strict Jinja evaluation and explicit
special-token parsing, plus the pinned engine's compact Llama 3 compatibility
shape for rows where that engine deliberately does not evaluate the full
metadata template. A missing or incompatible template is a typed refusal, not
a generic role-prefixed fallback. Both generation paths share one admission
slot and fail closed on unsupported request features. A model-backed parity gate
now compares the exact pinned revision and the in-tree runtime at the prompt-ID,
generated-ID, decoded-text, finish-reason, and usage boundaries across raw,
Unicode, single-turn chat, and multi-turn chat prompts. It unloads the pinned
model before loading the in-tree model so the check stays within hosted-CI
memory. The adapter is deliberately not composed into the serving binary yet,
so no public replica or gateway route has changed.

`LoadedModel` also exposes a synchronous incremental generation boundary for
the next serving slice. It reports each token with only its newly valid UTF-8
text suffix, keeps partial multi-byte characters buffered across tokens, and
accepts cancellation before the next forward pass. A consumer cancellation
returns normally as a distinct outcome rather than detaching inference. This is
engine capability only: no HTTP handler emits SSE yet.

## 5. The gap, stated plainly

Missing entirely, in rough order of difficulty:

- the remaining HTTP contract, including compatibility routes
- the remaining OpenAI request and response schemas beyond deterministic
  string-content completion and chat
- serving model registry and concurrent load/unload/admission policy
- SSE streaming for completions and chat
- embeddings models — a different model family, not a decode loop
- reranking models — likewise
- the agent workspace subsystem

Loading and owning one immutable model for greedy completion is in-tree, as are
pinned-compatible chat rendering, non-streaming HTTP adapters, and an exact
real-model generation parity gate over the current migration artifact;
production composition, attribution, streaming, broader template families, and
the other model families remain above that boundary.

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

## 7. How the migration proceeds

The first slice was intentionally below HTTP: `engine_core::runtime::LoadedModel`
provides one owned, fail-closed path from a GGUF file to deterministic raw
completion. The following slices added an isolated server adapter for health,
exact model discovery, non-streaming raw completion, pinned-compatible chat
completion, the real-model token parity gate over both generation routes, and a
cancellable per-token runtime callback with incremental UTF-8 decoding. The
pinned router remains the production surface while SSE, the remaining handlers,
and broader template families are built and tested against that runtime. This
keeps the external contract unchanged during the port and prevents serving
policy from leaking back into the numerics crate.

There is still no route-at-a-time cutover path with the API as it stands, and
the obvious one does not work. This section records why, so the idea is not
rediscovered and tried.

**`Router::merge` cannot layer an in-tree route over a pinned one.** Axum
rejects a duplicate method-and-path pair by panicking rather than letting the
later registration win. Verified against the vendored axum 0.7.9 by merging two
routers that both declare `GET /v1/health`:

```
Overlapping method route. Handler for `GET /v1/health` already exists
```

So an in-tree `/v1/health` merged with `router_with_state()` is a panic at
startup, not a migration step. And `router_with_state` returns one assembled
`Router` with no hook for removing or replacing a route, so there is nothing to
subtract the pinned handler with first.

**The contract test cannot prove the union is exact, either.** `contract.rs`
says so in its own words: *"Axum exposes no inverse route-tree introspection:
tests prove every declaration exists with these methods."* That direction is
the useful one — a declared route that disappears fails the test — but it
cannot detect a route the engine added. Any claim that a hybrid router
"silently adds nothing" would be unfounded.

A real incremental migration therefore needs a prerequisite that does not exist
yet. Roughly by how invasive they are:

1. **Change the pinned dependency to expose composable route groups** instead
   of one assembled router, so routes can be selected rather than overlaid.
   The only option that makes route-at-a-time genuinely safe, and it requires
   changing the engine — which is what the pin exists to hold still.
2. **Put in-tree routes in front and proxy the remainder**, mounting the
   pinned router as a `fallback` rather than merging it. Avoids the panic
   without touching the engine, at the cost of a second dispatch layer inside
   the replica and a fallback surface nobody enumerates.
3. **Replace the router wholesale** at a single cutover. Forfeits
   incrementality, but is honest that the surface is one unit.

The least-risky route implementation order is still easiest and most
load-bearing first — `/v1/health`, then `/v1/models`, then `/v1/completions`,
then non-streaming `/v1/chat/completions` with templating, then SSE for both
generation routes, then a deliberate decision about `/v1/embeddings` and the
rerank pair, which are different model families and may not belong in this
product's surface at all. None of that sequencing is eligible for production
cutover until one of the three prerequisites above is in place.

## 8. Recommendation

Continue the migration without weakening the pinned production contract:

1. **Keep the server pin in place during the port.** It remains the contract
   anchor until an in-tree surface passes the same model-backed contract.
2. **Build upward from `runtime::LoadedModel`.** Raw and chat completion now
   share the owned lifecycle; SSE and the remaining request schemas follow
   without coupling HTTP into `engine-core`.
3. **Choose a §7 cutover prerequisite before switching any route.** Route-group
   composition or a wholesale router replacement can make the transition
   explicit; overlapping `Router::merge` cannot.
4. **Mirror or vendor the pin independently if air-gapped availability becomes
   urgent.** That mitigates supply availability but is not a substitute for the
   in-tree refactor.
