# Engine ownership and the parity oracle

**Status:** production loading, generation, HTTP routing, streaming, queueing,
and cancellation are owned by this workspace. The original Camelid repository
is retained only as a pinned development dependency for behavioral parity and
legacy-contract tests.

## Production boundary

`camelid-enterprise serve` loads one immutable GGUF directly into
`engine_core::runtime::LoadedModel`. It does not construct a Camelid `AppState`,
does not mount the Camelid router, and has no HTTP model-management control
plane.

The production path is:

1. admit and freeze the deterministic-lane environment;
2. resolve the worker pool and host identity;
3. canonicalize and hash the requested GGUF;
4. bind the listener without serving responses;
5. load the GGUF into the in-tree runtime;
6. re-verify the previously hashed file identity;
7. compose the in-tree public router with the served-model and route filters;
8. apply attribution outermost and begin serving.

On macOS the loader supplies `engine_macos::q8_0_dot_rows` through the runtime's
explicit Q8_0 kernel seam, and on Windows `engine_windows::q8_0_dot_rows`. Both
are tested bit-for-bit against the portable implementation; the Windows kernel
routes to AVX2 or to the portable reference itself, so a host without AVX2 takes
the same numbers by a different path. Linux currently uses the portable kernel.

## Owned runtime

`crates/engine-core` owns:

- GGUF metadata and tensor parsing;
- tokenizer construction and incremental UTF-8 decoding;
- model configuration, tensor binding, and weights;
- forward execution and per-request KV caches;
- greedy blocking and incremental generation;
- cancellation before the next forward pass; and
- the platform-kernel injection seam.

The runtime remains synchronous and independent of HTTP. `crates/server` owns
the asynchronous policy around it.

## Owned HTTP surface

`crates/server/src/in_tree.rs` implements all ten routes in
`replica_contract::PUBLIC_ROUTES`:

- health and exact one-model discovery;
- raw and chat completions;
- OpenAI-shaped raw and chat SSE; and
- explicit pinned-compatible `501 not_implemented` responses for embeddings,
  Responses, Messages, rerank, and reranking.

The generation worker runs `--max-concurrency` requests at once (default: the
host's available parallelism) and admits eight more into a bounded queue. The
request past that gets the typed `engine_queue_full` response with
`Retry-After: 1`. `/v1/health` publishes the width as
`engine_generation_slots`, so `engine_queue_depth` can be read against its own
denominator.

Width is a throughput property, not a numeric one. Each generation constructs
its own decoder and KV cache over read-only weights and is never fused with
another, so a request emits the same tokens at any width. That is the line
between a worker pool and batching, and only the former is in this lane. Streaming uses a bounded 32-delta channel; a stalled client
backpressures generation, and a disconnected client cancels it at the next
token boundary.

The served-model middleware preserves the public aliases for the one loaded
model while rewriting them to its GGUF model ID. The route filter refuses every
path and method outside the published contract. Attribution remains outermost,
so successful replies, typed handler failures, and route refusals all identify
the lane, configuration, admission policy, weights, host, and worker width.

## Why the original repository remains

The exact historical revision remains in `dev-dependencies`:

```toml
camelid = { git = "https://github.com/timtoole02/Camelid", rev = "b4e3a9056567ed8145fc4fa29850d6f1f261ac2b" }
```

It is an oracle, not a production engine. The model-backed workflow loads the
same pinned GGUF through each implementation and compares:

- prompt token IDs;
- generated token IDs;
- decoded text;
- finish reason; and
- usage accounting.

Raw, Unicode, single-turn chat, and multi-turn chat cases are covered. The
pinned model is unloaded before the in-tree model is loaded so hosted runners
do not need memory for both simultaneously.

The legacy router tests also preserve exact compatibility-error messages and
the former private-route inventory. This gives future changes a behavioral
reference without putting the old engine in a release binary's normal
dependency graph.

`ENGINE_PIN` remains part of the deterministic configuration digest as the
parity baseline that defined this migration. Renaming or removing that field is
a versioned identity-contract decision, not part of the router cutover.

## Deliberate non-goals

This replica still serves one immutable generative model. It does not add:

- live model load or unload APIs;
- a multi-model in-process registry;
- embeddings or reranking model families;
- Responses or Anthropic request conversion;
- sampling modes outside the deterministic greedy lane; or
- the original application's workspace, telemetry, Web UI, or runtime-control
  routes.

Those capabilities require separate product and contract decisions. Their
absence is explicit on the wire rather than hidden behind a fallback router.

## Build consequence

Normal dependencies contain only workspace-owned engine crates and ordinary
Rust libraries. `cargo tree -p camelid-enterprise --edges normal` contains no
`camelid` package. Tests still resolve the pinned oracle, while production
builds no longer compile or link it.
