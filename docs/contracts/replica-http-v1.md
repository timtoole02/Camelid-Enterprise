# Camelid Enterprise Replica HTTP Contract v1

**Contract ID:** `camelid-enterprise-replica-http-v1`

**Pinned Camelid engine:** `b4e3a9056567ed8145fc4fa29850d6f1f261ac2b`

**Lane:** `deterministic`
**Status:** proposed contractual baseline

This document defines the HTTP behavior Camelid Enterprise intentionally carries
forward for one deterministic inference replica. It is not a promise that every
route present in the pinned desktop engine is a stable Enterprise API.

The dependency-free public route registry in `crates/replica-contract` is the
source of truth for contractual paths, methods, and classification, structured so
replicas and gateways share it without linking the inference engine. Both do:
the replica's served-route filter derives its allow list from
`replica_contract::PUBLIC_ROUTES`, and so does the gateway's forwarding table.
Neither declares a second inventory, so a route added here is served and
forwarded from the one edit, and there is no pair of lists that can drift apart
between releases. The private pinned-route inventory and executable conformance
live in `crates/server/src/contract.rs`; they drive the exact pinned
`camelid::api::router_with_state` and do not reimplement the engine API.

Static catalog mode keeps the same default-deny surface but intentionally routes
only the two generation routes with a proven JSON `model` selector; it serves
model discovery locally and refuses the remaining routes rather than inventing
a pool-selection rule. That gateway-specific behavior is documented in
`docs/architecture/gateway-model-catalog.md`, not promoted into this
single-replica contract.

## Evidence labels

- **Executable (no model):** exercised against the pinned engine's router under
  attribution (`camelid_enterprise::attributed_router`) with an empty
  `AppState`. Deliberately *without* the served-surface filters, because these
  rows pin what the **engine** answers; the filters in front of it are pinned
  separately, by the served-stack tests in `crates/server/src/main.rs` and
  `crates/server/src/surface.rs`.
- **Executable (model):** exercised by the ignored model-backed conformance test
  when `CAMELID_ENTERPRISE_TEST_MODEL` names a compatible local GGUF. That test
  drives `camelid_enterprise::replica_router` — the same composition `serve`
  uses, filters included — so its evidence is about the surface a client meets
  and not about a router nothing serves. This test is explicit/manual and is not
  part of the default CI matrix.
- **Pinned source:** verified against the exact engine revision above, but no
  independent Enterprise test drives the condition yet.
- **Unspecified:** deliberately not promised by this contract.

## Trust boundary

A replica is an internal service. It has no authentication or tenant identity.
Clients reach it through the Enterprise gateway; cluster policy must prevent
ordinary workloads from calling replica port `8181` directly.

The gateway may expose only the contractual routes below. Every `/api/*`, legacy
compatibility, workspace, model-lifecycle, and embedded WebUI route is
replica-private even when it exists in the pinned engine.

## Contractual routes

| Method | Path | Purpose | Evidence |
|---|---|---|---|
| `GET` | `/v1/health` | Liveness, model readiness, backend, queue depth | Executable (no model + model) |
| `GET` | `/v1/models` | List loaded models | Executable (no model + model) |
| `GET` | `/v1/models/{model}` | Inspect one loaded model | Executable (no model) |
| `POST` | `/v1/completions` | OpenAI-compatible text completion | Executable (model) |
| `POST` | `/v1/chat/completions` | OpenAI-compatible chat completion | Executable (model) |
| `POST` | `/v1/embeddings` | Explicit unsupported compatibility response | Executable (no model) |
| `POST` | `/v1/responses` | Explicit unsupported compatibility response | Executable (no model) |
| `POST` | `/v1/messages` | Explicit unsupported compatibility response | Executable (no model) |
| `POST` | `/v1/rerank` | Explicit unsupported compatibility response | Executable (no model) |
| `POST` | `/v1/reranking` | Alias of unsupported reranking response | Executable (no model) |

Axum also answers `HEAD` for registered `GET` routes. The pinned router provides
permissive CORS preflight (`Access-Control-Allow-Origin`, methods, and headers
are `*`) without credential support. Neither behavior adds another application
route.

## Request baseline

`POST` bodies use `Content-Type: application/json`.

The model-backed conformance test exercises this minimal request subset:

- Chat: `model`, `messages`, `stream`, `max_tokens`, `temperature`.
- Text completion: `model`, `prompt`, `stream`, `max_tokens`, `temperature`.

This is not a complete required/optional schema declaration. The pinned engine
owns field validation and accepts additional OpenAI fields and Camelid
extensions. Those extensions are implementation inventory, not an Enterprise
compatibility promise in contract v1. Promote field-level semantics here only
with executable conformance coverage and a deliberate contract review.

The deterministic output claim applies to greedy requests (`temperature: 0`)
within one engine revision, frozen configuration vector, hardware class, thread
count, model artifact, and request. It does not promise identical output across
changes to any of those inputs.

## Attribution

Every replica response, including errors and streams, carries:

```text
x-camelid-lane: deterministic
x-camelid-config-sha256: <first 12 hex characters of the config digest>
x-camelid-host: <hardware-class summary>
```

For non-streaming JSON objects returned by `/v1/completions` and
`/v1/chat/completions`, the middleware also adds:

```json
{
  "camelid_lane": "deterministic",
  "camelid_config_sha256": "<12-character digest>"
}
```

Original response fields remain present. Non-object JSON and non-JSON bodies are
not rewritten. SSE bodies are never buffered or rewritten.

Non-streaming JSON completion responses are buffered for attribution with a
64 MiB ceiling. Exceeding that ceiling fails closed with attributed HTTP `500`
and error type `server_error`; no unattributed completion body is emitted.

**Evidence:** executable attribution tests in `crates/server/src/attribution.rs`.

## Health and readiness

`GET /v1/health` always reports process health separately from generation
readiness. With no model loaded, the contractual fields are:

```json
{
  "ok": true,
  "engine": "camelid",
  "loaded_now": false,
  "generation_ready": false,
  "active_model_id": null,
  "backend": "none",
  "engine_queue_depth": 0
}
```

The response contains additional diagnostic fields whose values depend on host
and model and are not pinned by this contract.

A listening socket is not readiness. A replica is ready for client traffic only
when `/v1/models` is non-empty and `/v1/health` reports
`generation_ready: true`. Kubernetes probes use model discovery for this reason.

With no model, `GET /v1/models` returns exactly:

```json
{"object":"list","data":[]}
```

A missing `GET /v1/models/{model}` returns HTTP `404` with error code
`model_not_found`.

**Evidence:** executable no-model tests; model readiness is also covered by the
model-backed conformance test.

## Completion responses

Non-streaming completion response schemas are owned by the pinned Camelid engine
and follow its OpenAI-compatible shapes. Contract v1 additionally guarantees the
Enterprise attribution fields described above.

For `stream: true`, successful generation returns `text/event-stream`:

- each non-terminal `data:` payload is JSON;
- the terminal event is `data: [DONE]`;
- attribution remains in HTTP response headers;
- the attribution middleware does not modify event bytes.

**Evidence:** pinned source plus the explicit model-backed conformance test. The
default CI matrix compiles this test but does not execute model inference.

## Errors and overload

Engine API errors use this envelope:

```json
{
  "error": {
    "message": "<human-readable detail>",
    "type": "<stable category>",
    "code": "<machine-readable code>",
    "param": "<related request field or null>"
  }
}
```

Contractual no-model errors:

| Condition | Status | `error.type` | `error.code` |
|---|---:|---|---|
| Malformed completion JSON | `400` | `invalid_request` | `malformed_json` |
| Model ID is not loaded | `404` | `invalid_request` | `model_not_found` |
| Embeddings unavailable | `501` | `not_implemented` | `unsupported_embeddings` |
| Responses API unavailable | `501` | `not_implemented` | `unsupported_responses` |
| Messages API unavailable | `501` | `not_implemented` | `unsupported_messages` |
| Reranking unavailable | `501` | `not_implemented` | `unsupported_reranking` |

When the bounded generation queue rejects a job, the pinned engine returns HTTP
`503`, error code `engine_queue_full`, and `Retry-After: 1`. This is the
backpressure signal; callers must not assume automatic retry by the gateway.

**Evidence:** malformed JSON and no-model errors are executable without a
model. Queue saturation, `Retry-After`, attributed error shape, and depth
recovery are exercised by the explicit model-backed conformance test.

## Serving receipts

When `--serving-receipts <path>` is configured, the replica schedules one JSONL
append per completed request with:

```json
{
  "ts": 1784845685.88,
  "method": "POST",
  "path": "/v1/chat/completions",
  "status": 200,
  "lane": "deterministic",
  "config_sha256": "<full 64-character digest>",
  "host": "<hardware-class summary>"
}
```

Receipt writes are best-effort and asynchronous. A write/open failure does not
fail the client response. There is no fsync, delivery acknowledgement, tenant
identity, request body, response body, or durability guarantee in contract v1.
Concurrent records are written as whole JSON-plus-newline buffers so lines do
not interleave within one process. Because the write is scheduled on a
background blocking task that is not awaited by the request path, a receipt
still in flight when the process exits (for example on `SIGTERM`) can be lost;
this is consistent with "best-effort, no durability guarantee" but is worth
stating plainly rather than leaving implicit.

**Evidence:** executable append/concurrency tests in the attribution module.

## Startup and shutdown

The replica:

1. rejects any lane other than `deterministic`;
2. applies the frozen environment configuration and fails startup on a conflict;
3. canonicalizes the configured model path before binding;
4. binds the listener;
5. loads the startup model through its private `POST /api/models/load` route;
6. becomes ready only after that load succeeds.

Connection failures during the self-load are retried with bounded backoff. An
HTTP model-load failure exits the process instead of serving a model-less
replica indefinitely.

On Ctrl+C, and on Unix `SIGTERM`, the server stops accepting new connections and
uses Axum graceful shutdown to drain active connections. Deployment drain
windows must be longer than the longest permitted generation.

**Evidence:** production composition source, lane tests, real Docker model-load
and SIGTERM verification. Startup retry/process-exit paths are not unit-tested
because they intentionally terminate the process.

## Replica-private pinned implementation inventory

The complete private inventory is internal to the server contract module. Axum
does not expose inverse route-tree introspection, so the executable test proves
only that every *declared* route still exists with exactly its declared
methods at the pinned engine revision — it catches a declared route being
removed or having its methods change, but it cannot detect the engine adding a
new route at a pin bump, since there is nothing to iterate that isn't already
in the list. Completeness of the 61-route cardinality is therefore established
by source review at each immutable engine revision, not by the test suite; a
new route the reviewer misses is invisible to CI. This is a documentation-
accuracy risk, not a security gap: the gateway is a default-deny allowlist over
ten public routes, so an undiscovered private route is simply never exposed.
Changing the engine revision requires another full source inventory review.
These categories exist at the pinned engine revision but are **not** Enterprise
public contracts:

- Diagnostics: `/health`, `/api/capabilities`, telemetry, execution plans.
- Model/runtime operations: `/api/runtime/gpu`, `/api/models/*`, catalog,
  runnable verification/smoke, generation sessions and preflight.
- Workspace application: `/api/agent/workspace/*`.
- Legacy compatibility: `/tokenize`, `/detokenize`, `/apply-template`,
  `/models*`, `/props`, `/slots`, `/metrics`, `/completion`, `/infill`,
  `/embedding(s)`, and unversioned reranking routes.
- Embedded WebUI fallback for paths not matched by an explicit route.

The route registry executable test pins every explicit path and its methods using
side-effect-free `TRACE`/`Allow` probes. Presence in that inventory does not make
a route safe to expose or stable across future engine revisions.

## Intentionally unspecified

Contract v1 makes no promise about:

- authentication, users, tenants, authorization, or quotas;
- latency, throughput, maximum concurrency, or autoscaling thresholds;
- arbitrary model families, quantizations, or context sizes;
- undocumented Camelid request extensions;
- embedded WebUI assets or client-side routes;
- replica-private route schemas or compatibility across engine revisions;
- request-size limits beyond behavior inherited from the pinned HTTP stack;
- receipt durability or exactly-once delivery;
- automatic retries by gateway or replica.

Any future guarantee must enter the machine registry or this document with an
executable check or an explicitly recorded evidence gap.

## Validation commands

No-model contract:

```console
cargo test -p camelid-enterprise
```

Model-backed contract:

```console
CAMELID_ENTERPRISE_TEST_MODEL=/absolute/path/model.gguf \
  cargo test -p camelid-enterprise --test replica_contract_model \
  real_model_conforms_to_replica_http_v1 -- --ignored --exact --nocapture
```

The GitHub Actions workflow `replica contract - model-backed` downloads one
model artifact from an immutable repository revision, verifies its exact byte
length and SHA-256, and runs that command serially. It runs on pull requests that
change the contract/server ownership surface and by manual dispatch; unrelated
pull requests do not pay the approximately 808 MB download and CPU stress cost.
