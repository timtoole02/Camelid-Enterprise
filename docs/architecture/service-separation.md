# Camelid Enterprise — Service Separation Plan

> **Status: planning document.** This describes where the system is today and the
> service boundaries we intend to build toward. It is a map to follow, not a
> record of finished work. Sections are explicitly labelled **Today** (verified
> against the current tree) or **Target** (the proposed end state). Some target
> services are partially built; the tables state exactly which slices exist.

---

## 1. Why this document exists

Camelid Enterprise turns a single-user local inference engine into a private AI
platform an organization can run — from one box under a desk to a Kubernetes
cluster in a data center — without user data leaving the organization's control.

Almost every enterprise capability (multiple users, authentication, distributed
deployment, production operations) depends on one thing first: **cleanly
separating the current system into services with single responsibilities and
clear data ownership.** This document maps that separation so we can execute it
in reviewable steps instead of one large rewrite.

The guiding constraint, carried over from Camelid: **local-first.** A service
boundary is only acceptable if the whole system can still run entirely inside
one organization's trust boundary — on a single machine or across its own
cluster — with no dependency on an outside provider.

---

## 2. Ground truth — what exists **Today**

Verified against the current repository. This is the entire scope present right
now; the applications named in the project vision (WebUI, desktop app, agentic
terminal, Kanban agents) are **not in this repository**.

### 2.1 Repository layout (present)

```
Cargo.toml                     workspace: 8 member crates
crates/
  engine-core/                 platform-neutral engine (gguf, model, tensor,
                               forward, tokenizer, host, error)
  engine-macos/                per-platform kernels + host probe()
  engine-linux/                per-platform kernels + host probe()
  engine-windows/              per-platform kernels + host probe()
  gateway/                     transparent and static-catalog HTTP gateway
  identity/                    token -> opaque principal id primitive (SQLite)
  replica-contract/            shared public HTTP route registry
  server/                      the `camelid-enterprise` serving binary
deploy/
  docker/                      separate replica and gateway images
  k8s/deployment.yaml          one deterministic replica pool, one model
  k8s/service.yaml             ClusterIP service in front of the pool
  k8s/gateway-*.yaml           private gateway Deployment + Service
```

### 2.2 Components that exist Today

| Component | Crate / path | Responsibility (as built) |
|---|---|---|
| **Serving replica** | `crates/server` (`camelid-enterprise` bin) | CLI (`serve`), binds an HTTP listener, applies the deterministic lane, stamps attribution, loads one model at startup. |
| **Lane / config freeze** | `crates/server/src/lane.rs` | Applies a canonical env-var configuration vector, fails closed on any override, publishes its SHA-256. Engine pinned by revision (`ENGINE_PIN`). |
| **Attribution middleware** | `crates/server/src/attribution.rs` | Stamps `x-camelid-lane` / `x-camelid-config-sha256` / `x-camelid-host` on every response, injects fields into completion bodies, writes optional JSONL serving receipts (each carrying the gateway-stamped `request_id` when present, or `null` for direct-to-replica traffic). |
| **Gateway / static catalog** | `crates/gateway` (`camelid-enterprise-gateway` bin) | Two mutually exclusive modes. Transparent `--upstream` mode forwards the full `/v1` allowlist derived from `replica_contract::PUBLIC_ROUTES`; static catalog mode maps operator-configured `--model-route <backend-model-id>=<http://origin>` entries to replica pools. Before binding, catalog mode verifies that each configured id is advertised by its mapped pool's `/v1/models`, so it never accepts a public alias then forwards an invalid unchanged request. It serves model discovery locally and routes only completion/chat requests by a bounded JSON `model` selector; all other public routes are refused rather than sent to an arbitrary pool. Selector work has a separate memory-derived semaphore that queues rather than refuses, a bounded body-read deadline that reclaims a stalled slot, and, with identity enabled, a per-organization cap of half the global capacity, so one tenant cannot hold every global selector slot with incomplete bodies. Both modes preserve opaque streaming responses, filter hop-by-hop headers, retry nothing, bound concurrency, and expose no replica control routes. Auth is optional and runs before catalog selection; a selected `model_id` reaches gateway audit/usage records but never a replica. A catalog maps models to pools, not callers to models: every resolved token may use every catalog entry today. Accepted logs drain on clean shutdown within a bounded per-log deadline. |
| **OpenAI-compatible API** | **external** `camelid::api` (git dep, pinned rev `b4e3a905…`) | The gateway exposes `/v1/health`, model discovery, completions/chat, and the pinned engine's compatibility endpoints. Replica-local `/api` model management remains private. Provided by the pinned engine crate, **not** by this repo. |
| **Engine core** | `crates/engine-core` | GGUF container, model config, tensor/forward/tokenizer types. Host-agnostic. |
| **Platform kernels** | `crates/engine-{macos,linux,windows}` | Runtime CPU feature detection (`probe()`), platform kernels. macOS port landing first; Linux/Windows currently capability-detection only. |
| **Identity primitive** | `crates/identity` | Resolves an opaque bearer token to a principal plus explicitly token-scoped organization, backed by a local SQLite store (hashed tokens only). Operators can create/list organizations, add/remove memberships, and issue an organization-scoped token; removing a membership revokes its scoped tokens. Wired into the gateway as opt-in enforcement; still no RBAC or SSO. |
| **Deployment assets** | `deploy/` | Dockerfile (model mounted at runtime, not baked); K8s Deployment (Guaranteed QoS, one model per pool, startup/readiness probes on `/v1/models`) + Service. |

### 2.3 Properties that exist Today

- **Single model per replica.** One process serves one GGUF, one generation at
  a time. Capacity = replica count.
- **Deterministic lane only.** `throughput` lane is declared as planned in the
  README but not implemented; `serve --lane` rejects anything but
  `deterministic`.
- **Stateless replica.** No persistence beyond the optional append-only receipt
  log. No database.
- **Opt-in local identity.** SQLite-backed users, organizations, memberships,
  organization-scoped bearer tokens, expiry, rotation, and revocation exist.
  Gateway enforcement is optional and there is still no RBAC, session, or
  federation layer.
- **Gateway static model routing.** The gateway retains transparent fixed-origin
  mode for one replica or one K8s `Service`, and also has an immutable static
  catalog mode for multiple operator-configured exact backend-model-id-to-pool
  mappings, validated against each pool's `/v1/models` before binding. Catalog
  mode performs bounded JSON selection only for completions/chat, with a
  separate memory-derived selector-work limit that queues rather than refuses
  and, with identity enabled, a per-organization cap of half that capacity;
  it serves model discovery locally and
  otherwise fails closed rather than guessing a pool.
  It has no dynamic catalog, pool-health aggregation, per-organization model
  authorization, or durable shared control-plane state.

---

## 3. Honest gaps — what does **not** exist yet

So the plan never drifts into assuming work is done, this is the explicit list of
what the vision requires that is absent today:

- **No identity layer by default.** `crates/identity` plus gateway bearer-token
  enforcement exist, but enforcement is opt-in (`--identity-db`) and off by
  default. Local users, organizations, memberships, and organization-scoped
  tokens exist; RBAC, sessions, federation, and policy-driven tenant isolation
  do not.
- **No complete multi-tenant authorization model.** Organization membership and
  scoped tokens identify a tenant, but there are no roles, resource permissions,
  or application-data isolation policies.
- **No authorization.** No roles, permissions, or per-user quotas.
- **No complete model-routing control plane or durable metering.** Static
  model-to-pool routing, admission control, per-organization request quotas,
  and raw terminal transport accounting exist, but the catalog is immutable at
  process start; it has no dynamic registration, health aggregation, failover,
  or per-organization model policy. Quota state is per-process and byte counts
  are not model-token accounting.
- **No complete model management service.** A gateway static catalog maps public
  model ids to pools, but models are still mounted by path; there is no durable
  registry, upload, lifecycle API, or operator workflow beyond replica-local
  `/api/models/load`.
- **No application tier.** WebUI, desktop app, agentic terminal, and Kanban
  agent system are named in the vision but are not in this repository. Their
  internals are unknown from this tree and must not be assumed here.
- **No platform datastore.** Identity has a local SQLite database, but the
  decided PostgreSQL store for aggregated audit/usage records, metering rollups,
  and shared quota state is not built.
- **No observability stack.** Replica receipts, gateway audit/usage JSONL, and
  tracing exist, but there is no durable aggregation, metrics backend,
  centralized query surface, or health aggregation across replicas.
- **No single-box packaging.** No "run the whole platform on one machine"
  distribution (compose/bundle) that includes identity + gateway + apps.

---

## 4. The separation principle

We split along **responsibility and data ownership**, not along convenience.
Each service must be able to state, in one sentence, what it owns and what it
must never touch. A boundary earns its place only if:

1. It has a single, nameable responsibility.
2. It owns its data and no other service reaches into that data directly.
3. It can be deployed and scaled independently.
4. It preserves local-first: the whole set can run inside one trust boundary,
   on one machine or one cluster, with no external dependency.
5. It does not weaken the properties that already exist Today (determinism,
   attribution, fail-closed configuration).

A key rule from the current design carries forward: **inference replicas stay
stateless and single-purpose.** Identity, routing, and application state live
*above* the replica, never inside it. The replica's job is "produce attributed
tokens for one model"; everything multi-user is layered on top.

---

## 5. Target service map (proposed)

> **Target end state.** Names are working labels; the table below identifies
> the slices that already exist.

```
                         ┌──────────────────────────────┐
   users / clients ────► │        Gateway / Control      │  auth, routing,
                         │           Plane               │  quotas, metering
                         └───────┬───────────────┬───────┘
                                 │               │
                   ┌─────────────▼──┐      ┌─────▼───────────────┐
                   │  Identity &    │      │  Model / Catalog     │
                   │  Auth Service  │      │  Service             │
                   │  (users, orgs, │      │  (registry, load,    │
                   │   tokens)      │      │   lifecycle)         │
                   └─────────┬──────┘      └─────┬───────────────┘
                             │                   │
                             │            ┌──────▼───────────────┐
                             │            │  Inference Replica    │  ◄── exists Today
                             │            │  Pool(s)              │      (crates/server)
                             │            │  deterministic lane,  │
                             │            │  one model / replica, │
                             │            │  attributed, stateless│
                             │            └──────────────────────┘
                   ┌─────────▼───────────────────────────────────┐
                   │  Application Tier (external today)            │
                   │  WebUI · Desktop app · Agentic terminal ·     │
                   │  Kanban agent system                          │
                   └───────────────────────────────────────────────┘
                   ┌───────────────────────────────────────────────┐
                   │  Platform data + observability                 │
                   │  (audit/usage DB, metering, shared quotas,      │
                   │   receipts, metrics, logs)                      │
                   └───────────────────────────────────────────────┘
```

### 5.1 Proposed services and their boundaries

| Service | Owns | Must never | Exists today? |
|---|---|---|---|
| **Inference Replica Pool** | Producing attributed tokens for exactly one model, one generation at a time; lane guarantee; attribution stamping. | Know about users, auth, or other models. Hold cross-request state. | **Yes** — `crates/server`. Reuse as-is. |
| **Gateway / Control Plane** | Terminating client connections, authenticating requests, routing to the right model pool, quotas, rate limiting, usage metering. | Run inference. Store user credentials (delegates to Identity). | **Partial** — transparent forwarding, immutable static exact-backend-id-to-pool routing with pre-bind verification, memory-bounded selector work, and an authenticated per-organization selector cap; admission control, opt-in bearer auth, per-organization request quotas, audit records, and raw terminal transport accounting exist. Dynamic catalog management, pool-health aggregation, per-organization model policy, durable shared quota state, and model-token metering do not. |
| **Identity & Auth Service** | Users, orgs/teams, credentials, sessions, API tokens, roles/permissions. | Route inference or store conversation content. | **Partial** — local users, organizations, memberships, organization-scoped tokens, expiry, rotation, and revocation exist; no roles, sessions, remote refresh, or federation. |
| **Model / Catalog Service** | Registry of available models, their files, and lifecycle (register, load target, retire); mapping model name → replica pool. | Serve inference itself. Own user data. | **Initial static slice** — the gateway owns an immutable startup catalog of exact backend model id → pool mappings and local discovery; pre-bind verification proves the id exists in its pool. There is no durable registry, lifecycle, health aggregation, or dynamic reload. |
| **Application Tier** | End-user experiences: WebUI, desktop app, agentic terminal, Kanban agents. | Bypass the gateway to reach replicas directly. | **External** — not in this repo. |
| **Platform Data + Observability** | Aggregated audit/usage records, metering rollups, shared quota state, receipts, metrics, and logs. | Own identity records. Be reached directly by replicas or clients. | **Not built** — raw per-pod gateway logs, per-replica receipts, and stderr tracing exist, but the decided PostgreSQL aggregation store does not. |

### 5.2 Boundaries that must NOT move

- The replica stays **stateless and single-model**. Multi-tenancy is a
  gateway/identity concern, never pushed into the replica.
- **Attribution stays at the replica**, because only the replica knows the lane,
  config vector, and host that produced a response. The gateway may correlate
  *its own* identity-bearing records to a receipt — via the opaque
  `x-camelid-request-id` it stamps and the replica echoes into that receipt —
  but must not become the source of attribution, and identity never enters the
  replica's receipt or config vector.
- **Determinism and fail-closed config** remain replica-local invariants and are
  not relaxed to make routing easier.

---

## 6. Proposed migration path (phased, reviewable)

> Each phase is independently shippable and preserves what already works.
> Status below is verified against the current tree.

1. **Baseline & contracts — implemented.**
  `camelid-enterprise-replica-http-v1` separates the contractual `/v1` surface
  from the pinned engine's private implementation inventory. Its registry is
  dependency-free so replicas and gateways can share it, and is checked against
  the exact pinned router without invoking handlers; no-model tests cover
  health, discovery, typed errors, and attribution. An explicit
  model-backed test covers load, readiness, deterministic greedy output, and
  SSE with a compatible local GGUF, plus queue saturation, typed backpressure,
  and depth recovery. Both sides now consume the shared registry: the replica
  verifies its pinned router against it, and the gateway derives its forwarded
  allowlist from `PUBLIC_ROUTES` (with a test asserting it forwards exactly the
  contractual routes and methods), so neither can drift from the contract
  unnoticed.
2. **Gateway (pass-through first) — built.** A transparent fixed-origin gateway
  fronts the existing replica pool with no inference behavior change. It ships
  as a Rust binary, separate container, and private Kubernetes Service.
3. **Identity & auth — in progress.** `crates/identity` provides the
  token -> opaque principal plus token-scoped organization primitive
  (SQLite-backed, hashed tokens, no RBAC/SSO yet). The gateway now enforces it:
  `serve --identity-db <path>`
  rejects any request without a valid `Authorization: Bearer <token>` with a
  typed `401` before it reaches a replica; omitting the flag keeps the
  gateway's original unauthenticated pass-through unchanged. `create-user`,
  `list-users`,
  `create-organization`, `list-organizations`, `add-principal-to-organization`,
  `remove-principal-from-organization`,
  `issue-token [--organization <id>] [--expires-in-seconds <n>]`,
  `rotate-token [--expires-in-seconds <n> | --no-expiry]`,
  and `revoke-token` subcommands manage the local database. The gateway audit
  record includes the resolved opaque organization, but replicas still receive
  only the opaque request id. The identity schema migration is forward-only:
  take a backup before upgrading. A gateway binary older than the database it
  is pointed at refuses to open it at all rather than operating on a schema it
  does not understand, so a rollback after an upgrade fails closed at startup
  instead of silently losing the columns it cannot see.
  **Bearer tokens are only as safe as the transport they travel over:** this
  gateway does not terminate TLS, and a plaintext HTTP hop lets any on-path
  observer capture and replay one. Enabling
  `--identity-db` without a TLS-terminating ingress/reverse proxy (or mTLS, or
  a genuinely trusted private network) in front of it is not a secure
  deployment; the gateway logs a warning to this effect at startup. A token now
  *may* carry an expiry, which bounds how long a captured one is worth
  replaying, and `rotate-token` exchanges a live token for a fresh secret in a
  single transaction — no window in which both work, none in which neither
  does. Both are opt-in and neither is a substitute for transport security:
  a token issued without `--expires-in-seconds` never expires (the behavior of
  every token issued before this existed, and therefore of every token an
  upgrade migrates), expiry is evaluated against the local system clock so a
  backwards clock jump extends every outstanding token, and an expired token is
  still valid for the whole window before it lapses. Revocation remains the
  only control that takes effect immediately and does not consult a clock.
  Four further bounds, stated because they are easy to assume away:
  **rotation is operator-only** — it requires the plaintext token *and*
  filesystem access to the identity database, so a remote client told
  `401 {"type": "token_expired"}` has no shipped way to obtain a replacement;
  the distinct type and the RFC 6750 `WWW-Authenticate` challenge exist so a
  client can stop retrying a dead credential and an operator can see which
  refusal happened (the audit record carries a `reason`), not because a refresh
  endpoint exists. **Expiry is checked at admission
  only**, so a stream that began before its token lapsed runs to completion,
  bounded by `--max-connection-seconds` (300s default) rather than by the
  token. **Lapsed tokens are not swept** — an expired row survives until it is
  rotated, revoked, or its membership is removed, because resolution keeps
  reading it in order to answer "expired" instead of "invalid"; short lifetimes
  accumulate rows, and pruning is left to a change with a retention policy
  behind it rather than folded into a read path where replaying a dead token
  would force a write. **Creating the database concurrently is not supported**:
  schema migration is serialized under a write lock, but the initial
  `journal_mode` switch to WAL in a brand-new file is not, so several processes
  opening a database that does not yet exist can collide. Create it once — any
  CLI subcommand does — before starting processes that share it.
  Rotation defaults to reissuing with the lifetime the presented token was
  *issued* with, measured from now: the commonest reason to rotate is an expiry
  approaching, and the safe result of that is another bounded credential rather
  than a permanent one. `--no-expiry` drops the bound and has to be asked for.
  Per-request
  identity now reaches an audit trail without the replica learning identity:
  the gateway stamps an opaque `x-camelid-request-id` on each forwarded
  request, records `{ts, request_id, principal, organization, reason, method, path, model_id, status}` to its
  own optional `--audit-log`, and the replica echoes that id into its serving
  receipt — so `gateway_audit ⨝ replica_receipt ON request_id` reconstructs
  "which principal's request was served by which deterministic configuration."
  Two honest bounds on that log: the audited `status` is the response *head*,
  not stream completion, so it is a correlation and request-initiation record,
  not a metering substrate — it cannot distinguish a full generation from one
  truncated mid-stream. And the replica echoes the correlation id verbatim, so
  join integrity rests on replica network isolation (a client able to reach the
  replica directly can forge one), not on anything cryptographic.
  Still missing: no way to require auth by default, no automatic or
  policy-enforced rotation (an operator must run `rotate-token`), no minimum or
  default token lifetime, and no remote credential-refresh endpoint. Static
  catalog routing is described under phases 4–5; it does not yet attach a model
  entitlement to an organization. Per-organization request quotas landed
  separately and are described under phase 4.

4. **Multi-user routing & quotas — in progress.** The gateway enforces an
  optional per-organization request-rate quota (`--org-request-quota` /
  `--org-request-quota-window-seconds`, requiring `--identity-db`): a
  fixed-window counter, keyed by the organization resolved during
  authentication, rejects a request over budget with a typed `429` and a
  `Retry-After` header before it consumes an admission permit or reaches a
  replica, without charging requests that fail authentication. A successfully
  authenticated request is charged before admission and forwarding, so a
  gateway `503` or upstream `502` also counts: quota is a bound on attempted
  gateway work, not a record of completed inference. Authentication must query
  SQLite before the organization is known, so the quota does not bound
  identity-store lookup load from a valid over-budget token; token caching is a
  separate revocation-sensitive design problem. The counter is in-memory and
  per-process — it resets on restart and is not shared across gateway replicas
  behind the same Service, which is an explicit trade-off (a coarse per-tenant
  cap, not a durable metering/billing substrate). The shipped gateway manifest
  has two replicas: each process can admit under `2 × limit` in a short burst
  across a fixed-window boundary, so the two-pod deployment can admit under
  `4 × limit` for one organization in that span, distributed nondeterministically
  by Kubernetes. With `--usage-log` plus `--identity-db`, each authenticated,
  quota-admitted request also produces a separate best-effort terminal JSONL
  record with `started_ts`, `duration_ms`, `response_head_status`, opaque
  request/response byte counts, and a `stream_outcome`: `completed`,
  `body_error`, `gateway_timeout`, or cause-agnostic `incomplete`. A request
  byte count is `null` unless the forwarded body reached EOF; zero means the
  gateway observed a complete empty body. It is intentionally not token
  billing: bytes are not tokens, and files remain per-pod until Phase 6
  aggregates them durably. Startup preflights log destinations and rejects
  audit/usage aliases; runtime logging has a bounded dedicated writer queue with
  rate-limited loss warnings. A clean shutdown drains that queue before the
  process exits, bounded by a five-second deadline per log, after which it
  reports how many records were still queued — an upper bound on loss, since
  the writer keeps draining. That drain only happens if the deployment budgets
  for it: the termination grace period has to exceed the connection cap plus
  the per-log deadlines, which the shipped manifest now does and the previous
  300s-against-300s configuration did not. An abrupt crash still loses the
  queue. **Initial static model routing is built:** `serve --model-route
  <backend-model-id>=<http://origin>` is mutually exclusive with `--upstream`;
  before binding it verifies that every configured id is advertised by its
  mapped pool's `/v1/models`, then serves `/v1/models` and
  `/v1/models/{model}` from the immutable map. It routes only
  `/v1/completions` and `/v1/chat/completions` by their required bounded JSON
  `model` field. Selector work has a separate memory-derived semaphore, so
  stalled bodies cannot bypass a bounded number of materializations; the
  default 32 MiB selector budget and 2 MiB request limit allow 8 selectors;
  each reservation covers a raw body and the decoded model id. That capacity
  is a queue rather than a gate: a request waits up to five seconds for a slot
  and is shed with a typed `503` only if the wait expires, because a slot is
  held for the milliseconds a body takes to arrive and refusing on contact
  fails valid requests that merely overlapped. Waiting is bounded on the other
  side too -- a body that does not arrive within fifteen seconds is refused
  with `408` and its slot reclaimed, so slow clients cannot hold the budget for
  the whole connection cap -- and a request whose declared `Content-Length`
  already exceeds the limit is refused with `413` from its head, before it
  takes a slot. With identity enabled, one organization may hold at most half
  the global capacity by default (four slots at the default budget), derived
  from the budget so the invariant survives reconfiguration: no tenant takes
  more than half, and capacity always remains for another one. The selector
  cannot choose an
  origin, selection happens after authentication
  but before quota/inference admission, responses remain streaming, and each
  selected id is added to gateway audit/usage evidence. Requests with malformed,
  missing, unknown, oversized, non-object, or non-`application/*` media-type
  selectors
  reach neither a pool nor a quota counter; failed generations are not retried.
  Catalog mode deliberately returns typed `501` for `/v1/health` and
  compatibility POST routes whose contract does not prove a model selector,
  rather than pretending to aggregate pool health or guessing a destination.
  It is **not** model authorization: every resolved token can use every
  configured entry. Still no state in replicas.
5. **Model/catalog service — initial static routing slice implemented.** The
  gateway now owns an operator-configured, process-lifetime catalog mapping a
  exact backend model id to one replica pool; catalog discovery is stable and
  local. This does not yet promote ad-hoc `--model` + `/api/models/load` into a
  model management service. Still missing: durable registry and lifecycle APIs,
  upload/registration workflow, dynamic reload, pool-health aggregation,
  failover, per-organization model policy, and a durable catalog store. The
  exact current routing contract is `gateway-model-catalog.md`.
6. **Platform data + observability — not started, now unblocked.** Introduce the
  durable aggregation store for audit/usage records, metering rollups, shared
  quota state, receipts, metrics, and logs.
  The store was blocked on an unanswered §7 question; that is now decided as
  self-hosted PostgreSQL (`platform-datastore.md`), which also supplies the
  shared, durable quota state Phase 4 could not provide in-process. Aggregation
  still inherits the gateway logs' best-effort, lossy-on-exit contract: it reads
  what survived and cannot retroactively complete it.
7. **Application tier integration — not started.** Bring the external apps
  (WebUI, desktop, agentic terminal, Kanban) onto the gateway contract —
  reworked for auth and multi-user as the vision requires.
8. **Single-box packaging + cluster parity — partial.** Separate Docker images
  and Kubernetes resources exist for the replica and gateway. A complete
  one-command distribution does not.

The contract is documented at `docs/contracts/replica-http-v1.md`. Future API
guarantees must update its machine registry and evidence; private pinned-engine
routes do not become Enterprise contracts merely because they exist.

---

## 7. Open questions to resolve before building

These are unknowns from the current tree; they must be answered (not assumed)
before the corresponding phase starts.

- Where do the application-tier codebases (WebUI, desktop, agentic terminal,
  Kanban) live, and what API contract do they expect today?
- The gateway is now a Rust service. Before adding control-plane behavior, decide
  whether TLS termination remains external (Ingress/reverse proxy) or belongs in
  the gateway.
- What is the identity model: local accounts only, or federation (OIDC/SAML) for
  organizations — while still allowing a fully offline single-box mode?
- **Resolved.** The platform datastore is **PostgreSQL**, self-hosted inside the
  deployment's trust boundary, as the single backend for both a desk box and a
  cluster — see `platform-datastore.md`. Rejected: SQLite for this role (not
  because it is single-process — it is not — but because multi-pod access needs
  a shared volume, meaning a network filesystem where its locking guarantees do
  not hold, and because `crates/identity` has already produced three
  concurrency defects with two processes on one node), and two backends behind
  one trait (two migration paths and two isolation models, where the failure
  modes that matter surface in whichever is exercised least). "No external
  dependency" constrains who holds the data, not how many processes hold it.
  Phase 6 does not wait on identity: the platform store owns aggregated audit
  and usage records, metering rollups and shared quota state, while principals,
  organizations and tokens stay with identity until that separate question is
  settled — either by migrating it too, or by moving CLI operations onto an
  authenticated gateway admin API so a single process owns the file.
- **Resolved (scoped, deliberately not started).** The pinned engine
  (`camelid` @ `b4e3a905`) stays — see `engine-dependency.md`. `crates/server`
  imports two symbols from it, but they supply the whole HTTP surface: ten
  contractual `/v1` routes plus private model-lifecycle, telemetry, tokenize,
  metrics, embeddings, rerank and agent-workspace routes. `engine-core` is
  ~9,400 lines of numerics that depends only on `serde` and does no HTTP, so
  the gap is the entire serving layer, not the maths. Crucially, the risks that
  matter — availability and air-gapped builds — are fixed by **mirroring or
  vendoring the pinned revision**, not by un-pinning; integrity is already
  sound because the revision is a SHA recorded in `Cargo.lock`. There is no
  incremental migration path today: axum's `Router::merge` panics on a
  duplicate method-and-path pair (`Overlapping method route`, verified against
  0.7.9), `router_with_state` exposes no way to remove a route, and
  `contract.rs` states that axum offers no inverse route-tree introspection, so
  a hybrid router could not be proven exact either. Route-at-a-time needs a
  prerequisite first — composable route groups from the engine, a proxy
  fallback, or a wholesale cutover.
- **Resolved.** Per-request user/tenant context reaches an audit trail without
  the replica learning identity: the gateway mints an opaque, authoritative
  `x-camelid-request-id`, keeps identity in its own append-only audit log, and
  the replica records only that opaque id in its serving receipt. The two logs
  join on `request_id`, and correlation integrity rests on replica network
  isolation (the id is echoed verbatim and forgeable by a direct client), not
  on cryptography. Open follow-on: the audit log records the response *head*
  status only, so a durable **metering** substrate (Phase 6) still needs
  stream-completion accounting and request/response byte counts the audit log
  itself does not capture. An identity-bound optional gateway usage log now
  supplies those raw transport fields and terminal outcomes, but durable
  aggregation, loss handling, and model-token accounting remain Phase 6 work.

---

## 8. How to use this document

- Treat **Section 2** as the source of truth for what exists; update it only when
  code actually lands.
- Treat **Sections 5–6** as the plan; refine boundaries here *before* writing
  code for a phase.
- When a phase completes, move the relevant rows from "Target" to "Today" and
  record what was verified — keeping the honest split between built and planned.
