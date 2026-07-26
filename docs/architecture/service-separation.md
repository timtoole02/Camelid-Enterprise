# Camelid Enterprise — Service Separation Plan

> **Status: planning document.** This describes where the system is today and the
> service boundaries we intend to build toward. It is a map to follow, not a
> record of finished work. Sections are explicitly labelled **Today** (verified
> against the current tree) or **Target** (proposed, not yet built). Nothing in
> the **Target** sections is implemented yet.

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
Cargo.toml                     workspace: 6 member crates
crates/
  engine-core/                 platform-neutral engine (gguf, model, tensor,
                               forward, tokenizer, host, error)
  engine-macos/                per-platform kernels + host probe()
  engine-linux/                per-platform kernels + host probe()
  engine-windows/              per-platform kernels + host probe()
  gateway/                     transparent fixed-origin HTTP gateway
  identity/                    token -> opaque principal id primitive (SQLite)
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
| **Transparent gateway** | `crates/gateway` (`camelid-enterprise-gateway` bin) | Fixed-origin forwarding for the `/v1` inference allowlist it derives from `replica_contract::PUBLIC_ROUTES` (so the allowlist cannot silently drift from the replica's public contract), with opaque streaming bodies, hop-by-hop header filtering, no retries, bounded concurrency (admission-controlled), and no response rewriting. Replica control routes are not exposed. Optionally enforces bearer-token auth (see below), checked before the admission permit is taken; unauthenticated pass-through remains the default. Stamps a gateway-authoritative `x-camelid-request-id` on every forwarded request (overwriting any client value) and, with `serve --audit-log <path>`, writes one JSONL audit line per handled request — including auth/admission rejections — as `{ts, request_id, principal, organization, method, path, status}`. Principal and organization remain gateway-local; only the opaque request id reaches a replica. |
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
- **No identity layer.** No authentication, no authorization, no users, no
  tenants, no API keys. The `--addr 0.0.0.0` warning in the README is the only
  access-control guidance; security is entirely "put it on a trusted network."
- **Transparent gateway only.** A fixed-origin gateway fronts one replica or one
  K8s `Service`. There is no per-user or per-model routing, authentication,
  quota, or rate limiting. It rejects non-inference paths and bounds concurrent
  request streams.

---

## 3. Honest gaps — what does **not** exist yet

So the plan never drifts into assuming work is done, this is the explicit list of
what the vision requires that is absent today:

- **No identity layer by default.** `crates/identity` plus gateway bearer-token
  enforcement exist, but enforcement is opt-in (`--identity-db`) and off by
  default; without it there is still no authentication, no users beyond a flat
  local table, and no tenants. No orgs, RBAC, sessions, or SSO/OIDC.
- **No multi-user / multi-tenant model.** No concept of a team, org, or data
  isolation between principals beyond the token → principal mapping itself.
- **No authorization.** No roles, permissions, or per-user quotas.
- **No control-plane behavior.** The transparent gateway exists, but there is no
  request routing by user or model, admission control, rate limiting, or usage
  metering.
- **No model management service.** Models are mounted by path; there is no
  registry, catalog, upload, or lifecycle API beyond `/api/models/load`.
- **No application tier.** WebUI, desktop app, agentic terminal, and Kanban
  agent system are named in the vision but are not in this repository. Their
  internals are unknown from this tree and must not be assumed here.
- **No persistence layer.** No database for users, conversations, audit history,
  or agent state.
- **No observability stack.** Only receipts + tracing to stderr; no metrics,
  centralized logs, or health aggregation across replicas.
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

> **Target — not built yet.** Names are working labels.

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
                   │  (users/conversations/audit DB, receipts,      │
                   │   metrics, logs)                               │
                   └───────────────────────────────────────────────┘
```

### 5.1 Proposed services and their boundaries

| Service | Owns | Must never | Exists today? |
|---|---|---|---|
| **Inference Replica Pool** | Producing attributed tokens for exactly one model, one generation at a time; lane guarantee; attribution stamping. | Know about users, auth, or other models. Hold cross-request state. | **Yes** — `crates/server`. Reuse as-is. |
| **Gateway / Control Plane** | Terminating client connections, authenticating requests, routing to the right model pool, quotas, rate limiting, usage metering. | Run inference. Store user credentials (delegates to Identity). | **Partial** — transparent forwarding plus opt-in bearer-token enforcement exist; routing, quotas, and metering do not. |
| **Identity & Auth Service** | Users, orgs/teams, credentials, sessions, API tokens, roles/permissions. | Route inference or store conversation content. | **Partial** — `crates/identity` resolves tokens to principals and gates the gateway; no orgs, roles, sessions, or federation. |
| **Model / Catalog Service** | Registry of available models, their files, and lifecycle (register, load target, retire); mapping model name → replica pool. | Serve inference itself. Own user data. | Partial — only `/api/models/load` on the replica exists. |
| **Application Tier** | End-user experiences: WebUI, desktop app, agentic terminal, Kanban agents. | Bypass the gateway to reach replicas directly. | **External** — not in this repo. |
| **Platform Data + Observability** | Durable state (users, conversations, audit trail), receipts aggregation, metrics, logs. | Be reached directly by replicas or by clients. | Partial — only per-replica JSONL receipts + stderr tracing. |

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
  `create-organization`, `list-organizations`, `add-principal-to-organization`,
  `remove-principal-from-organization`, `issue-token [--organization <id>]`,
  and `revoke-token` subcommands manage the local database. The gateway audit
  record includes the resolved opaque organization, but replicas still receive
  only the opaque request id. The identity schema migration is forward-only:
  take a backup before upgrading, and do not roll an older gateway binary back
  against the migrated database because it cannot mint new organization-scoped
  tokens.
  **Bearer tokens are only as safe as the transport they travel over:** this
  gateway does not terminate TLS, tokens do not expire, and a plaintext HTTP
  hop lets any on-path observer capture and replay one indefinitely. Enabling
  `--identity-db` without a TLS-terminating ingress/reverse proxy (or mTLS, or
  a genuinely trusted private network) in front of it is not a secure
  deployment; the gateway logs a warning to this effect at startup. Per-request
  identity now reaches an audit trail without the replica learning identity:
  the gateway stamps an opaque `x-camelid-request-id` on each forwarded
  request, records `{ts, request_id, principal, organization, method, path, status}` to its
  own optional `--audit-log`, and the replica echoes that id into its serving
  receipt — so `gateway_audit ⨝ replica_receipt ON request_id` reconstructs
  "which principal's request was served by which deterministic configuration."
  Two honest bounds on that log: the audited `status` is the response *head*,
  not stream completion, so it is a correlation and request-initiation record,
  not a metering substrate — it cannot distinguish a full generation from one
  truncated mid-stream. And the replica echoes the correlation id verbatim, so
  join integrity rests on replica network isolation (a client able to reach the
  replica directly can forge one), not on anything cryptographic.
  Still missing: no way to require auth by default, no token expiry/rotation,
  no routing or quotas.

   **Rebase hazard:** this work is cut from `main` before admission control
   (a bounded in-flight semaphore) lands in the separate gateway-hardening
   PR. When rebased onto that work, the auth check must run *before* the
   admission permit is acquired — otherwise an unauthenticated flood consumes
   permits (and the SQLite lookup time behind each one) before ever being
   rejected, starving legitimate authenticated traffic — and any local
   health-check route that work introduces must stay exempt from auth the
   way it stays exempt from admission.
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
  record with its `response_head_status`, opaque request/response byte counts,
  and `completed`, `body_error`, or `client_disconnect` outcome. It is
  intentionally not token billing: bytes
  are not tokens, records are queued asynchronously and can be lost on process
  exit, and files remain per-pod until Phase 6 aggregates them durably. Still
  missing: routing by model (there is exactly one upstream today). Still no
  state in replicas.
5. **Model/catalog service — not started.** Promote model management out of
  ad-hoc `--model` + `/api/models/load` into a catalog that maps model → pool.
6. **Platform data + observability — not started.** Introduce the durable store
  (users, conversations, audit) and aggregate receipts/metrics/logs.
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
- What datastore satisfies both "one box under a desk" and "data center scale"
  without an external managed dependency?
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
  supplies those raw transport fields and outcomes, but durable aggregation,
  loss handling, and model-token accounting remain Phase 6 work.

---

## 8. How to use this document

- Treat **Section 2** as the source of truth for what exists; update it only when
  code actually lands.
- Treat **Sections 5–6** as the plan; refine boundaries here *before* writing
  code for a phase.
- When a phase completes, move the relevant rows from "Target" to "Today" and
  record what was verified — keeping the honest split between built and planned.
