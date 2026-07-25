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
Cargo.toml                     workspace: 5 member crates
crates/
  engine-core/                 platform-neutral engine (gguf, model, tensor,
                               forward, tokenizer, host, error)
  engine-macos/                per-platform kernels + host probe()
  engine-linux/                per-platform kernels + host probe()
  engine-windows/              per-platform kernels + host probe()
  gateway/                     transparent fixed-origin HTTP gateway
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
| **Attribution middleware** | `crates/server/src/attribution.rs` | Stamps six headers on every response — `x-camelid-lane`, `-config-sha256`, `-admission-sha256`, `-model-sha256`, `-host`, `-worker-threads` — injects the same facts into completion bodies, writes optional JSONL serving receipts with the digests at full length. |
| **Transparent gateway** | `crates/gateway` (`camelid-enterprise-gateway` bin) | Fixed-origin HTTP forwarding with opaque streaming bodies, hop-by-hop header filtering, and no retries or response rewriting. Returns a typed `502` only when it cannot reach the upstream. |
| **OpenAI-compatible API** | **external** `camelid::api` (git dep, pinned rev `b4e3a905…`) | `/v1/chat/completions`, `/v1/completions`, `/v1/models`, `/v1/health` and the engine's own control plane (`/api/models/load` and the rest). Provided by the pinned engine crate, **not** by this repo; the replica serves an allow list over it and refuses everything else, the control plane included (`crates/server/src/surface.rs`). |
| **Engine core** | `crates/engine-core` | GGUF container, model config, tensor/forward/tokenizer types. Host-agnostic. |
| **Platform kernels** | `crates/engine-{macos,linux,windows}` | Runtime CPU feature detection (`probe()`), platform kernels. macOS port landing first; Linux/Windows currently capability-detection only. |
| **Deployment assets** | `deploy/` | Dockerfile (model mounted at runtime, not baked); K8s Deployment (Guaranteed QoS, one model per pool, explicit `--threads`, startup/readiness probes on `/v1/health` for `generation_ready`) + Service. |

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
  quota, or rate limiting.

---

## 3. Honest gaps — what does **not** exist yet

So the plan never drifts into assuming work is done, this is the explicit list of
what the vision requires that is absent today:

- **No authentication / identity.** No login, sessions, tokens, or user store.
- **No multi-user / multi-tenant model.** No concept of a user, team, org, or
  data isolation between them.
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
| **Gateway / Control Plane** | Terminating client connections, authenticating requests, routing to the right model pool, quotas, rate limiting, usage metering. | Run inference. Store user credentials (delegates to Identity). | **Partial** — transparent fixed-origin forwarding exists; control-plane behavior does not. |
| **Identity & Auth Service** | Users, orgs/teams, credentials, sessions, API tokens, roles/permissions. | Route inference or store conversation content. | No. |
| **Model / Catalog Service** | Registry of available models, their files, and lifecycle (register, load target, retire); mapping model name → replica pool. | Serve inference itself. Own user data. | Partial — only `/api/models/load` on the replica exists. |
| **Application Tier** | End-user experiences: WebUI, desktop app, agentic terminal, Kanban agents. | Bypass the gateway to reach replicas directly. | **External** — not in this repo. |
| **Platform Data + Observability** | Durable state (users, conversations, audit trail), receipts aggregation, metrics, logs. | Be reached directly by replicas or by clients. | Partial — only per-replica JSONL receipts + stderr tracing. |

### 5.2 Boundaries that must NOT move

- The replica stays **stateless and single-model**. Multi-tenancy is a
  gateway/identity concern, never pushed into the replica.
- **Attribution stays at the replica**, because only the replica knows the lane,
  config vector, and host that produced a response. The gateway may *add*
  request/user context to receipts but must not become the source of attribution.
- **Determinism and fail-closed config** remain replica-local invariants and are
  not relaxed to make routing easier.

---

## 6. Proposed migration path (phased, reviewable)

> Each phase is independently shippable and preserves what already works.
> Status below is verified against the current tree.

1. **Baseline & contracts — partial.** The gateway contract tests pin method,
  path/query, body, status, attribution, bidirectional streaming, and gateway
  failure behavior. The complete external replica API still needs a dedicated
  contract specification.
2. **Gateway (pass-through first) — built.** A transparent fixed-origin gateway
  fronts the existing replica pool with no inference behavior change. It ships
  as a Rust binary, separate container, and private Kubernetes Service.
3. **Identity & auth — not started.** Add the Identity service and make the
  gateway require authentication. Replicas remain unchanged and unauthenticated
  *behind* the gateway on a private network.
4. **Multi-user routing & quotas — not started.** Gateway routes by model and
  enforces per-user/per-org limits; usage metering begins. Still no state in
  replicas.
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

The remaining baseline work is to write down the exact HTTP contract the replica
  exposes today (endpoints, attribution headers/fields, receipt schema, health
  semantics) as the fixed interface everything above it will depend on.

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
- How is per-request user/tenant context threaded into serving receipts without
  making the replica aware of identity?

---

## 8. How to use this document

- Treat **Section 2** as the source of truth for what exists; update it only when
  code actually lands.
- Treat **Sections 5–6** as the plan; refine boundaries here *before* writing
  code for a phase.
- When a phase completes, move the relevant rows from "Target" to "Today" and
  record what was verified — keeping the honest split between built and planned.
