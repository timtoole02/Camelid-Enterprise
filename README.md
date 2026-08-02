> [!CAUTION]
> **Work in progress — not ready for use.** This project is under active
> development and is **not working end to end yet**. Do not try to install, build,
> or deploy it expecting a functioning server. The commands and features below
> describe the intended design and are not all implemented. Watch the repository
> for a first release.

<div align="center">

# 🐪 Camelid Enterprise

**Deterministic-by-default serving for the Camelid inference engine.**

A production serving layer that makes execution posture a declared property of a deployment — reproducible output, attributed on every response, scaled horizontally.

[![CI][ci-badge]][ci-workflow]
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/built_with-Rust-dea584.svg)](https://www.rust-lang.org/)
[![Platforms](https://img.shields.io/badge/platforms-macOS%20%7C%20Linux%20%7C%20Windows-64748b.svg)](#repository-layout)
[![Lane](https://img.shields.io/badge/lane-deterministic-16a34a.svg)](#how-the-deterministic-lane-works)

[Quick start](#quick-start) · [How it works](#how-the-deterministic-lane-works) · [Deployment](deploy/README.md) · [Roadmap](#roadmap)

</div>

---

Camelid Enterprise wraps the [Camelid](https://github.com/timtoole02/Camelid) engine with the operational layer that production deployments need and inference engines don't ship: a declared execution **lane** per replica, output you can hold a replica to, attribution stamped on every response, and configuration that fails closed instead of drifting.

It runs as a single Rust binary serving an OpenAI-compatible API — the same one your clients already speak.

```console
$ camelid-enterprise serve --model /srv/models/Llama-3.2-1B-Instruct-Q8_0.gguf
[lane] deterministic | in-tree engine | parity oracle pin b4e3a9056567ed8145fc4fa29850d6f1f261ac2b | config vector sha256 b62869e99117 | admission sha256 318fb6d65c0f | model sha256 3f8a1c04b7e2 | host macos/aarch64 cores=8 simd=dotprod+i8mm+neon | worker threads 8 | generation slots 8
[lane] model /srv/models/Llama-3.2-1B-Instruct-Q8_0.gguf
[lane] listening on http://127.0.0.1:8181
[lane] loading model; nothing is served until the load completes
[lane] model loaded as 'Llama 3.2 1B Instruct'; replica ready

$ curl -s http://127.0.0.1:8181/v1/chat/completions -d '{ … }' \
    | jq '{camelid_lane, camelid_config_sha256, camelid_admission_sha256, camelid_model_sha256, camelid_host, camelid_worker_threads}'
{
  "camelid_lane": "deterministic",
  "camelid_config_sha256": "b62869e99117",
  "camelid_admission_sha256": "318fb6d65c0f",
  "camelid_model_sha256": "3f8a1c04b7e2",
  "camelid_host": "macos/aarch64 cores=8 simd=dotprod+i8mm+neon",
  "camelid_worker_threads": 8
}
```

## Why Camelid Enterprise

LLM serving stacks quietly trade reproducibility for performance. Batching, speculative decoding, and per-deployment kernel tuning all change the numerics under a request, so the same prompt can produce different output depending on load, neighbors, and flags — and nothing in the response tells you which you got. For chat, that's fine. For evaluations, regression testing, caching, audit trails, and regulated workloads, it isn't.

- **Reproducible output, on purpose.** On the deterministic lane, the same greedy request yields the identical token stream on every run — including across process restarts.
- **Execution posture is declared, not accidental.** A replica declares its lane at startup; the response carries which one produced it, under which configuration, from which weights, on which machine, at which pool width.
- **Fails closed.** A configuration that would move a replica off its declared posture is a startup error, not a silent degradation. Admission is deny-by-default: an environment variable nobody wrote a rule for is refused, rather than waved through because nobody listed it as dangerous — and the admission policy itself is published as a digest, so a client can tell two builds apart by what they would refuse.
- **Attribution everywhere.** Every response is tagged in headers, in the completion body, and in an optional audit receipt.
- **A serving surface, not an application.** The replica serves the routes of a versioned HTTP contract — generation, model listing, health, and the engine's own typed compatibility replies — and refuses the rest of its router, model management and runtime control included. It also refuses a generation request that names weights other than its own, because the engine resolves that field against the filesystem — withholding the model-management routes is necessary and, on its own, is not enough.
- **Scales like a stateless service.** One replica serves one model; capacity is replica count. Docker and Kubernetes manifests are in the box.

## Lanes

A replica declares its lane at startup, and every response is attributable to it.

| Lane | Status | What it delivers |
|---|---|---|
| **`deterministic`** | ✅ **Shipped** | Reproducible greedy requests: the same request yields the identical token stream on every run, within one hardware class, worker-pool width and configuration. |
| **`throughput`** | 🚧 Planned | Continuously batched execution, tuned for aggregate throughput rather than per-request reproducibility. |

## How the deterministic lane works

- **Owned engine, pinned oracle.** The serving runtime lives in this workspace. The original engine remains pinned by exact revision only in `dev-dependencies`, where model-backed parity tests use it as a behavioral oracle.
- **Declared posture, identified engine.** At startup the lane refuses any `CAMELID_*` variable it has no rule for. Every response then carries three separate claims: `camelid_posture` (which forward-pass implementation ran), `camelid_engine_sha256` (which build of it), and `camelid_config_sha256` (the posture plus the parity-oracle pin). They are separate because they move at different rates — the engine digest changes with every engine edit, the config digest only when the posture set or the pin does. [ADR 0004](docs/adr/0004-engine-identity-and-execution-posture.md) defines each preimage.
- **Identified weights.** The GGUF is hashed whole before the port is bound, and its digest rides on every response. Two replicas serving different files are told apart from the outside, which the engine's own model *name* cannot do.
- **Whole-generation execution.** A request is executed end to end and is never fused with another, so its output never depends on what else is in flight. A replica runs several generations at once (`--max-concurrency`, default: host cores) and the width changes throughput only — each generation owns its decoder and KV cache over read-only weights, so the same request emits the same tokens at any width. Batching, which *would* move the output by fusing requests into shared kernel shapes, is the `throughput` lane's job and is deliberately not this one's.
- **A declared surface.** The served routes are an allow list, and it is the contract's own registry rather than a second list this crate keeps privately. The in-tree router has no model-management, runtime-control, workspace, or Web UI fallback. The served-model filter also refuses a generation request whose `model` field names anything but this replica's own weights. Together these controls make the published model digest a claim about the whole process lifetime rather than about its first second.
- **Fail closed.** An unrecognized `CAMELID_*` variable refuses startup; a full queue returns a typed `503` with `Retry-After`. There is no silent fallback to a faster, weaker execution mode.

## Quick start

> **Before you begin.** Model files are large — roughly 1–8 GB each. Give yourself some free disk space and a few minutes for the first model to download. Normal builds compile only the in-tree engine; test builds also fetch the pinned parity oracle.

```bash
# Build the serving binary
cargo build --release --bin camelid-enterprise

# Serve a local GGUF model
./target/release/camelid-enterprise serve --model /path/to/model.gguf

# In another process, put the transparent gateway in front of the replica
cargo run --release --bin camelid-enterprise-gateway -- serve \
  --upstream http://127.0.0.1:8181
```

The replica exposes an OpenAI-compatible API on `127.0.0.1:8181` by default, and its served surface is the versioned [Replica HTTP Contract v1](docs/contracts/replica-http-v1.md). Nothing in this repository keeps a second inventory of that surface: the replica's route filter and the gateway's forwarding table both read the dependency-free registry in `crates/replica-contract`, which is the contract's own machine-readable form, so one edit changes what the replica serves, what the gateway forwards and what the contract publishes. The list is `GET /v1/health`, `GET /v1/models`, `GET /v1/models/<id>`, `POST /v1/completions` and `POST /v1/chat/completions`, plus five compatibility paths the in-tree server answers with a typed `501` — `/v1/embeddings`, `/v1/responses`, `/v1/messages`, `/v1/rerank` and `/v1/reranking` — and `HEAD` and CORS preflight where they apply. Every path or method outside that contract answers `403` with `"code":"route_not_served"`.

On the two generation routes the `model` field is checked as well as the path. It may name this replica's own weights — the GGUF model id, the path the replica was started with, or that file's name or stem — or it may be omitted. Accepted aliases are rewritten to the one loaded model id. Anything else answers `404` with `"code":"model_not_served"`, whether or not a file of that name exists on disk.

The contract document also fixes attribution, health and readiness, typed errors, streaming, receipts, startup and shutdown behaviour, the evidence behind each claim, and what it deliberately leaves unspecified. Its executable conformance tests live in `crates/server`.

With the gateway running, clients use `127.0.0.1:8080`. It forwards the contractual routes and nothing else, preserves streaming bodies, status codes and replica attribution without inspecting or retrying inference requests, and stamps a gateway-authoritative `x-camelid-request-id` on every forwarded request — the correlation id the replica records in its serving receipt.

> **Listening is not readiness.** The model is read whole and hashed *before* the port is bound — a replica must never answer a request without being able to say what produced it — and then, with the port bound but nothing served from it, the same file is loaded. On a large GGUF from cold cache that is tens of seconds of quiet between `listening` and `replica ready`. Probe `GET /v1/health` for `"generation_ready":true`; a bare TCP-connect check would report ready far too early.

### Static model routing

The gateway also supports a static catalog for one pool per model. Choose
exactly one gateway mode at startup: legacy transparent `--upstream`, or one or
more `--model-route <model-id>=<http://replica-pool>` entries. A catalog id is
not an alias: it must exactly equal an id advertised by that pool's
`/v1/models` endpoint. Catalog startup verifies every mapping before binding.

```bash
cargo run --release --bin camelid-enterprise-gateway -- serve \
  --model-route 'Llama 3.2 1B Instruct=http://llama-pool:8181'
```

In catalog mode, `GET /v1/models` and `GET /v1/models/{model}` are served from
the configured catalog. `POST /v1/completions` and
`POST /v1/chat/completions` select their pool from a required JSON `model`
field; the request bytes are then forwarded unchanged and responses still
stream. The catalog is immutable until restart. It is an operator inventory,
not pool readiness: use the replica readiness probes for readiness, and the
gateway's `/healthz` for gateway liveness. See [deploy/README.md](deploy/README.md)
for selector memory/concurrency limits, endpoint restrictions, and the security
model.

```bash
curl http://127.0.0.1:8181/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Llama 3.2 1B Instruct",
    "messages": [{"role": "user", "content": "2+2="}],
    "temperature": 0,
    "max_tokens": 4
  }'
```

> [!WARNING]
> **A replica is an internal service.** It has no authentication and no tenant identity, and `serve --addr 0.0.0.0:8181` makes its API reachable by every device that can reach the host. Client traffic belongs at the gateway, with cluster policy keeping ordinary workloads off the replica port — [`deploy/k8s/replica-network-policy.yaml`](deploy/k8s/replica-network-policy.yaml) is that policy for Kubernetes, and on one box the replica stays on loopback. The route contract and the model check bound *what* an admitted caller may ask for, never *who* may ask; they are defence in depth behind that isolation, not a replacement for it. The gateway can require a bearer token (`serve --identity-db <path>`), but enforcement is opt-in and the gateway terminates no TLS.

## Attribution

Every response is attributable to the lane that produced it, in three places so no consumer misses it:

| Location | Fields |
|---|---|
| **Response headers** (streams included) | `x-camelid-lane`, `x-camelid-config-sha256`, `x-camelid-admission-sha256`, `x-camelid-model-sha256`, `x-camelid-host`, `x-camelid-worker-threads` |
| **Completion body** (non-streaming JSON) | `camelid_lane`, `camelid_config_sha256`, `camelid_admission_sha256`, `camelid_model_sha256`, `camelid_host`, `camelid_worker_threads` |
| **Serving receipt** (opt-in, `--serving-receipts <path>`) | one JSONL line per request, digests at full length, plus the gateway's `request_id` |

```json
{"admission_sha256":"318fb6d65c0fb2cd3630594b08cc70a1bc3ae0bca7b8bd15c121458e651959f6","config_sha256":"b62869e991172aadb0204c526ff41fd7486434320884bda323e36cff6e13b00d","host":"macos/aarch64 cores=8 simd=dotprod+i8mm+neon","lane":"deterministic","method":"POST","model_sha256":"3f8a1c04b7e2d95608f31a7c4be0d2593a6e18cf7b402d95e6c1380af472bb19","path":"/v1/chat/completions","request_id":"req_9f2c1ad4e7b30516","status":200,"ts":1784845685.882345,"worker_threads":8}
```

`request_id` is the one receipt field the replica does not mint. It is the
correlation id the gateway stamps on every request it forwards, echoed here so a
receipt joins to the gateway's own audit line for the same request — which is
where identity lives, and where it stays. It is `null` when a request reaches the
replica without one.

The digests are published at twelve hex characters on the wire — an audit and
comparison handle, not a cryptographic binding — and at all sixty-four on
receipts, which is the copy an audit compares against a file.

These are separate fields because they are separate claims, and none stands in
for another. `config_sha256` is a compile-time constant, identical on every
conforming replica, which is exactly what makes it worth comparing; it therefore
covers neither the machine, nor the pool width, nor the weights.
`admission_sha256` answers a different question — *what would this replica have
refused?* — because under deny-by-default the allow list is the admission
surface, and a build that quietly gained one permit would otherwise publish
exactly what a clean replica publishes. **Two replicas can publish the same
`config_sha256` and still emit different tokens.** That is the digest's scope
rather than a defect, and the other four fields are what distinguish them.
[ADR 0002](docs/adr/0002-replica-identity-surface.md) states each claim and its
limits in full.

## Configuration

```
camelid-enterprise serve [OPTIONS] --model <MODEL>
```

| Option | Environment | Default | Description |
|---|---|---|---|
| `--model <path>` | `CAMELID_ENTERPRISE_MODEL` | — | GGUF model to load at startup. Hashed whole before the port is bound. |
| `--addr <addr>` | `CAMELID_ENTERPRISE_ADDR` | `127.0.0.1:8181` | Bind address. |
| `--lane <lane>` | — | `deterministic` | Serving lane for this replica (per-deployment). |
| `--threads <n>` | `CAMELID_ENTERPRISE_THREADS` | pool default | Sizes this process's data-parallel worker pool. The width it resolves to is read back from the pool and published on every response, so it is part of the replica's declared identity. `0` is refused. |
| `--serving-receipts <path>` | — | off | Append one JSONL serving receipt per request. |

Those three environment names, plus `CAMELID_ENTERPRISE_TEST_MODEL` (read only by
this workspace's own gated tests), are the **entire** `CAMELID_*` allow list.
Admission is deny-by-default across the whole `CAMELID_` prefix: a variable is
refused because no rule admits it, not because it appears on a list of dangerous
names. That inversion is the point. The engine reads hundreds of keys at the
pinned revision and assembles some of their names at runtime, so no list of
forbidden names can ever be complete, and every name such a list missed would be
a knob on the forward pass that left the published digest unchanged.

Three names outside that prefix are refused as well: `VECLIB_MAXIMUM_THREADS`,
which sizes the platform BLAS pool this replica reports nothing about, and
`RAYON_NUM_THREADS` together with its live deprecated alias `RAYON_RS_NUM_CPUS`,
which are a second, undocumented way to size the pool `--threads` owns. Naming
one spelling of a lever and not the other would be a hint rather than a refusal.

Admission is not a promise that a permitted variable cannot move the numerics.
Two of the permitted names plainly do. `CAMELID_ENTERPRISE_MODEL` selects the
weights, and different weights share nothing; `CAMELID_ENTERPRISE_THREADS` sizes
the worker pool, and the guarantee below is scoped *per* width because
bit-exactness across widths has not been established for the whole forward pass.
Both are permitted because what they resolve to is published on every response —
the model digest and the pool width are read back and stamped — not because they
are inert. The rule both lists share is that **a lever this replica
publishes is one a client can check; a lever it neither publishes nor refuses is
the hole.**

A refusal names every offending variable at once, says what each one does, and
prints the allow list, so the answer to "what may I set, then?" comes from the
process that refused you.

## Deployment

The deterministic lane scales **horizontally**: one replica serves one model, one generation at a time, so aggregate throughput is simply `replicas × single-stream throughput`. Route tenants to lanes at the gateway above the Service; the per-response attribution lets the gateway and clients verify what they got.

```bash
# Docker — loopback-published on purpose; see the exposure note above
docker build -f deploy/docker/Dockerfile -t camelid-enterprise:0.1.0 .
docker run -p 127.0.0.1:8181:8181 -v /path/to/models:/models:ro \
    camelid-enterprise:0.1.0 --model /models/model.gguf

# Kubernetes — the network policy is not optional; it is what keeps the
# replica port off the rest of the namespace
kubectl apply -f deploy/k8s/deployment.yaml -f deploy/k8s/service.yaml \
    -f deploy/k8s/gateway-deployment.yaml -f deploy/k8s/gateway-service.yaml \
    -f deploy/k8s/replica-network-policy.yaml
```

Bare-metal Apple Silicon hosts run the binaries directly; [deploy/macos/](deploy/macos/README.md) has the launchd units and the readiness and drain procedures.

See [deploy/README.md](deploy/README.md) for the trust boundary, the full scaling model, probe configuration, the drain sequence, and sizing guidance. Three things there are easy to miss and expensive to rediscover: the replica network policy only filters anything if the cluster CNI enforces NetworkPolicy at all; a replica's drain time is the **sum** of every generation it has already accepted, not the longest one; and the Kubernetes replica pod spec must keep `enableServiceLinks: false`, or every rollout after the first refuses to start.

The supplied gateway Deployment runs two replicas. If you later opt into its
per-organization fixed-window quota, each gateway process has an independent
in-memory counter; a short burst across a window boundary can therefore admit
under `4 ×` the configured limit across the two pods. See
[deploy/README.md](deploy/README.md) for the full quota sizing constraint and
deployment requirements.

### Enabling authentication

Bearer-token auth is opt-in via `--identity-db <path>`. Two things to get right
before the first start:

**Create the database before running more than one process against it.** Any
CLI subcommand creates it. Schema migration is serialized under a write lock,
but the initial `journal_mode=WAL` switch on a file that does not yet exist is
not, so several processes opening a brand-new database at once can collide and
fail to start. This matters directly for the two-replica gateway Deployment
above sharing one volume.

```bash
# Once, before starting the gateway:
camelid-enterprise-gateway create-user --identity-db /var/lib/camelid/identity.sqlite alice
camelid-enterprise-gateway list-users  --identity-db /var/lib/camelid/identity.sqlite
```

**Record the principal id.** `list-users` is the only way back to it. Issue a
credential with a lifetime, and refresh it before it lapses:

```bash
camelid-enterprise-gateway issue-token  --identity-db <db> <principal> --expires-in-seconds 86400
camelid-enterprise-gateway rotate-token --identity-db <db> -   # reads the token from stdin
```

`rotate-token` gives the replacement the same lifetime the presented token was
issued with, so refreshing a credential that is nearing expiry yields another
bounded one; `--expires-in-seconds` sets a different lifetime and `--no-expiry`
removes the bound entirely. An *expired* token cannot be rotated — re-issue
instead, which is
why the principal id has to be recoverable. Rotation needs filesystem access to
the identity database, so a remote client that receives
`401 {"type": "token_expired"}` cannot refresh itself; that is an operator
action today.

## Scope

Reproducibility holds for greedy decoding (`temperature: 0`), per hardware class, worker-pool width and configuration vector, across process restarts. It does not extend across different hardware classes, thread counts, weights, or engine revisions.

The scope is drawn conservatively on purpose — where the results are known to hold rather than one step past it. Thread count is the honest example: the engine documents its prefill matmul as parallelizing over independent output rows with serial per-row accumulation, so *that kernel* is bit-exact across widths, and the claim is still scoped away from differing widths because the same has not been established for the whole forward pass. A scope is worth something only if it is never widened by assertion.

Five published fields are what make the scope checkable rather than a footnote: the configuration digest for the vector and parity baseline, the admission digest for what the replica would have refused, the model digest for the weights, and the host summary and worker width for the machine and the width. A client that cares compares all five.

## Repository layout

The engine is being brought in-tree so platform code never mixes — each platform is its own crate, enforced at the crate boundary rather than by scattered `#[cfg]`.

```
crates/
├── engine-core/      Platform-neutral: GGUF/model lifecycle, tokenizer,
│                     quant/tensor kernels, deterministic decode. No host code.
├── engine-macos/     Apple Silicon backend — NEON / dot-product kernels.
├── engine-linux/     Linux backend — x86 AVX/VNNI and CUDA (in progress).
├── engine-windows/   Windows backend — x86-64 AVX2 Q8_0 kernel.
├── gateway/          Transparent streaming entry point in front of replicas.
├── identity/         Bearer token → opaque principal id, for the gateway.
├── replica-contract/ Dependency-free public route registry, shared by both.
└── server/           The lane-attributed serving binary, and the contract it owns.
deploy/               Docker images, Kubernetes manifests, macOS launchd units.
docs/adr/             Architecture decision records.
docs/contracts/       The versioned replica HTTP contract.
```

`engine-core` never inspects the host; anything keyed on detected hardware lives in exactly one platform crate, and the server links only the crate for its target OS. Accelerated kernels are proven bit-identical to the portable reference on real hardware — acceleration delivers speed without changing a single output bit.

## Roadmap

- **Throughput lane** — continuous batching behind the same attribution surface.
- **Per-tenant lane routing** at the gateway.
- **Additional model families and request schemas** — embeddings, reranking, and richer compatibility APIs as explicit product slices.
- **Hardware-class-pinned CI** that verifies reproducible results on every change.

## License

Camelid Enterprise is released under the [Apache License 2.0](LICENSE).

[ci-badge]: https://github.com/timtoole02/Camelid-Enterprise/actions/workflows/ci.yml/badge.svg
[ci-workflow]: https://github.com/timtoole02/Camelid-Enterprise/actions/workflows/ci.yml
