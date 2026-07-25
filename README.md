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
[lane] deterministic | engine pin b4e3a9056567ed8145fc4fa29850d6f1f261ac2b | config vector sha256 30d77c260803 | admission sha256 45121fb83fef | model sha256 3f8a1c04b7e2 | host macos/aarch64 cores=8 simd=dotprod+i8mm+neon | worker threads 8
[lane] model /srv/models/Llama-3.2-1B-Instruct-Q8_0.gguf
[lane] listening on http://127.0.0.1:8181
[lane] loading model; nothing is served until the load completes
[lane] model loaded as 'Llama 3.2 1B Instruct'; replica ready

$ curl -s http://127.0.0.1:8181/v1/chat/completions -d '{ … }' \
    | jq '{camelid_lane, camelid_config_sha256, camelid_admission_sha256, camelid_model_sha256, camelid_worker_threads}'
{
  "camelid_lane": "deterministic",
  "camelid_config_sha256": "30d77c260803",
  "camelid_admission_sha256": "45121fb83fef",
  "camelid_model_sha256": "3f8a1c04b7e2",
  "camelid_worker_threads": 8
}
```

## Why Camelid Enterprise

LLM serving stacks quietly trade reproducibility for performance. Batching, speculative decoding, and per-deployment kernel tuning all change the numerics under a request, so the same prompt can produce different output depending on load, neighbors, and flags — and nothing in the response tells you which you got. For chat, that's fine. For evaluations, regression testing, caching, audit trails, and regulated workloads, it isn't.

- **Reproducible output, on purpose.** On the deterministic lane, the same greedy request yields the identical token stream on every run — including across process restarts.
- **Execution posture is declared, not accidental.** A replica declares its lane at startup; the response carries which one produced it, under which configuration, from which weights, on which machine, at which pool width.
- **Fails closed.** A configuration that would move a replica off its declared posture is a startup error, not a silent degradation. Admission is deny-by-default: an environment variable nobody wrote a rule for is refused, rather than waved through because nobody listed it as dangerous — and the admission policy itself is published as a digest, so a client can tell two builds apart by what they would refuse.
- **Attribution everywhere.** Every response is tagged in headers, in the completion body, and in an optional audit receipt.
- **A serving surface, not an application.** The replica serves generation, model listing and health, and refuses the rest of the engine's router. It also refuses a generation request that names weights other than its own, because the engine resolves that field against the filesystem — withholding the model-management routes is necessary and, on its own, is not enough.
- **Scales like a stateless service.** One replica serves one model; capacity is replica count. Docker and Kubernetes manifests are in the box.

## Lanes

A replica declares its lane at startup, and every response is attributable to it.

| Lane | Status | What it delivers |
|---|---|---|
| **`deterministic`** | ✅ **Shipped** | Reproducible greedy requests: the same request yields the identical token stream on every run, within one hardware class, worker-pool width and configuration. |
| **`throughput`** | 🚧 Planned | Continuously batched execution, tuned for aggregate throughput rather than per-request reproducibility. |

## How the deterministic lane works

- **Pinned engine.** The engine is pinned by exact revision in `Cargo.toml`. What's serving is never "whatever was latest."
- **Frozen configuration.** At startup the lane applies a canonical configuration vector — the order-stable CPU forward pass, speculation off, performance tunables at their defaults — then hashes it (SHA-256). The hash travels with every response, so a replica's exact posture is legible from the outside.
- **Identified weights.** The GGUF is hashed whole before the port is bound, and its digest rides on every response. Two replicas serving different files are told apart from the outside, which the engine's own model *name* cannot do.
- **One generation at a time.** Requests execute whole-generation serialized, so output never depends on what else is in flight.
- **A fixed surface.** The served routes are an allow list; the engine's model-management and runtime-control routes are refused. So is a generation request whose `model` field names anything but this replica's own weights — the engine would otherwise load that file on demand and answer from it. The two together are what make the published model digest a claim about the whole process lifetime rather than about its first second.
- **Fail closed.** An unrecognized `CAMELID_*` variable refuses startup; a full queue returns a typed `503` with `Retry-After`. There is no silent fallback to a faster, weaker execution mode.

## Quick start

> **Before you begin.** Model files are large — roughly 1–8 GB each. Give yourself some free disk space and a few minutes for the first model to download. The first build also fetches and compiles the pinned engine, so it is slower than later builds.

```bash
# Build the serving binary
cargo build --release --bin camelid-enterprise

# Serve a local GGUF model
./target/release/camelid-enterprise serve --model /path/to/model.gguf

# In another process, put the transparent gateway in front of the replica
cargo run --release --bin camelid-enterprise-gateway -- serve \
  --upstream http://127.0.0.1:8181
```

The replica exposes an OpenAI-compatible API on `127.0.0.1:8181` by default. The served surface is an allow list, and this is all of it — `POST /v1/chat/completions`, `POST /v1/completions`, `GET /v1/models`, `GET /v1/models/<id>`, `GET /health` and `GET /v1/health`, plus `HEAD` and CORS preflight where they apply. Everything else on the engine's router answers `403` with `"code":"route_not_served"`.

On the two generation routes the `model` field is checked as well as the path. It may name this replica's own weights — the id the engine filed them under, the path the replica was started with, or that file's name or stem — or it may be omitted. Anything else answers `404` with `"code":"model_not_served"`, whether or not a file of that name exists on disk.

With the gateway running, clients use `127.0.0.1:8080`; it preserves streaming bodies, status codes, and replica attribution without inspecting or retrying inference requests.

> **Listening is not readiness.** The model is read whole and hashed *before* the port is bound — a replica must never answer a request without being able to say what produced it — and then, with the port bound but nothing served from it, the same file is loaded. On a large GGUF from cold cache that is tens of seconds of quiet between `listening` and `replica ready`. Probe `GET /v1/health` for `"generation_ready":true`; a bare TCP-connect check would report ready far too early.

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
> `serve --addr 0.0.0.0:8181` makes the API reachable by every device that can reach the host. The served-route allow list bounds *what* a caller may ask for, never *who* may ask — there is no authentication here, and the gateway does not add any in this release. Only bind `0.0.0.0` on a trusted network, behind your own access controls.

## Attribution

Every response is attributable to the lane that produced it, in three places so no consumer misses it:

| Location | Fields |
|---|---|
| **Response headers** (streams included) | `x-camelid-lane`, `x-camelid-config-sha256`, `x-camelid-admission-sha256`, `x-camelid-model-sha256`, `x-camelid-host`, `x-camelid-worker-threads` |
| **Completion body** (non-streaming JSON) | `camelid_lane`, `camelid_config_sha256`, `camelid_admission_sha256`, `camelid_model_sha256`, `camelid_worker_threads` |
| **Serving receipt** (opt-in, `--serving-receipts <path>`) | one JSONL line per request, digests at full length |

```json
{"admission_sha256":"45121fb83fef631f8464c32dada6100b23f0a0af80347031f812803ee9ec2a09","config_sha256":"30d77c2608036f8475372ace9ec125ffc5fa16d8d63f0355a08c32c69f4449b7","host":"macos/aarch64 cores=8 simd=dotprod+i8mm+neon","lane":"deterministic","method":"POST","model_sha256":"3f8a1c04b7e2d95608f31a7c4be0d2593a6e18cf7b402d95e6c1380af472bb19","path":"/v1/chat/completions","status":200,"ts":1784845685.882345,"worker_threads":8}
```

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

Admission is not a promise that a permitted variable cannot move the numerics:
`CAMELID_ENTERPRISE_THREADS` plainly does. It is permitted because the width it
produces is read back from the pool and published on every response. The rule
both lists share is that **a lever this replica publishes is one a client can
check; a lever it neither publishes nor refuses is the hole.**

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

# Kubernetes
kubectl apply -f deploy/k8s/deployment.yaml -f deploy/k8s/service.yaml \
    -f deploy/k8s/gateway-deployment.yaml -f deploy/k8s/gateway-service.yaml
```

Bare-metal Apple Silicon hosts run the binaries directly; [deploy/macos/](deploy/macos/README.md) has the launchd units and the readiness and drain procedures.

See [deploy/README.md](deploy/README.md) for the full scaling model, probe configuration, the drain sequence, and sizing guidance. Two things there are easy to miss and expensive to rediscover: a replica's drain time is the **sum** of every generation it has already accepted, not the longest one; and the Kubernetes replica pod spec must keep `enableServiceLinks: false`, or every rollout after the first refuses to start.

## Scope

Reproducibility holds for greedy decoding (`temperature: 0`), per hardware class, worker-pool width and configuration vector, across process restarts. It does not extend across different hardware classes, thread counts, weights, or engine revisions.

The scope is drawn conservatively on purpose — where the results are known to hold rather than one step past it. Thread count is the honest example: the engine documents its prefill matmul as parallelizing over independent output rows with serial per-row accumulation, so *that kernel* is bit-exact across widths, and the claim is still scoped away from differing widths because the same has not been established for the whole forward pass. A scope is worth something only if it is never widened by assertion.

Five published fields are what make the scope checkable rather than a footnote: the configuration digest for the vector and engine pin, the admission digest for what the replica would have refused, the model digest for the weights, and the host summary and worker width for the machine and the width. A client that cares compares all five.

## Repository layout

The engine is being brought in-tree so platform code never mixes — each platform is its own crate, enforced at the crate boundary rather than by scattered `#[cfg]`.

```
crates/
├── engine-core/      Platform-neutral: GGUF parsing, tokenizer, quant/tensor
│                     kernels, the order-stable reference math. No host code.
├── engine-macos/     Apple Silicon backend — NEON / dot-product kernels.
├── engine-linux/     Linux backend — x86 AVX/VNNI and CUDA (in progress).
├── engine-windows/   Windows backend (in progress).
├── gateway/          Transparent streaming entry point in front of replicas.
└── server/           The lane-attributed serving binary.
deploy/               Docker images, Kubernetes manifests, macOS launchd units.
docs/adr/             Architecture decision records.
```

`engine-core` never inspects the host; anything keyed on detected hardware lives in exactly one platform crate, and the server links only the crate for its target OS. Accelerated kernels are proven bit-identical to the portable reference on real hardware — acceleration delivers speed without changing a single output bit.

## Roadmap

- **Throughput lane** — continuous batching behind the same attribution surface.
- **Per-tenant lane routing** at the gateway.
- **Engine port completion** — the forward pass and decode loop, landing subsystem by subsystem behind the pinned engine until the in-tree engine is stream-identical.
- **Hardware-class-pinned CI** that verifies reproducible results on every change.

## License

Camelid Enterprise is released under the [Apache License 2.0](LICENSE).

[ci-badge]: https://github.com/timtoole02/Camelid-Enterprise/actions/workflows/ci.yml/badge.svg
[ci-workflow]: https://github.com/timtoole02/Camelid-Enterprise/actions/workflows/ci.yml
