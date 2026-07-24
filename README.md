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
$ camelid-enterprise serve --model models/Llama-3.2-1B-Instruct-Q8_0.gguf
[lane] deterministic | engine pin b4e3a905… | config vector sha256 30d77c260803 | host macos/aarch64 cores=8 simd=dotprod+i8mm+neon
[lane] listening on http://127.0.0.1:8181
[lane] model loaded; replica ready

$ curl -s http://127.0.0.1:8181/v1/chat/completions -d '{ … }' | jq '{camelid_lane, camelid_config_sha256}'
{
  "camelid_lane": "deterministic",
  "camelid_config_sha256": "30d77c260803"
}
```

## Why Camelid Enterprise

LLM serving stacks quietly trade reproducibility for performance. Batching, speculative decoding, and per-deployment kernel tuning all change the numerics under a request, so the same prompt can produce different output depending on load, neighbors, and flags — and nothing in the response tells you which you got. For chat, that's fine. For evaluations, regression testing, caching, audit trails, and regulated workloads, it isn't.

- **Reproducible output, on purpose.** On the deterministic lane, the same greedy request yields the identical token stream on every run — including across process restarts.
- **Execution posture is declared, not accidental.** A replica commits to a lane at startup; the response carries which one produced it.
- **Fails closed.** A configuration that would move a replica off its declared posture is a startup error, not a silent degradation.
- **Attribution everywhere.** Every response is tagged in headers, in the completion body, and in an optional audit receipt.
- **Scales like a stateless service.** One replica serves one model; capacity is replica count. Docker and Kubernetes manifests are in the box.

## Lanes

A replica declares its lane at startup, and every response is attributable to it.

| Lane | Status | What it promises |
|---|---|---|
| **`deterministic`** | ✅ **Shipped** | Greedy requests are reproducible: the same request yields the identical token stream on every run, within one hardware class and configuration. |
| **`throughput`** | 🚧 Planned | Continuously batched execution for aggregate throughput, with no per-request reproducibility claim. |

## How the deterministic lane works

- **Pinned engine.** The engine is pinned by exact revision in `Cargo.toml`. What's serving is never "whatever was latest."
- **Frozen configuration.** At startup the lane applies a canonical configuration vector — the order-stable CPU forward pass, speculation off, performance tunables at their defaults — then hashes it (SHA-256). The hash travels with every response, so a replica's exact posture is legible from the outside.
- **One generation at a time.** Requests execute whole-generation serialized, so output never depends on what else is in flight.
- **Fail closed.** Overriding a pinned key refuses startup; a full queue returns a typed `503` with `Retry-After`. There is no silent fallback to a faster, weaker execution mode.

## Quick start

> **Before you begin.** Model files are large — roughly 1–8 GB each. Give yourself some free disk space and a few minutes for the first model to download. The first build also fetches and compiles the pinned engine, so it is slower than later builds.

```bash
# Build the serving binary
cargo build --release --bin camelid-enterprise

# Serve a local GGUF model
./target/release/camelid-enterprise serve --model /path/to/model.gguf
```

The replica exposes the engine's OpenAI-compatible API (`/v1/chat/completions`, `/v1/completions`, `/v1/models`, …) on `127.0.0.1:8181` by default.

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
> `serve --addr 0.0.0.0:8181` makes the API reachable by every device that can reach the host. Only bind `0.0.0.0` on a trusted network, behind your own access controls.

## Attribution

Every response is attributable to the lane that produced it, in three places so no consumer misses it:

| Location | Fields |
|---|---|
| **Response headers** (streams included) | `x-camelid-lane`, `x-camelid-config-sha256` |
| **Completion body** | `camelid_lane`, `camelid_config_sha256` |
| **Serving receipt** (opt-in, `--serving-receipts <path>`) | one JSONL line per request |

```json
{"ts":1784845685.88,"method":"POST","path":"/v1/chat/completions","status":200,"lane":"deterministic","config_sha256":"30d77c26…"}
```

## Configuration

```
camelid-enterprise serve [OPTIONS] --model <MODEL>
```

| Option | Default | Description |
|---|---|---|
| `--model <path>` | — | GGUF model to load at startup. |
| `--addr <addr>` | `127.0.0.1:8181` | Bind address. |
| `--lane <lane>` | `deterministic` | Serving lane for this replica (per-deployment). |
| `--threads <n>` | auto | Worker threads; part of the replica's declared identity. |
| `--serving-receipts <path>` | off | Append one JSONL serving receipt per request. |

Engine performance tunables (`CAMELID_*` environment variables) that would move the deterministic lane off its canonical configuration are rejected at startup by design.

## Deployment

The deterministic lane scales **horizontally**: one replica serves one model, one generation at a time, so aggregate throughput is simply `replicas × single-stream throughput`. Route tenants to lanes at the gateway above the Service; the per-response attribution lets the gateway and clients verify what they got.

```bash
# Docker
docker build -f deploy/docker/Dockerfile -t camelid-enterprise:0.1.0 .
docker run -p 8181:8181 -v /path/to/models:/models:ro \
    camelid-enterprise:0.1.0 --model /models/model.gguf

# Kubernetes
kubectl apply -f deploy/k8s/deployment.yaml -f deploy/k8s/service.yaml
```

See [deploy/README.md](deploy/README.md) for the full scaling model, probe configuration, and sizing guidance.

## Scope of the guarantee

Reproducibility is promised for greedy decoding (`temperature: 0`), per hardware class and configuration vector, across process restarts. It is deliberately **not** promised across different hardware classes, thread counts, or engine revisions — those change the arithmetic, and pretending otherwise is how serving stacks end up with guarantees nobody can honor. The engine pin plus the configuration hash state exactly what a replica vouches for.

## Repository layout

The engine is being brought in-tree so platform code never mixes — each platform is its own crate, enforced at the crate boundary rather than by scattered `#[cfg]`.

```
crates/
├── engine-core/      Platform-neutral: GGUF parsing, tokenizer, quant/tensor
│                     kernels, the order-stable reference math. No host code.
├── engine-macos/     Apple Silicon backend — NEON / dot-product kernels.
├── engine-linux/     Linux backend — x86 AVX/VNNI and CUDA (in progress).
├── engine-windows/   Windows backend (in progress).
└── server/           The lane-attributed serving binary.
deploy/               Dockerfile and Kubernetes manifests.
```

`engine-core` never inspects the host; anything keyed on detected hardware lives in exactly one platform crate, and the server links only the crate for its target OS. Accelerated kernels are proven bit-identical to the portable reference on real hardware — acceleration never perturbs the determinism guarantee.

## Roadmap

- **Throughput lane** — continuous batching behind the same attribution surface.
- **Per-tenant lane routing** at the gateway.
- **Engine port completion** — the forward pass and decode loop, landing subsystem by subsystem behind the pinned engine until the in-tree engine is stream-identical.
- **Hardware-class-pinned CI** for the reproducibility guarantee.

## License

Camelid Enterprise is released under the [Apache License 2.0](LICENSE).

[ci-badge]: https://github.com/timtoole02/Camelid-Enterprise/actions/workflows/ci.yml/badge.svg
[ci-workflow]: https://github.com/timtoole02/Camelid-Enterprise/actions/workflows/ci.yml
