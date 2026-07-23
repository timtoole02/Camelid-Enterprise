# Camelid-Enterprise

**Deterministic-by-default serving for the [Camelid](https://github.com/timtoole02/Camelid) inference engine.**

Camelid-Enterprise is a serving distribution built on the Camelid engine. It adds
the operational layer that production deployments need and inference engines don't
ship: a declared execution posture per replica (a *lane*), reproducible output you
can hold it to, attribution on every response, and configuration that fails closed
instead of drifting.

## Why

LLM serving stacks quietly trade reproducibility for performance. Batching,
speculative decoding, and per-deployment kernel tuning all change the numerics
under a request, so the same prompt can produce different output depending on
load, neighbors, and flags — and nothing in the response tells you which you got.

For chat, that's fine. For evals, regression testing, caching, audit trails, and
regulated workloads, it isn't. Camelid-Enterprise makes execution posture an
explicit property of a deployment instead of an accident of load, and stamps every
response with the posture that produced it.

## Lanes

A replica declares its lane at startup and every response is attributable to it.

| Lane | Status | What it promises |
|---|---|---|
| `deterministic` | **shipped** | Greedy requests are reproducible: the same request yields the identical token stream on every run, including across process restarts, within one hardware class and configuration. |
| `throughput` | planned | Continuously batched execution for aggregate throughput, with no per-request reproducibility claim. |

## How the deterministic lane works

- **Pinned engine.** The engine dependency is pinned by git revision in
  `Cargo.toml`. What's serving is never "whatever was latest."
- **Frozen configuration.** At startup the lane applies a canonical configuration
  vector: the order-stable CPU forward pass, speculation off, performance
  tunables at their defaults. The vector is hashed (SHA-256) and the hash travels
  with every response. Overriding a pinned key doesn't degrade the guarantee —
  it refuses to start.
- **One generation at a time.** Requests execute whole-generation serialized, so
  output never depends on what else is in flight.
- **Fail closed.** When the bounded queue is full, the replica returns a typed
  `503` with `Retry-After`. There is no silent fallback to a faster, weaker
  execution mode.

## Repository layout

The engine port is organized so platform code never mixes:

```
crates/
  engine-core/      platform-neutral: GGUF container parsing, shared types
  engine-macos/     Apple Silicon backend (in progress — porting first)
  engine-linux/     Linux backend (capability detection; kernels follow macOS)
  engine-windows/   Windows backend (capability detection; kernels follow)
  server/           the lane-attributed serving binary
deploy/             Dockerfile and Kubernetes manifests (see deploy/README.md)
```

`engine-core` never inspects the host; anything keyed on detected hardware
lives in exactly one platform crate, and the server links only the crate for
the target OS. While the port proceeds, serving runs on the pinned upstream
engine; ported modules take over subsystem by subsystem.

## Quickstart

```console
$ cargo build --release
$ ./target/release/camelid-enterprise serve --model /path/to/model.gguf
[lane] deterministic | engine pin b4e3a905… | config vector sha256 30d77c260803
[lane] listening on http://127.0.0.1:8181
[lane] model loaded; replica ready
```

The replica exposes the engine's OpenAI-compatible API:

```console
$ curl -s http://127.0.0.1:8181/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d '{"model": "Llama 3.2 1B Instruct",
         "messages": [{"role": "user", "content": "2+2="}],
         "temperature": 0, "max_tokens": 4}'
```

```json
{
  "choices": [{ "message": { "role": "assistant", "content": "2 + 2 = 4" }, ... }],
  "camelid_lane": "deterministic",
  "camelid_config_sha256": "30d77c260803",
  ...
}
```

Every response also carries `x-camelid-lane` and `x-camelid-config-sha256`
headers (streams included), and `--serving-receipts <path>` appends one JSONL
receipt per request for audit:

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

Engine performance tunables (`CAMELID_*` environment variables) that would move
the deterministic lane off its canonical configuration are rejected at startup by
design.

## Scope of the guarantee

Reproducibility is promised for greedy decoding (`temperature: 0`), per hardware
class and configuration vector, across process restarts. It is deliberately
**not** promised across different hardware classes, thread counts, or engine
revisions — those change the arithmetic, and pretending otherwise is how serving
stacks end up with guarantees nobody can honor. The engine pin plus the config
hash state exactly what a replica vouches for.

## Roadmap

- Throughput lane: continuous batching behind the same attribution surface.
- Per-tenant lane routing at the gateway.
- Hardware-class-pinned CI for the reproducibility guarantee.

## License

Apache-2.0.
