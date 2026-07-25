# Deployment

- [Scaling model](#scaling-model)
- [What a replica publishes, and what it serves](#what-a-replica-publishes-and-what-it-serves)
- [Starting and draining](#starting-and-draining)
- [Docker](#docker)
- [Kubernetes](#kubernetes)
- [Bare metal](#bare-metal)

## Scaling model

The deterministic lane scales **horizontally**. One replica serves one model,
one generation at a time — that serialization is what makes its output
reproducible, so a replica never gets faster under load, it gets *more
neighbors*. Capacity planning is therefore simple multiplication:

    aggregate throughput = replicas × single-stream throughput

Route tenants to lanes at the gateway above the Service; the per-response
attribution headers let the gateway and clients verify what they got. Keep each
replica pool on **one instance type**, and start every replica in it with the
**same `--threads`**: the lane's behavior is scoped to a hardware class and a
worker-pool width, and neither is covered by the configuration digest, so a pool
that mixes either is really several pools wearing one Service.

The gateway in this release is deliberately transparent: one fixed upstream,
opaque streaming request/response bodies, no retries, and no response rewriting.
On Kubernetes, point it at the replica Service and let Kubernetes balance the
identical pool. On one box, point it directly at the replica. Authentication and
tenant-aware routing have not landed; keep both services on a trusted network.

When a replica's queue is full it returns `503` + `Retry-After`; treat that as
the autoscaling signal (scale on queue-full rate or p95 latency, not CPU — a
serialized replica at steady decode is *supposed* to sit near its CPU limit).

## What a replica publishes, and what it serves

Every response carries six headers, and they answer different questions. A
consumer that compares one of them has checked one of them.

| Header | Answers |
|---|---|
| `x-camelid-lane` | which lane produced this |
| `x-camelid-config-sha256` | is this the configuration I audited? (12 hex; constant across a conforming fleet) |
| `x-camelid-admission-sha256` | would this replica refuse what I think it refuses? (12 hex; constant across a conforming fleet) |
| `x-camelid-model-sha256` | are these the weights I audited? (12 hex; `shasum -a 256` of the GGUF) |
| `x-camelid-host` | is this the hardware class the guarantee is scoped to? |
| `x-camelid-worker-threads` | is this the pool width the pool actually came up at? |

Non-streaming completion bodies carry the same facts as `camelid_lane`,
`camelid_config_sha256`, `camelid_admission_sha256`, `camelid_model_sha256`,
`camelid_host` and `camelid_worker_threads`, and each serving receipt carries them with the digests
at full length. The gateway does not strip them — it removes only hop-by-hop
headers and the names a `Connection` header nominates — so a probe *through* the
gateway still sees the replica's own identity. Its contract tests pin that for
`x-camelid-lane`, `x-camelid-config-sha256` and `x-camelid-host`; the other three
pass through by the same name-agnostic rule but are not yet asserted.

Two replicas may publish the same `config_sha256` and still emit different
tokens, because the digest deliberately does not cover the machine, the width or
the weights. That is the digest's scope, not a defect;
[ADR 0002](../docs/adr/0002-replica-identity-surface.md) states it in full.

**The replica serves an allow list.** `GET`/`HEAD`/`OPTIONS` on `/health`,
`/v1/health`, `/v1/models` and `/v1/models/<id>`; `POST`/`OPTIONS` on
`/v1/completions` and `/v1/chat/completions`. Everything else — including the
engine's unauthenticated model-load, model-unload and runtime-control routes —
answers `403` with `"code":"route_not_served"`. It is not an access-control
layer: it bounds *what* a caller may ask for, never *who* may ask, so keeping the
port private is still the deployment's job.

**A generation request cannot repoint the replica either.** The engine resolves a
request's `model` field against the filesystem before anything else, so on the
two generation routes that field is checked as well as the path: it may name this
replica's own weights or be omitted, and anything else answers `404` with
`"code":"model_not_served"` — identically whether or not a file of that name is
on the mount. Withholding the model-management routes and checking the field are
*together* what make the published model digest a claim about the whole process
lifetime rather than about its first second. This matters most where the manifests
put it: the shipped Kubernetes PVC mounts the whole model pool at `/models`, so
without the body check any replica in the fleet could be pointed at any other
model in it, over a route the path filter admits.

**The replica refuses to start on environment variables it does not recognize.**
Admission is deny-by-default over the whole `CAMELID_` prefix, plus three foreign
names: `VECLIB_MAXIMUM_THREADS`, and `RAYON_NUM_THREADS` with its live deprecated
alias `RAYON_RS_NUM_CPUS`. Four names are permitted:
`CAMELID_ENTERPRISE_MODEL`, `CAMELID_ENTERPRISE_ADDR`,
`CAMELID_ENTERPRISE_THREADS` and `CAMELID_ENTERPRISE_TEST_MODEL`. The refusal
names every offender at once, says what each does, and prints the allow list.
`x-camelid-admission-sha256` is that policy's identity: two builds whose allow
lists differ publish different values, and everything else about them can be
identical.

## Starting and draining

**Listening is not readiness, and the gap is now visible in the log.** The model
is read whole and hashed *before* the port is bound, so a replica never answers a
request it cannot attribute; the port is then bound and the same file is loaded,
with nothing served until that finishes. Expect:

```
[lane] listening on http://127.0.0.1:8181
[lane] loading model; nothing is served until the load completes
[lane] model loaded as 'Llama 3.2 1B Instruct'; replica ready
```

The name in the last line is the key the engine filed the weights under, and it
is the name a request's `model` field may use. The bind therefore happens between
two reads of the file, and only the second one is inside the quiet window below.

On a multi-gigabyte GGUF from cold cache the middle of that is tens of seconds of
silence. It is not a hang — it is what makes "serving" imply "these are the
weights whose digest it publishes". Two consequences for probes:

- During that window connections sit unanswered in the listen backlog, so an
  HTTP probe fails (correctly) while a bare **TCP-connect probe would report
  ready far too early**. Do not add a `tcpSocket` readiness probe against the
  replica.
- Size the startup budget against the model. The container `HEALTHCHECK` allows a
  120s start period and the Kubernetes `startupProbe` allows 300s; a large model
  on slow storage can want more.

**A stop during startup is immediate.** The drain handler is armed only once the
replica begins serving, so a stop signal during the load terminates the process
at once — correctly, because nothing is in flight.

**A drain is a sequence, not a signal.** In order:

1. **Deregister** at whatever fronts the fleet (Service endpoint removal, load
   balancer, ingress) and let its own connection drain finish.
2. **Poll the replica's queue to zero** on its own loopback — from inside the
   container or pod (`docker exec`, `kubectl exec`), not through the gateway,
   because the poll has to outlive the gateway and a poll *through* it is itself
   in-flight work there:
   ```sh
   until [ "$(curl -fsS http://127.0.0.1:8181/v1/health \
     | sed -n 's/.*"engine_queue_depth":\([0-9]*\).*/\1/p')" = 0 ]; do sleep 2; done
   ```
   As written this waits forever if the probe cannot be reached, which is the
   safe direction — a replica you cannot reach is not one you have confirmed is
   idle — but put a deadline on it in anything unattended.
3. **Stop the gateway before the replica.** Stopping the replica first leaves the
   gateway answering stragglers with a typed `502`, turning a clean drain into
   client-visible errors.
4. **Stop the replica.**

Facts to size the grace period against:

- **Drain time is a sum, not a maximum.** The engine runs one generation at a
  time, a posted job cannot be aborted, and the bounded queue holds 8 waiting
  plus 1 running — so a full replica owes up to 9 generations, each bounded only
  by the engine's 15-minute wall clock. The honest worst case is 135 minutes, and
  no grace period covers it. That is what step 2 is for.
- **`max_tokens` 800 is a default, not a ceiling.** It is applied when a request
  omits `max_tokens`; a client asking for 8000 gets 8000. Capping it above the
  replica is what turns the typical drain into a guaranteed one.
- **A second signal does not escalate.** The shutdown signal resolves once and
  does not re-arm; `kill -9` is the only way to shorten a drain in progress.
- **Receipts are not flushed by the drain.** Each is written from a detached
  task, so one may land after the response it describes. A stop does not wait for
  it.

## Docker

```console
$ docker network create camelid-enterprise
$ docker build -f deploy/docker/Dockerfile -t camelid-enterprise:0.1.0 .
$ docker build -f deploy/docker/Dockerfile.gateway -t camelid-enterprise-gateway:0.1.0 .
$ docker run --name camelid-replica --network camelid-enterprise \
  -v /path/to/models:/models:ro \
  camelid-enterprise:0.1.0 --model /models/model.gguf
$ docker run --name camelid-gateway --network camelid-enterprise \
  -p 127.0.0.1:8080:8080 \
  camelid-enterprise-gateway:0.1.0 \
  --upstream http://camelid-replica:8181
```

Send client traffic to `http://127.0.0.1:8080`. The replica is reachable only
inside the Docker network; the gateway is the single entry point. Neither image
bakes a model. Container builds are Linux — bare-metal Apple Silicon hosts run
the binaries directly.

Both images use an exec-form `ENTRYPOINT`, so SIGTERM reaches the process as PID 1
rather than a shell. Stopping is where the defaults are wrong for this workload:
`docker stop` allows **10 seconds** before SIGKILL, which is shorter than one
generation. Drain first, then stop the gateway container, then the replica, and
give both a real grace period:

```console
$ docker stop -t 900 camelid-gateway
$ docker stop -t 900 camelid-replica
```

The gateway's grace period matters as much as the replica's: it forwards response
bodies as opaque streams and waits for in-flight requests at shutdown, so a
generation is in flight *there* for its whole duration, and a short timeout
truncates a client the replica is still generating for.

## Kubernetes

```console
$ kubectl apply -f deploy/k8s/deployment.yaml -f deploy/k8s/service.yaml \
  -f deploy/k8s/gateway-deployment.yaml -f deploy/k8s/gateway-service.yaml
```

Adjust before applying:

- **PVC** — the manifests expect a `camelid-models` claim holding the GGUF.
- **Resources** — requests equal limits (Guaranteed QoS) on purpose; size to
  the model. Set the `nodeSelector` so one pool = one instance type.
- **`--threads`** — set explicitly in the shipped manifest (`8`), matching the
  cpu request/limit. Keep the two in step, and keep the value identical across
  the pool and with any other artifact that starts a replica of it (the launchd
  unit passes `--threads` too). Do not drop the flag: unset, the width comes
  from the cgroup quota and the affinity mask, so editing the cpu limit or
  landing on a differently-masked node changes the served width with no diff in
  any manifest and no change to `config_sha256` or `admission_sha256` — a pool
  that is really several pools wearing one Service.
- **`enableServiceLinks: false`** — already set on the replica pod spec, and it
  must stay. Kubernetes derives legacy Docker-link variables from Service names
  and injects them into every pod scheduled *after* the Service exists; this
  repository ships two Services whose names begin with the prefix the replica's
  admission scan claims, so a pod receives **sixteen** `CAMELID_`-prefixed
  variables it never asked for and refuses to start. The timing is what makes it
  expensive: the first rollout's pods precede the Services and come up clean, and
  every rollout after it fails. The gateway Deployment deliberately does not set
  this — the gateway runs no admission scan.
- **Probes** — listening is not readiness: the model loads after the port is
  bound, so the startup and readiness probes ask `GET /v1/health` for
  `"generation_ready":true`, and so does the container `HEALTHCHECK`. A model
  whose runtime could not be built is still listed by `/v1/models`, so the
  weaker "is anything listed?" probe admits a pod that fails every generation
  request. Liveness stays on `/v1/models`: it asks whether the process still
  answers, which is the question a restart is the right response to. Do not add
  a `tcpSocket` readiness probe against the replica — see
  [Starting and draining](#starting-and-draining).
- **`terminationGracePeriodSeconds`** — defaults to 30, which is shorter than one
  generation. Raise it on the replica pod, and drain before deleting a pod rather
  than relying on it.
- **Receipts** — the example writes JSONL serving receipts to an `emptyDir`;
  point it at durable storage if receipts are part of your audit trail. Note that
  a receipt is written for *every* request, probes included: a health-probe line
  is about 420 bytes, so the readiness probe alone (every 10s) writes roughly
  3.6 MB a day, and the startup probe's 5s interval doubles the rate while it
  runs.
- **Gateway exposure** — `camelid-enterprise-gateway` is a `ClusterIP` Service.
  Keep it private in this unauthenticated release. Add an Ingress or external
  load balancer only after an access-control layer is in place.
- **Gateway probes** — readiness traverses the gateway to `/v1/models`, so it
  reflects replica availability. Liveness is TCP-only, so an unavailable model
  pool does not cause a gateway restart loop.

## Bare metal

Apple Silicon hosts run the binaries directly rather than in a container.
[`deploy/macos/`](macos/README.md) has the launchd units, the service-account
layout, log rotation, and the readiness and drain procedures for a racked Mac —
including when a box should and should not run its own gateway.
