# Deployment

- [Scaling model](#scaling-model)
- [Trust boundary](#trust-boundary)
- [What a replica publishes, and what it serves](#what-a-replica-publishes-and-what-it-serves)
- [Gateway contract](#gateway-contract)
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

The gateway has two explicit startup modes. Legacy transparent mode uses one
fixed `--upstream`, keeps request and response bodies opaque and streaming, and
forwards the replica contract unchanged. Static catalog mode uses one or more
`--model-route <model-id>=<http://replica-pool>` entries instead; it maps a
model to a pool without letting a client select an arbitrary origin. The modes
are mutually exclusive. The stock Kubernetes manifest remains single-upstream
mode. Authentication and per-organization quotas are optional in either mode;
the manifest configures neither an identity database nor a quota, so keep both
services on a trusted network until an operator supplies identity and enables
authentication.
Catalog ids are exact backend ids, not aliases: before binding, catalog mode
queries each configured pool's `/v1/models` and refuses startup unless that
pool advertises the configured id. This matters because selection forwards the
client body unchanged; an alias would otherwise produce a late
`model_not_found` from the correctly selected pool.
With both an identity database and `CAMELID_GATEWAY_USAGE_LOG=<path>`, the
gateway also writes an append-only JSONL terminal transport record for each
authenticated, quota-admitted request:
`{ts, started_ts, duration_ms, request_id, principal, organization, method, path, model_id, response_head_status, request_bytes, response_bytes, stream_outcome}`.
`model_id` is the selected static-catalog entry, or `null` for legacy
transparent mode, catalog discovery, and requests rejected before selection.
`stream_outcome` is `completed` when the response reached EOF, `body_error`
when it returned an error, `gateway_timeout` when the gateway's connection
deadline closed it, or `incomplete` when the stream was dropped without an
observable cause. `request_bytes` is `null` unless the gateway forwarded the
request and observed its body reach EOF; `0` is therefore a known empty body,
not a default for an unmeasured one. Byte counts are opaque payload bytes
observed by the gateway, not tokenizer usage or billable inference units.
The log is best-effort while running: a full writer queue or a failing disk
drops records, with a rate-limited warning for each. On a clean shutdown
(SIGTERM, including a Kubernetes rolling update) the gateway drains whatever
the queue already accepted before exiting — **provided the termination grace
period leaves room for it**; see the shutdown-budget note below, because a
grace period equal to the connection cap does not. If the drain does not finish
within five seconds it stops waiting and logs an `error` naming how many
records were still queued; the writer keeps going, so that count is an upper
bound on what is lost, not a measurement. An abrupt crash still loses the
queue, and each pod writes its own file. It is useful evidence for later
durable aggregation, not a replacement for it.
Only the OpenAI-compatible `/v1` inference surface is public through the gateway;
replica `/api`, embedded WebUI, workspace, and model-lifecycle routes return 404.
The gateway admits at most 256 concurrent request streams by default (including
the full lifetime of streaming responses); set `CAMELID_GATEWAY_MAX_IN_FLIGHT`
to tune that bound. Every accepted client connection is also force-closed after
`CAMELID_GATEWAY_MAX_CONNECTION_SECONDS` (default 300s) regardless of activity.
This is a coarse wall-clock cap, not an idle timeout: it is what stops a client
that drips a request body one byte at a time, or opens a response stream and
never reads it, from pinning an admission permit indefinitely. Size it to
comfortably exceed the slowest real generation this deployment serves.

When a replica's queue is full it returns `503` + `Retry-After`; treat that as
the autoscaling signal (scale on queue-full rate or p95 latency, not CPU — a
serialized replica at steady decode is *supposed* to sit near its CPU limit).

## Trust boundary

**A replica is an internal service.** It has no authentication, no tenant
identity, and no notion of a caller at all. Clients reach it through the gateway,
and the deployment is responsible for keeping ordinary workloads off the replica
port. That is the boundary; everything else on this page sits behind it.

On Kubernetes that responsibility is
[`k8s/replica-network-policy.yaml`](k8s/replica-network-policy.yaml), which
admits port `8181` only from gateway-labelled pods in the same namespace. It has
three preconditions worth checking before treating it as a control rather than a
formality, all of them stated in the manifest itself: the CNI must actually
enforce `networking.k8s.io/v1` NetworkPolicy, or the API accepts the resource and
filters nothing; pod labels are selectors and not workload identity, so namespace
RBAC has to stop untrusted principals from labelling a pod
`app=camelid-enterprise-gateway`; and NetworkPolicy cannot block the node itself,
which also covers `hostNetwork` workloads. In Docker, the equivalent is putting
the replica on a private network and publishing only the gateway. On one box, it
is leaving the replica on loopback.

**What the replica itself enforces is defence in depth, not the boundary.** The
route contract and the generation-body model check below bound *what* an admitted
caller may ask for; neither bounds *who* may ask, and neither becomes an access
control layer by being thorough. They exist because the failure they prevent —
a replica whose published identity stops describing what it is serving — is
reachable by anyone already inside the boundary, including a gateway bug. Read
them as the second layer, and keep the first one.

**The gateway can require a bearer token, and does not by default.**
`serve --identity-db <path>` rejects any request without a valid
`Authorization: Bearer <token>` with a typed `401` before it reaches a replica;
omitting the flag leaves the original unauthenticated pass-through. Tokens do not
expire and the gateway terminates no TLS, so enabling it over a plaintext hop
lets an on-path observer capture and replay one indefinitely — put a
TLS-terminating ingress or reverse proxy in front of it, or a genuinely private
network. The gateway logs a warning to this effect at startup.

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

A receipt carries one field the replica did not mint: `request_id`, the
correlation id the gateway stamps on every request it forwards, recorded verbatim
and `null` for traffic that arrived without one. It is what joins a receipt to
the gateway's own audit line — `gateway_audit ⨝ replica_receipt ON request_id` —
so *which principal asked* stays in the gateway's log and never enters the
replica's. Two bounds on that join: the id is echoed as received, so a client
that can reach the replica directly can forge one, which is another thing the
[trust boundary](#trust-boundary) is holding up; and the gateway audits the
response *head*, so the pair is a correlation record and not a metering one.

Two replicas may publish the same `config_sha256` and still emit different
tokens, because the digest deliberately does not cover the machine, the width or
the weights. That is the digest's scope, not a defect;
[ADR 0002](../docs/adr/0002-replica-identity-surface.md) states it in full.

**The replica serves the contract's routes and nothing else.** The served surface
is [Replica HTTP Contract v1](../docs/contracts/replica-http-v1.md), and the
replica does not keep a private second copy of it: the route filter defers to the
dependency-free registry in `crates/replica-contract`, which is that contract's
machine-readable form. Admitted, with `HEAD` and CORS preflight where they apply:

| Method | Path | Answered by |
|---|---|---|
| `GET` | `/v1/health`, `/v1/models`, `/v1/models/<id>` | the engine |
| `POST` | `/v1/completions`, `/v1/chat/completions` | the engine |
| `POST` | `/v1/embeddings`, `/v1/responses`, `/v1/messages`, `/v1/rerank`, `/v1/reranking` | the engine's typed `501` "unsupported" replies |

The last row is a contractual surface rather than a capability: the pinned engine
answers those paths with a typed `501` and an `unsupported_*` code, and the
contract carries that through so a client SDK's capability probe gets the
engine's own answer instead of a refusal it has to special-case.

Everything else — the engine's unauthenticated model-load, model-unload and
runtime-control routes, the workspace family, telemetry and execution-plan
streams, the legacy completion-server routes (one of which is a second generation
route attribution does not inject body fields for), and the embedded web UI —
answers `403` with `"code":"route_not_served"`. The filter is an allow list for
the reason admission is: a route a later engine pin invents arrives refused
rather than arriving served and waiting to be noticed.

None of this is an access-control layer. It bounds *what* an admitted caller may
ask for, never *who* may ask — see [Trust boundary](#trust-boundary), which is
the layer that decides who reaches the port at all.

**A generation request cannot repoint the replica either.** The engine resolves a
request's `model` field against the filesystem before anything else, so on the
two generation routes that field is checked as well as the path: it may name this
replica's own weights or be omitted, and anything else answers `404` with
`"code":"model_not_served"` — identically whether or not a file of that name is
on the mount. Withholding the model-management routes and checking the field are
*together* what make the published model digest a claim about the whole process
lifetime rather than about its first second. It is a **separate** control on
purpose, and no route list is a substitute for it: `/v1/chat/completions` is
contractual, the gateway forwards it, and the network policy admits the gateway —
so every layer above correctly lets this request through, and the field inside it
is the only place left to look. This matters most where the manifests put it: the
shipped Kubernetes PVC mounts the whole model pool at `/models`, so without the
body check any replica in the fleet could be pointed at any other model in it,
over a route every other control is required to admit.

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

## Gateway contract

Both modes default-deny every route outside this replica surface:

- `/v1/health`
- `/v1/models` and `/v1/models/{model}`
- `/v1/completions` and `/v1/chat/completions`
- `/v1/embeddings`, `/v1/responses`, `/v1/messages`, `/v1/rerank`, and
  `/v1/reranking` (the compatibility responses are owned by the pinned engine)

Everything else returns `404` at the gateway without contacting the replica. In
particular, `/api/*`, legacy `/models/*`, workspace routes, and the embedded
WebUI are replica-local and never public client paths.

The list above is not maintained here or in the gateway. Both the gateway's
forwarding table and the replica's route filter are built from
`replica_contract::PUBLIC_ROUTES`, so there is one edit rather than two lists
that can disagree. That matters in the direction that is easy to miss: a route
added to the contract and served by the replica but absent from a hand-written
gateway table would have been `404`ed at the trust boundary — an availability
failure visible only in production, on the release that added it.

`GET /healthz` is gateway-local and returns `204`; Kubernetes uses it for both
readiness and liveness so replica saturation or temporary unavailability does
not remove otherwise healthy gateway endpoints and amplify an outage.
All gateway responses include permissive CORS visibility headers, matching the
pinned replica API, so browser clients can inspect typed gateway `502`/`503`
errors and route-level `404`/`405` failures. Cross-origin preflight (`OPTIONS`
with `Access-Control-Request-Method`) is answered locally by the CORS layer
and never reaches the replica or consumes an admission permit. No credentials
are enabled.

In transparent mode, request and response bodies remain streaming and opaque.
Catalog mode keeps response bodies streaming and opaque, but materializes a
JSON generation request up to `--max-model-selection-body-bytes` (2 MiB by
default) to read its `model` field. A separate
`--model-selection-memory-budget-bytes` limit (32 MiB by default) allows at
most `memory_budget / (2 * max_body)` concurrent materialized selectors: 8 at
the defaults. Each slot reserves a raw body plus the decoded model-id copy.

That capacity queues rather than refuses. A request that finds it busy waits up
to five seconds for a slot and gets a typed `503` only if the wait expires: a
slot is held for the milliseconds a body takes to arrive, so refusing on
contact would fail valid requests that merely overlapped. Holding a slot is
bounded in turn -- a request that has one must deliver its body within fifteen
seconds or it is refused with `408` and the slot is reclaimed, so slow clients
cannot occupy the budget for the whole `--max-connection-seconds` window. A
request whose declared `Content-Length` already exceeds the body limit is
refused with `413` from its head, before it takes a slot. Catalog mode accepts
only `application/json` and `application/*+json`, and only a JSON *object*;
malformed, missing, unknown, non-object, oversized, or other-media-type
selectors are rejected before quota, inference admission, or any upstream call.
Only `POST /v1/completions` and `POST /v1/chat/completions` are routable in
catalog mode because the pinned Enterprise contract proves their JSON `model`
field. Catalog discovery is served locally and is charged against
`--org-request-quota` like any other authenticated request. `/v1/health` and
the currently unsupported compatibility POST routes return typed `501` rather
than being sent to an arbitrary pool; `/healthz` remains the gateway liveness
endpoint. Catalog discovery reports configured inventory, not replica
readiness.

When identity is enabled, one organization may hold at most half the global
selector capacity by default -- four slots at the default budget
(`--max-org-model-selections` / `CAMELID_GATEWAY_MAX_ORG_MODEL_SELECTIONS`
changes it). The bound is derived from the budget rather than fixed, so the
invariant holds however the budget is configured: no tenant takes more than
half, and there is always capacity left for another one. This per-process
permit is acquired before the global selector slot, and the two acquisitions
share one five-second wait rather than one each. It is intentionally separate
from request quota: invalid selectors remain uncharged, but cannot monopolize
selector capacity.

Catalog startup contacts every configured pool, so **the gateway will not start
while any configured pool is unreachable**; on Kubernetes a gateway rollout
that coincides with a pool outage restarts until the pool answers. That is
deliberate -- a catalog the gateway cannot vouch for is worse than a gateway
that is visibly not ready -- and it does not affect the stock manifest, which
remains single-upstream mode.

Authentication runs before catalog selection, so an authenticated deployment
does not disclose model inventory to an anonymous caller. Catalog mode does
not implement per-organization model permissions: every authenticated caller
can use every configured entry. Do not present this as an IDOR control; roles
and resource policy remain future authorization work. The gateway removes HTTP
hop-by-hop headers and strips client-supplied `Forwarded`, `X-Forwarded-*`, and
`X-Real-IP` values because this release does not yet establish trusted client
identity. It stamps its own `x-camelid-request-id` on every forwarded request,
overwriting any client-supplied value, and with `serve --audit-log <path>`
writes one JSONL line per handled request — auth and admission rejections
included — as `{ts, request_id, principal, method, path, status}`. It does not
retry requests. The upstream connection pool keeps at
most 32 idle connections for 30 seconds, and the configurable concurrency
limit is held until a streaming response completes, disconnects, or the
connection's maximum duration elapses. Saturation returns a typed `503` with a
`Retry-After` of 1-3 seconds (randomized, so clients rejected at the same
instant do not retry in lockstep).

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
  -f deploy/k8s/gateway-deployment.yaml -f deploy/k8s/gateway-service.yaml \
  -f deploy/k8s/replica-network-policy.yaml
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
- **Gateway exposure** — `camelid-enterprise-gateway` is a `ClusterIP` Service,
  and it is the only thing that should be reachable from outside the namespace.
  Bearer-token enforcement is opt-in and the gateway terminates no TLS, so add an
  Ingress or external load balancer only behind a TLS terminator and with
  `--identity-db` in place. See [Trust boundary](#trust-boundary).
- **Gateway probes** — readiness and liveness use the local `/healthz` endpoint.
  Replica availability is represented by the replica Deployment's own readiness
  and by typed gateway `502` responses, not by removing healthy gateway pods.
- **Per-organization quota** — when you opt in with an identity database plus
  `CAMELID_GATEWAY_ORG_REQUEST_QUOTA`, the counter is fixed-window, in-memory,
  and local to each gateway process. The supplied manifest runs **two** gateway
  replicas, so Kubernetes distributes an organization's traffic across two
  independent counters: each process can admit under `2 × limit` in a short
  fixed-window-boundary burst, making the deployment-wide worst case under
  `4 × limit`. Size the per-pod value with that bound in mind. A strict global
  organization quota needs durable, shared state and is not implemented.
- **Transport usage log** — `CAMELID_GATEWAY_USAGE_LOG` requires
  `CAMELID_GATEWAY_IDENTITY_DB`; use a writable, durable mounted path if the
  JSONL evidence must survive pod replacement. The stock manifest supplies
  neither identity nor a usage-log volume, so it emits no usage records. These
  raw gateway byte counters and terminal outcomes are not tokenizer usage,
  billing, or a durable aggregation service. The gateway opens each configured
  audit and usage file before it binds its listener, rejects missing parents or
  unwritable destinations at startup, and rejects audit and usage paths that
  resolve to the same file. Runtime writes use a bounded dedicated writer queue:
  a full queue or writer failure loses records but emits rate-limited warnings.
- **Shutdown drain** — both gateway and replica stop accepting connections on
  SIGTERM and drain active streams. The gateway then drains its JSONL logs, so
  the grace period has to cover both stages, not just the first:

  ```
  terminationGracePeriodSeconds
    >= CAMELID_GATEWAY_MAX_CONNECTION_SECONDS   # connection drain
     + 5s per configured JSONL log              # audit and/or usage
     + margin
  ```

  The gateway manifest ships 330s against a 300s connection cap and two
  possible logs. Setting the grace period *equal* to the connection cap leaves
  no budget for the log drain: a connection accepted just before SIGTERM can
  consume the whole period, and kubelet then kills the process mid-drain. Raise
  the grace period whenever you raise the connection cap. The gateway logs the
  budget it needs at startup. Keep the replica window at least as long as the
  gateway window, and size both above the longest permitted generation.
- **NetworkPolicy enforcement** — the replica ingress policy permits port 8181
  only from gateway-labeled pods in the same namespace (plus node traffic that
  Kubernetes always allows for probes). The cluster CNI **must** enforce
  `networking.k8s.io/v1` NetworkPolicy; otherwise applying the resource has no
  filtering effect. Pod labels are selectors, not workload identity: namespace
  RBAC must prevent untrusted principals from creating or relabeling pods as
  `app=camelid-enterprise-gateway`. NetworkPolicy also cannot block access from
  the node itself and may treat `hostNetwork` workloads as node traffic. Validate
  CNI enforcement, namespace RBAC, and host access before treating the gateway
  as a security boundary.
- **Immutable application image** — the example uses the release tag for
  readability. Production automation must replace it with the published image
  digest (`image@sha256:...`) so a rollout cannot change bytes under one tag.

## Bare metal

Apple Silicon hosts run the binaries directly rather than in a container.
[`macos/`](macos/README.md) has the launchd units, the service-account layout,
log rotation, and the readiness and drain procedures for a racked Mac —
including when a box should and should not run its own gateway. There is no
NetworkPolicy on a single box, so the replica's own isolation is the bind
address: keep it on loopback and let the gateway be the only listener anything
else can reach.
