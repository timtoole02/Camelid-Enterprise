# Deployment

## Scaling model

The deterministic lane scales **horizontally**. One replica serves one model,
one generation at a time — that serialization is what makes its output
reproducible, so a replica never gets faster under load, it gets *more
neighbors*. Capacity planning is therefore simple multiplication:

    aggregate throughput = replicas × single-stream throughput

Route tenants to lanes at the gateway above the Service; the per-response
`x-camelid-lane` and `x-camelid-config-sha256` headers let the gateway and
clients verify what they got. Keep each replica pool on **one instance type**:
the lane's behavior is scoped to a hardware class, so a pool that mixes node
types is really several pools wearing one Service.

The gateway in this release is deliberately transparent: one fixed upstream,
opaque streaming request/response bodies, no retries, and no response rewriting.
On Kubernetes, point it at the replica Service and let Kubernetes balance the
identical pool. On one box, point it directly at the replica. Authentication and
tenant-aware routing have not landed; keep both services on a trusted network.
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

## Gateway contract

The gateway forwards only these replica routes:

- `/v1/health`
- `/v1/models` and `/v1/models/{model}`
- `/v1/completions` and `/v1/chat/completions`
- `/v1/embeddings`, `/v1/responses`, `/v1/messages`, `/v1/rerank`, and
  `/v1/reranking` (compatibility responses are owned by the pinned engine)

Everything else returns `404` at the gateway without contacting the replica.
In particular, `/api/*`, legacy `/models/*`, workspace routes, and the embedded
WebUI are replica-local and never public client paths. This list is a
hand-maintained mirror of the pinned engine's public surface; nothing today
verifies it stays in sync with the replica's actual contract if the engine
pin moves (see `crates/replica-contract`, landing separately, for the
authoritative tested inventory).
`GET /healthz` is gateway-local and returns `204`; Kubernetes uses it for both
readiness and liveness so replica saturation or temporary unavailability does
not remove otherwise healthy gateway endpoints and amplify an outage.
All gateway responses include permissive CORS visibility headers, matching the
pinned replica API, so browser clients can inspect typed gateway `502`/`503`
errors and route-level `404`/`405` failures. Cross-origin preflight (`OPTIONS`
with `Access-Control-Request-Method`) is answered locally by the CORS layer
and never reaches the replica or consumes an admission permit. No credentials
are enabled.

Request and response bodies remain streaming and opaque. The gateway removes
HTTP hop-by-hop headers and strips client-supplied `Forwarded`, `X-Forwarded-*`,
and `X-Real-IP` values because this release does not yet establish trusted
client identity. It does not retry requests. The upstream connection pool keeps
at most 32 idle connections for 30 seconds, and the configurable concurrency
limit is held until a streaming response completes, disconnects, or the
connection's maximum duration elapses. Saturation returns a typed `503` with
a `Retry-After` of 1-3 seconds (randomized, so clients rejected at the same
instant do not retry in lockstep).

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
- **Probes** — listening is not readiness: the model loads after bind, so the
  startup/readiness probes check for a non-empty `/v1/models` list.
- **Receipts** — the example writes JSONL serving receipts to an `emptyDir`;
  point it at durable storage if receipts are part of your audit trail.
- **Gateway exposure** — `camelid-enterprise-gateway` is a `ClusterIP` Service.
  Keep it private in this unauthenticated release. Add an Ingress or external
  load balancer only after an access-control layer is in place.
- **Gateway probes** — readiness and liveness use the local `/healthz` endpoint.
  Replica availability is represented by the replica Deployment's own readiness
  and by typed gateway `502` responses, not by removing healthy gateway pods.
- **Shutdown drain** — both gateway and replica stop accepting connections on
  SIGTERM and drain active streams. The examples grant 300 seconds before
  kubelet may send SIGKILL; keep the replica window at least as long as the
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
