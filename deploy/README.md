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

When a replica's queue is full it returns `503` + `Retry-After`; treat that as
the autoscaling signal (scale on queue-full rate or p95 latency, not CPU — a
serialized replica at steady decode is *supposed* to sit near its CPU limit).

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
  -f deploy/k8s/gateway-deployment.yaml -f deploy/k8s/gateway-service.yaml
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
- **Gateway probes** — readiness traverses the gateway to `/v1/models`, so it
  reflects replica availability. Liveness is TCP-only, so an unavailable model
  pool does not cause a gateway restart loop.
