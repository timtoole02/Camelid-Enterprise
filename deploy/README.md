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

When a replica's queue is full it returns `503` + `Retry-After`; treat that as
the autoscaling signal (scale on queue-full rate or p95 latency, not CPU — a
serialized replica at steady decode is *supposed* to sit near its CPU limit).

## Docker

```console
$ docker build -f deploy/docker/Dockerfile -t camelid-enterprise:0.1.0 .
$ docker run -p 8181:8181 -v /path/to/models:/models:ro \
    camelid-enterprise:0.1.0 --model /models/model.gguf
```

The image binds `0.0.0.0:8181` and bakes no model; mount one read-only.
Container builds are Linux — bare-metal Apple Silicon hosts run the binary
directly.

## Kubernetes

```console
$ kubectl apply -f deploy/k8s/deployment.yaml -f deploy/k8s/service.yaml
```

Adjust before applying:

- **PVC** — the manifests expect a `camelid-models` claim holding the GGUF.
- **Resources** — requests equal limits (Guaranteed QoS) on purpose; size to
  the model. Set the `nodeSelector` so one pool = one instance type.
- **Probes** — listening is not readiness: the model loads after bind, so the
  startup/readiness probes check for a non-empty `/v1/models` list.
- **Receipts** — the example writes JSONL serving receipts to an `emptyDir`;
  point it at durable storage if receipts are part of your audit trail.
