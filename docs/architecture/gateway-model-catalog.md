# Gateway static model catalog

**Status:** implemented as the first model-routing slice. This document states
the gateway behavior that exists now. It is intentionally narrower than a
model-management service.

## Scope

The gateway has two mutually exclusive startup modes:

- **Transparent mode:** `serve --upstream http://replica:8181`. This is the
  existing one-origin proxy. It forwards every contractual route unchanged.
- **Catalog mode:** one or more repeated
  `--model-route <model-id>=<http://replica-pool>` values. The validated static
  catalog maps an exact backend model id to one operator-configured HTTP origin.

Catalog mode is immutable for the lifetime of the process. Before binding its
listener, the gateway calls every configured pool's `/v1/models` endpoint and
refuses startup unless each catalog id is advertised verbatim by its mapped
pool. IDs are therefore not friendly aliases: the gateway forwards a generation
body unchanged, so an alias would otherwise route correctly and then fail at
the replica. It has no database, hot reload, upload, model lifecycle API,
health aggregation, or dynamic pool membership. The operator owns the
configuration; clients never supply an origin, hostname, or path used for
routing.

## Routing contract

`GET /v1/models` returns the configured catalog in stable lexical id order.
`GET /v1/models/{model}` returns one configured item or a typed `404` with
`model_not_found`. These are catalog inventory endpoints, not readiness checks.
They do not contact replica pools.

Only these requests are routable in catalog mode:

- `POST /v1/completions`
- `POST /v1/chat/completions`

They must carry `Content-Type: application/json` or `application/*+json` and a
non-empty top-level JSON `model` string in a JSON **object**. The selector has
exactly one job: look up a static catalog entry. The original URI, headers, and
exact request bytes are forwarded to that entry after selection. Query
parameters, including a `model` query parameter, do not affect origin selection.
`text/*+json` and all other media types are refused locally. A body that is not
an object is refused too: serde fills a derived struct from a JSON array
positionally, so `["alpha"]` would otherwise be read as `{"model":"alpha"}` and
routed, while the replica reading the same bytes maps position zero onto a
different field of its own request type.

## Selector capacity

The gateway materializes at most `--max-model-selection-body-bytes` bytes
(2 MiB by default) while selecting a model. It separately reserves
`--model-selection-memory-budget-bytes` (32 MiB by default) for that work, so
at default settings at most $32\text{ MiB} / (2 \times 2\text{ MiB}) = 8$
selector bodies can be materialized concurrently. Each slot reserves the raw
body plus the decoded copy of the JSON model id.

That capacity is a queue, not a gate. A request that finds it busy **waits up
to five seconds** for a slot and receives a typed `503` only if the wait
expires. Refusing on contact instead would fail requests that are valid and
would have been served a moment later: a selector slot is held for the
milliseconds a body takes to arrive, so any momentary overlap between two
ordinary requests collides. Measured on loopback with fail-fast admission,
sixteen concurrent 64 KiB requests from one organization lost 75% of the burst,
and sixty-four requests spread across sixty-four distinct organizations lost
84% to the global bound alone. Waiters hold no body memory, so the declared
memory budget is unchanged by queueing.

Waiting is safe because holding is bounded. A request that has a slot must
deliver its body within **fifteen seconds** or it is refused with `408` and the
slot is reclaimed. Without that deadline the only limit on how long one slot
stays occupied is `--max-connection-seconds` (300 by default), and a handful of
dribbling clients could hold the budget for minutes while every waiter behind
them timed out.

A request whose declared `Content-Length` already exceeds the body limit is
refused with `413` from its head alone, before it takes a slot or the bandwidth
behind one. Malformed JSON, a missing model, an unknown model, a non-object
body, and an unsupported media type are all rejected before quota, inference
admission, and forwarding. This is a deliberate availability bound, not an
attempt to validate the full inference schema: the pinned replica remains
authoritative for that.

With `--identity-db`, catalog mode also bounds how much of that capacity one
organization may hold at once. The default is **half the global capacity**
(four slots at the default budget); `--max-org-model-selections` overrides it.
The bound is derived rather than fixed so the invariant survives a
reconfigured budget: no single organization can take more than half the
selector memory, and there is always capacity left for another tenant. This
permit is acquired after authentication and before global selector memory, and
the two acquisitions share one five-second budget rather than one each. It
prevents one tenant's incomplete JSON body from occupying every global selector
slot, while malformed or unknown selectors remain quota-free. The
per-organization limit is per gateway process, like the fixed-window quota; it
is not distributed policy or billing state.

`/v1/health` and the currently unsupported compatibility POST routes have no
model selection contract. Catalog mode returns typed `501` for them rather than
arbitrarily choosing a pool or claiming aggregate readiness. `/healthz` remains
the gateway's local liveness endpoint. Replica readiness stays with each pool's
own readiness probe.

## Operational notes

Catalog startup contacts every configured pool, so the gateway **cannot start
while any configured pool is unreachable** — it exits with the origin that
failed rather than binding a listener that would answer `model_not_found` for
part of its catalog. On Kubernetes this means a gateway rollout that coincides
with a replica pool being down will `CrashLoopBackOff` until the pool answers.
That is the intended trade: a catalog the gateway cannot vouch for is worse
than a gateway that is visibly not ready. The stock manifest
(`deploy/k8s/gateway-deployment.yaml`) remains single-upstream mode and is
unaffected.

Local catalog discovery is charged against `--org-request-quota` like any other
authenticated request. Quota bounds gateway work, not inference, and answering
`GET /v1/models` is gateway work.

## Security and evidence

With `--identity-db`, authentication happens before catalog selection. An
anonymous caller cannot enumerate catalog entries or distinguish an unknown
model from a valid one. A resolved token may use every entry: catalog mode does
not add organization-to-model permissions, so it is not an IDOR authorization
mechanism. That policy needs roles and an explicit entitlement model.

After a request selects a catalog entry, `model_id` is written to the gateway
audit and terminal usage JSONL records. It is `null` for transparent mode,
catalog-list responses, and requests refused before selection. The existing
opaque `request_id` still joins gateway evidence to replica receipts.

The tests use two real TCP upstream servers and cover model-specific routing,
query-parameter non-override, stable local discovery, exact backend-id
verification, malformed/missing/unknown/oversized/non-object selector rejection,
strict media types, no-upstream-call failure paths, stalled-body selector
saturation, reclaiming a slot from a body that misses the read deadline,
refusing an over-declared body before it spends capacity, queueing a
thirty-two-request burst across four organizations without refusing any of it,
authentication-before-discovery, cross-organization selector fairness, quota
preservation for invalid requests, streaming response preservation, and matching
audit/usage model identities. A
separate ignored real-GGUF test starts the attributed replica over TCP, discovers
its actual model id, preflights the catalog, generates through the gateway, and
joins its audit evidence to the replica receipt; CI runs it when gateway code
changes.

## Follow-on boundaries

This slice deliberately does not build model uploads, registration workflows,
automatic failover, per-model health aggregation, dynamic reload, per-model
authorization, durable catalog storage, token metering, or distributed quota
state. Those require the model service and PostgreSQL-backed platform data
work described in the broader architecture plan.