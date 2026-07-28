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
non-empty top-level JSON `model` string. The selector has exactly one job: look
up a static catalog entry. The original URI, headers, and exact request bytes
are forwarded to that entry after selection. Query parameters, including a
`model` query parameter, do not affect origin selection. `text/*+json` and all
other media types are refused locally.

The gateway materializes at most `--max-model-selection-body-bytes` bytes
(2 MiB by default) while selecting a model. It separately reserves
`--model-selection-memory-budget-bytes` (32 MiB by default) for that work, so
at default settings at most $32\text{ MiB} / (2 \times 2\text{ MiB}) = 8$
selector bodies can be materialized concurrently. Each slot reserves the raw
body plus a possible decoded copy of an escaped JSON model id. A full selector
budget receives a typed `503` without reading another body. An over-limit
request receives `413`; malformed JSON, missing model, unknown model, and
unsupported media type are rejected before quota, inference admission, and
forwarding. This is a deliberate availability bound, not an attempt to validate
the full inference schema: the pinned replica remains authoritative for that.

With `--identity-db`, catalog mode also limits each organization to one
concurrent selector body by default (`--max-org-model-selections` changes that
bound). This permit is acquired after authentication and before global selector
memory. It prevents one tenant's incomplete JSON body from occupying every
global selector slot, while malformed or unknown selectors remain quota-free.
The per-organization limit is per gateway process, like the fixed-window quota;
it is not distributed policy or billing state.

`/v1/health` and the currently unsupported compatibility POST routes have no
model selection contract. Catalog mode returns typed `501` for them rather than
arbitrarily choosing a pool or claiming aggregate readiness. `/healthz` remains
the gateway's local liveness endpoint. Replica readiness stays with each pool's
own readiness probe.

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
verification, malformed/missing/unknown/oversized selector rejection, strict
media types, no-upstream-call failure paths, stalled-body selector saturation,
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