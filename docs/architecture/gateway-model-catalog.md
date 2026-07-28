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
  catalog maps a public model id to one operator-configured HTTP origin.

Catalog mode is immutable for the lifetime of the process. It has no database,
hot reload, upload, model lifecycle API, health aggregation, or dynamic pool
membership. The operator owns the configuration; clients never supply an
origin, hostname, or path used for routing.

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
`model` query parameter, do not affect origin selection.

The gateway materializes at most `--max-model-selection-body-bytes` bytes
(2 MiB by default) while selecting a model. An over-limit request receives
`413`; malformed JSON, missing model, unknown model, and unsupported media type
are rejected before quota, admission, and forwarding. This is a deliberate
availability bound, not an attempt to validate the full inference schema: the
pinned replica remains authoritative for that.

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
query-parameter non-override, stable local discovery, malformed/missing/
unknown/oversized selector rejection, no-upstream-call failure paths,
authentication-before-discovery, quota preservation for invalid requests,
streaming response preservation, and matching audit/usage model identities.

## Follow-on boundaries

This slice deliberately does not build model uploads, registration workflows,
automatic failover, per-model health aggregation, dynamic reload, per-model
authorization, durable catalog storage, token metering, or distributed quota
state. Those require the model service and PostgreSQL-backed platform data
work described in the broader architecture plan.