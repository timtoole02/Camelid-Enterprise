# Camelid Enterprise console

A browser console for a Camelid Enterprise deployment: sign in, see what the
endpoint is serving, and chat with it.

It is a client of the published HTTP contract and nothing more. Every request it
makes goes to a gateway, carries the caller's bearer token, and lands on one of
the routes in `docs/contracts/replica-http-v1.md`.

## What it does

- **Sign in with an access token.** The token is verified against the gateway
  before a session exists, and the principal and organization shown in the UI are
  the gateway's answer — never anything typed alongside the token.
- **Report what the endpoint serves.** Readiness comes from `/v1/health`
  (`loaded_now`, `generation_ready`, `active_model_id`); the model list comes
  from `/v1/models`.
- **Chat.** Streaming `POST /v1/chat/completions`, with the replica's attribution
  headers captured per turn so a reply can still say which lane, configuration
  and weights produced it.

## What it deliberately does not do

There is no model loading, unloading, downloading, or deletion, and no runtime
flag flipping. Those are not missing features — a replica does not serve routes
for them. It publishes a model digest, a config digest and a host summary
describing the weights it hashed before it bound its port, and a control plane
reachable over that same port would let a caller invalidate all three. Changing
what a deployment serves is an operator action taken elsewhere.

The console also keeps no server-side state. Conversations live in the browser,
keyed by principal so two people signing in on one machine never read each
other's threads. The gateway's audit log is an operator record, not a transcript
store.

## Run it

Start a gateway with an identity database, so it requires a token:

```bash
cargo run -p camelid-enterprise-gateway -- serve --addr 127.0.0.1:8080 --upstream http://127.0.0.1:8181 --identity-db ./identity.db
```

Create a user and issue them a token (this prints the token once — it is stored
only as a hash):

```bash
cargo run -p camelid-enterprise-gateway -- create-user --identity-db ./identity.db alice
```

```bash
cargo run -p camelid-enterprise-gateway -- issue-token --identity-db ./identity.db <principal-id>
```

Then run the console and open <http://127.0.0.1:4175>:

```bash
npm ci && npm run dev
```

Sign in with the gateway URL and that token. Give each person their own token:
one principal per user is what makes the per-organization quotas and the audit
log mean anything.

A gateway started without `--identity-db` runs unauthenticated. The console
detects that and signs in without a credential rather than prompting for one
nobody can mint.

## Build

```bash
npm run build
```

Output is static assets in `dist/`, servable by anything. Nothing in the build
assumes a particular mount path or origin — the gateway endpoint is chosen at
sign-in and stored per browser.
