# macOS (Apple Silicon), bare metal

A deterministic-lane replica on a racked Mac runs as a **launchd system
daemon**. Not a LaunchAgent: an agent needs a user session, and a headless
machine has none after an unattended reboot — which is the case this unit exists
for. The usual reason to prefer an agent, that daemons get no window-server
session and so have restricted GPU access, does not apply here: the
deterministic lane is CPU-only by construction. A future accelerated lane would
have to revisit that choice.

| File | Purpose |
|---|---|
| `com.camelid.enterprise.plist` | The replica unit. Read the comments before editing. |
| `com.camelid.enterprise.gateway.plist` | The gateway unit. **Optional** — see [Does this box run the gateway?](#does-this-box-run-the-gateway) |
| `com.camelid.enterprise.conf` | newsyslog rotation for receipts and launchd stdio. |
| `install.sh` | Service account, layout, units, rotation. Idempotent. |
| `uninstall.sh` | Removes the units and the binaries. Removes no data. |

## Install

```bash
cargo build --release --bin camelid-enterprise
sudo deploy/macos/install.sh
```

To install the optional gateway unit alongside the replica:

```bash
cargo build --release --bin camelid-enterprise --bin camelid-enterprise-gateway
sudo deploy/macos/install.sh --with-gateway
```

The installer creates a hidden `_camelid` role account in the reserved system id
range, lays out the directories below, installs the units and the rotation
stanza, and starts the replica **only if a model is already in place**. A
`KeepAlive` job that fails closed is a restart loop, not a diagnostic, so a
missing model stops the installer with instructions instead.

Put the GGUF at `/usr/local/var/camelid-enterprise/models/model.gguf` (owned
`root`, mode `0644`), then:

```bash
sudo launchctl bootstrap system /Library/LaunchDaemons/com.camelid.enterprise.plist
```

Re-running `install.sh` upgrades in place. It stops the running units first —
gateway before replica, the same order as a drain — because replacing a binary
underneath a live process leaves it serving from an unlinked inode while the
unit claims otherwise.

### Layout

```
/usr/local/bin/camelid-enterprise                             root:wheel      0755
/usr/local/bin/camelid-enterprise-gateway                     root:wheel      0755   optional
/Library/LaunchDaemons/com.camelid.enterprise.plist           root:wheel      0644
/Library/LaunchDaemons/com.camelid.enterprise.gateway.plist   root:wheel      0644   optional
/etc/newsyslog.d/com.camelid.enterprise.conf                  root:wheel      0644
/usr/local/var/camelid-enterprise/                            root:_camelid   0755   working directory
/usr/local/var/camelid-enterprise/models/model.gguf           root:_camelid   0644   read-only to the service account
/usr/local/var/camelid-enterprise/receipts/serving.jsonl      _camelid        0640   written by the replica
/usr/local/var/log/camelid-enterprise/lane.{out,err}.log      _camelid        0640   launchd stdio
/usr/local/var/log/camelid-enterprise/gateway.{out,err}.log   _camelid        0640   launchd stdio, optional
```

Nothing lives in a repository checkout, and nothing lives under a user's home
directory, `/Volumes`, Desktop, Documents or Downloads: a daemon that touches
those raises a consent prompt with no session to answer it. `/usr/bin`,
`/usr/sbin` and `/usr/libexec` are read-only system locations, so `/usr/local`
it is. Every path is space-free because newsyslog's config format is
whitespace-delimited.

The model is owned by `root` and only readable by the service account. A replica
that can rewrite its own model can change what it serves without changing
anything it publishes about itself.

## Does this box run the gateway?

**Default: no.** Install the replica alone and let whatever fronts the fleet be
the entry point. Reasons, all checkable against `crates/gateway`:

- The gateway forwards to **one fixed origin**, with no retries and no response
  rewriting. In Kubernetes it earns its place by fronting a Service; per box, in
  front of one replica, it balances nothing and retries nothing.
- It rewrites `Host` to the upstream authority and adds no forwarding headers,
  so the replica sees the gateway as its peer. Every per-client fact is lost at
  the first hop. Losing it once at the fleet edge is better than once per box.
- Its shutdown waits for in-flight requests, and it forwards response bodies as
  opaque streams — so a streaming generation is in-flight work **at the
  gateway** for its whole duration. A per-box gateway therefore inherits the
  replica's entire drain budget, and becomes a second process that can truncate
  a stream the replica is still producing. This is the non-obvious cost, and it
  is why the gateway unit carries the same `ExitTimeOut` as the replica.

**Install it (`--with-gateway`) when this Mac serves clients directly with no
load balancer in front of it.** Then the box has one entry point on `0.0.0.0:8080`
and the replica stays on `127.0.0.1:8181`.

One property makes an edge gateway acceptable rather than merely tolerable: the
gateway does **not** strip `x-camelid-*`. It removes only hop-by-hop headers and
the names a `Connection` header nominates, and its contract tests assert that
`x-camelid-lane`, `x-camelid-config-sha256` and `x-camelid-host` come back
unchanged through the real proxy. So a load balancer probing *through* a gateway
still sees the replica's own identity, and the identity-pinning probe below works
from either side. (The contract suite has not yet been extended to
`x-camelid-admission-sha256`, `x-camelid-model-sha256` and
`x-camelid-worker-threads`; the forwarding rule is name-agnostic, so they pass
through for the same reason, but they are not pinned by a test yet.)

launchd expresses no ordering between jobs. None is needed at boot — while the
replica is starting the gateway answers `502`, which is the correct answer. It
matters at **stop**, in the other direction; see the drain sequence.

## Configuring the unit

Everything is an argv flag in `ProgramArguments`; **the unit sets no `CAMELID_*`
environment variable of any kind.** The replica refuses to start on any
`CAMELID_*` name it does not recognize, and the strongest posture is one where
its startup scan finds nothing at all to classify.

`VECLIB_MAXIMUM_THREADS` is the refused name most likely to be already exported
on a Mac, and the replica refuses to start on it. It resizes Accelerate's
internal BLAS pool; the engine calls into Accelerate on this platform, including
from a tensor matmul path with no size floor and no flag in front of it; and
vecLib's documented response is to change how it blocks and parallelizes those
calls, which moves f32 summation order. Nothing the replica publishes reports
vecLib's pool width, so a replica running under it would look identical to one
that is not. `RAYON_NUM_THREADS` and its live deprecated alias
`RAYON_RS_NUM_CPUS` are refused too — both size the pool `--threads` owns, and
naming one spelling and not the other would leave the refusal bypassable by a
synonym. See [When it will not start](#when-it-will-not-start).

Review before starting:

- **`--threads`** — sizes this process's global data-parallel worker pool, and
  the width the pool actually comes up at is read back and published on every
  response as `x-camelid-worker-threads`. It must be **identical on every replica
  in a pool**: the reproducibility claim is scoped to a hardware class and a
  width, and `config_sha256` deliberately does not cover one. The shipped `8` is
  a placeholder. `--threads 0` is refused rather than reinterpreted.
- **`--model`** — must exist at start. The replica canonicalizes the path, reads
  the file whole to compute the digest it publishes, and fails closed before
  binding a port if it cannot.
- **`--addr`** — `127.0.0.1:8181`. See [Exposure](#exposure) before widening it.
- **`--serving-receipts`** — remove the flag if you do not want an audit trail;
  point it at durable storage if you do and this disk is not. Note that a
  receipt is written for **every** request, health probes included — see
  [Logs and rotation](#logs-and-rotation).
- **`--lane`** — `deterministic` is the only value this release accepts.

## What a healthy start looks like

```console
[lane] deterministic | engine pin b4e3a9056567ed8145fc4fa29850d6f1f261ac2b | config vector sha256 30d77c260803 | admission sha256 45121fb83fef | model sha256 3f8a1c04b7e2 | host macos/aarch64 cores=8 simd=dotprod+i8mm+neon | worker threads 8
[lane] model /usr/local/var/camelid-enterprise/models/model.gguf
[lane] listening on http://127.0.0.1:8181
[lane] loading model; nothing is served until the load completes
[lane] model loaded as 'Llama 3.2 1B Instruct'; replica ready
```

`model sha256` is the first twelve hex characters of `shasum -a 256` over your
own GGUF; the value above is an example. The name on the last line is the key
the engine filed the weights under — read from the GGUF's own metadata — and it
is the name a client's `model` field may use.

**The gap between the third line and the fifth is not a hang.** The model is read
whole and hashed before the port is bound; the port is then bound and the same
file is loaded, with nothing served until the load returns, so on a
multi-gigabyte GGUF from cold cache the process is quiet for tens of seconds.
That is deliberate: it is what makes "this replica is serving" imply "these are
the weights whose digest it publishes". During that window connections sit
unanswered in the listen backlog, so an HTTP probe fails — correctly — while a
bare TCP-connect check would report the replica ready far too early. Use the
HTTP probe below.

## Readiness

launchd has no readiness concept at all: `KeepAlive` observes process exit, never
"up and not serving". Readiness belongs to whatever fronts the box.

```sh
/usr/bin/curl -fsS --max-time 3 http://127.0.0.1:8181/v1/health \
  | /usr/bin/grep -q '"generation_ready":true'
```

Exit 0 means admit traffic. Use `/v1/health`, not `/v1/models`: a model can be
listed while it cannot generate, because `generation_ready` additionally requires
a live runtime. macOS ships no `jq`, and `grep` is order-independent, so an
upstream field reordering cannot break it. Both `generation_ready` and
`engine_queue_depth` come from the same response, so one route serves readiness
and the drain.

> The Kubernetes startup and readiness probes and the container `HEALTHCHECK`
> now run the same check. They used to grep `/v1/models` for an id, which
> reports healthy on a replica whose model is listed but whose runtime could not
> be built — every generation request on such a pod fails. The Kubernetes
> liveness probe stays on `/v1/models`, because it asks a different question:
> whether the process still answers at all.

### Pinning the identity from the probe

Attribution is stamped on **every** response, `/v1/health` included, so a gateway
or load balancer that should refuse a nonconforming replica can check it in the
same request:

```sh
H="$(mktemp)"; B="$(mktemp)"
curl -fsS --max-time 3 -D "$H" -o "$B" http://127.0.0.1:8181/v1/health \
  && grep -qi '^x-camelid-lane: deterministic'            "$H" \
  && grep -qi '^x-camelid-config-sha256: 30d77c260803'    "$H" \
  && grep -qi '^x-camelid-admission-sha256: 45121fb83fef' "$H" \
  && grep -qi '^x-camelid-model-sha256: <first 12 hex>'   "$H" \
  && grep -qi '^x-camelid-worker-threads: 8'              "$H" \
  && grep -q  '"generation_ready":true'                   "$B"
```

Each line catches something the others cannot, which is why all of them are
worth pinning:

| Header | Catches |
|---|---|
| `x-camelid-config-sha256` | a configuration-vector or engine-pin drift |
| `x-camelid-admission-sha256` | a build whose admission policy is not the one you audited |
| `x-camelid-model-sha256` | a replica serving different weights under the same file name |
| `x-camelid-host` | a replica that came up on the wrong hardware class |
| `x-camelid-worker-threads` | a replica at the wrong pool width |

Get the model value with `shasum -a 256 <your GGUF> | cut -c1-12`. The
configuration digest is `30d77c260803` at engine pin
`b4e3a9056567ed8145fc4fa29850d6f1f261ac2b`; it changes at a pin bump. The
admission digest is `45121fb83fef`, and it does *not* change at a pin bump —
only when the allow list or the foreign-refusal list does.
[ADR 0002](../../docs/adr/0002-replica-identity-surface.md) says what each does
and does not claim.

Both values are maintained by hand — here, in the repository `README.md` and in
the ADR — including the two literals inside the probe above, which are
executable: a stale copy there fails against a correctly-built replica. The
workspace test suite asserts that all three documents carry the digests the
binary computes, so a policy edit that updates the code and forgets this page
fails the build rather than shipping. If a check here ever disagrees with a
replica you trust, compare against `crates/server/src/lane.rs` before concluding
the replica is at fault.

If you want an in-box watchdog, a second LaunchDaemon with `StartInterval`
running the readiness probe and calling
`launchctl kickstart -k system/com.camelid.enterprise` after N consecutive
failures is the shape. Keep N high enough that one long generation on a busy
replica cannot trip it.

## Draining a replica

Both processes handle SIGTERM by refusing new connections and waiting for
in-flight work, and `launchctl bootout` sends SIGTERM — so a stop *is* drained.
But the drain is the sequence below, and `ExitTimeOut` is only its backstop.

**1. Deregister the box** at whatever load balancer fronts the fleet, and let its
own connection drain finish. Nothing after this is safe until new requests have
stopped arriving.

This is the step a per-box gateway cannot do for you. It forwards to one fixed
origin and holds no pool or health state, so there is nothing to deregister *at*;
the deregistration happens above it.

**2. Poll the replica on loopback until its queue is empty.** This is the actual
drain.

```sh
until [ "$(curl -fsS http://127.0.0.1:8181/v1/health \
  | sed -n 's/.*"engine_queue_depth":\([0-9]*\).*/\1/p')" = 0 ]; do sleep 2; done
```

Loopback, not through the gateway: the poll has to outlive the gateway, and a
poll *through* the gateway is itself in-flight work there.

Note the failure mode of that loop as written: if `curl` fails the substitution
is empty, which is not `0`, so it waits forever rather than falling through. That
is the safe direction for a drain — a replica you cannot reach is not a replica
you have confirmed is idle — but put a deadline on it in any script that runs
unattended, and treat hitting the deadline as "investigate", not "proceed".

**3. If the gateway unit runs on this box, stop it FIRST.**

```sh
sudo launchctl bootout system/com.camelid.enterprise.gateway
```

Order matters. Stopping the replica first leaves the gateway answering every
straggler with `502 {"error":{"type":"gateway_error"}}`, turning a clean drain
into visible client errors. Stopping the gateway first makes the box refuse
connections outright, which is what "down" should look like.

**4. Stop the replica.**

```sh
sudo launchctl bootout system/com.camelid.enterprise
```

It stops accepting, finishes what is in flight (already nothing after step 2) and
exits 0. `KeepAlive` is `SuccessfulExit: false`, so launchd leaves it down — a
clean stop is an operator decision launchd should not undo.

**5. Backstop only:** launchd SIGKILLs at `ExitTimeOut` (900s) if step 4 hangs.

Four things about this that are easy to get wrong:

**Receipts are not flushed by the drain.** Each receipt is written from a
detached task, so a receipt may land after the response it describes, and a stop
does not wait for one. Nothing is lost in a normal exit, but do not read step 4
as a flush barrier.

**A second signal does not escalate.** The shutdown signal resolves once and the
graceful shutdown does not re-arm, so further SIGTERMs and SIGINTs during a drain
are ignored and `kill -9` is the only way to shorten it. That is deliberate: an
accepted generation cannot be aborted on the engine thread, so a signal that
abandoned the drain would discard the work *and* its receipt with the client
still waiting. If you need the process gone now, `kill -9` is the honest way to
say so.

**A stop during startup is immediate.** The drain handler is armed only once the
replica begins serving, and the startup model load happens before that — so a
SIGTERM during the load terminates the process at once (exit `143`, the signal's
default disposition). That is the correct outcome: nothing is in flight, no
generation, no receipt, no client.

**Drain time is a sum, not a maximum.** See below.

### Worst case, honestly

The engine runs one generation at a time and **a posted job cannot be aborted**.
The bounded queue holds 8 waiting jobs plus 1 running, so a replica can have
**9 accepted generations** it is obliged to finish. Each is bounded only by the
engine's wall-clock ceiling of **15 minutes**. Drain time is their **sum**, not
the longest of them, so the true upper bound on step 2 is **9 × 15 = 135
minutes**.

That is also why `ExitTimeOut` cannot be the drain mechanism, and why reaching it
does not mean one generation was still running: at 900s launchd SIGKILLs a
replica that may still owe several. No honest `ExitTimeOut` covers the worst
case, and setting it to 0 (infinity) would stall system shutdown forever.

In practice the binding limit is `max_tokens`, and its default is narrower than
it sounds: **800 is applied when a request omits `max_tokens`**, not a ceiling
imposed on one that asks for more. A client requesting 8000 gets 8000, bounded
only by the 15-minute wall clock. So a typical drain is one generation long and
step 2 returns in seconds, but a single client can make it much longer, and
capping `max_tokens` above the replica is what turns the typical case into a
guaranteed one. **Alert if step 2 exceeds ~15 minutes:** at that point a
generation is sitting on the wall-clock ceiling.

Second-order: cancellation is cooperative, so a client that *disconnects* stops
its generation within one decode step. A load balancer that hard-closes
connections at deregistration shortens the drain dramatically, at the cost of
discarding partial work. Prefer draining connections gracefully.

## Restart and upgrade

```bash
sudo launchctl kickstart -k system/com.camelid.enterprise   # restart in place
sudo deploy/macos/install.sh                                # upgrade the binary
sudo deploy/macos/install.sh --with-gateway                 # …and the gateway
```

Restart with `kickstart -k`, not by signalling the process: a bare `kill` races
the supervisor's own restart logic. Note that every restart re-reads and
re-hashes the whole GGUF before binding, which is why `ThrottleInterval` is 60s
rather than launchd's 10s default.

## Logs and rotation

The two log streams rotate differently, and the stanza is written around that.

**Receipts rotate cleanly.** The attribution middleware re-opens the receipt file
by path for every write, so newsyslog's rename-and-recreate is picked up on the
very next request — no signal, no restart, no lost lines. Nightly or at 10 MB,
two weeks retained, bzip2'd.

Size that threshold against your probe interval, not against your traffic. A
receipt is written for **every** request the replica answers, and a readiness
probe is a request. A `GET /v1/health` line is **about 420 bytes** — three
64-character digests, the host summary, lane, method, path, status, timestamp and
worker width — so at one probe every five seconds probe traffic alone is roughly
**7.3 MB a day**. That sits under the stanza's 10 MB size threshold, but not by
much: any real traffic on top of it crosses the threshold before the nightly
rotation gets there, and a probe every two seconds crosses it on probes alone. On
a quiet replica the receipt log is mostly health checks.

**launchd stdio does not rotate cleanly.** launchd opens `StandardOutPath` and
`StandardErrorPath` once at job spawn and holds the descriptors, so a rotation
while the process is running leaves it appending to the *archive*; once that
archive ages past the retained count it is deleted while still open and the
daemon writes to an unlinked inode. The size-only thresholds in the stanza are
therefore a disk-full backstop, not a schedule — at `RUST_LOG=info` a serialized
replica should never reach them. If one of these ever does rotate,
`launchctl kickstart -k` the owning job to reattach.

newsyslog runs hourly at `:30`, so one hour is the finest granularity available.

## Reboot survival

Two prerequisites live outside the unit, and both fail silently — the unit looks
correct and the machine simply never comes back.

- **FileVault.** If enabled, the boot volume stays locked after an unattended
  reboot and no LaunchDaemon runs until someone authenticates at the console.
  Either leave FileVault off on a racked machine, or use
  `sudo fdesetup authrestart` for every *planned* reboot — which does not help
  after a power cut.
- **Power management.**
  ```bash
  sudo pmset -a sleep 0 disksleep 0 womp 1 autorestart 1
  sudo systemsetup -setrestartpowerfailure on
  ```

## Exposure

The replica serves the routes of
[Replica HTTP Contract v1](../../docs/contracts/replica-http-v1.md) and refuses
everything else with `403` and `"code":"route_not_served"`. It keeps no private
second copy of that list — the route filter reads the dependency-free registry in
`crates/replica-contract`, which is the contract's machine-readable form.
Admitted, with `HEAD` and CORS preflight where they apply:

| Method | Path | Answered by |
|---|---|---|
| `GET` | `/v1/health`, `/v1/models`, `/v1/models/<id>` | the engine |
| `POST` | `/v1/completions`, `/v1/chat/completions` | the engine |
| `POST` | `/v1/embeddings`, `/v1/responses`, `/v1/messages`, `/v1/rerank`, `/v1/reranking` | the engine's typed `501` "unsupported" replies |

The last row is a contractual surface rather than a capability: the pinned engine
answers those paths with a typed `501` and an `unsupported_*` code, and the
contract carries that through so a client SDK's capability probe gets the
engine's own answer instead of a refusal it has to special-case.

**Point your health check at `/v1/health`, not `/health`.** The engine's bare
`/health` is replica-private diagnostics under the contract and answers `403`
here; `/v1/health` is the route the installer, the drain loop and every probe in
this repository poll, and it is the one that reports `generation_ready`.

Not served, and this is the point: the engine's `/api/models/load`,
`/api/models/unload` and `/api/runtime/gpu` are **unauthenticated** on its own
router, and a replica that can be reconfigured over its serving port cannot vouch
for what produced its output. The startup model is loaded through the engine's
own handler **in-process**, with no listener of any kind involved, so from a
client's perspective there is no route that can change the weights and never was
one. The engine's legacy completion-server-compatible routes are refused with the
rest; one of them is a second generation route that attribution does not inject
body fields for.

The two generation routes carry a `model` field, and the engine resolves that
field against the filesystem before anything else, so it is checked too: it may
name this replica's own weights — the id above, the path in `--model`, or that
file's name or stem — or be omitted. Anything else answers `404` with
`"code":"model_not_served"`, identically whether or not a file of that name is on
this disk. Withholding the routes without checking the field would have left the
model swappable by an ordinary completion request, over a route the contract
requires the replica to serve.

`x-camelid-model-sha256` is the external check that the weights have not moved.
It is the SHA-256 of the GGUF this process hashed before it bound its port, and
it is on every response. Compare it against `shasum -a 256` of the file you
intended to serve.

The unit still binds `127.0.0.1`, and that is still the right default:
**the route contract bounds what a caller can ask for, not who may ask.** Anyone
who can reach the port can spend the replica's single generation slot, and the
engine applies a permissive CORS policy to the routes that are served, so they
are reachable from any web origin. Widen `--addr` only behind something that
authenticates, or on an isolated network with a packet filter doing the same.

The replica itself authenticates nothing and is not going to. The gateway can:
`camelid-enterprise-gateway serve --identity-db <path>` rejects any request
without a valid `Authorization: Bearer <token>` with a typed `401`, before the
request takes an admission permit or reaches a replica. Enforcement is **opt-in**
— without `--identity-db` the gateway forwards exactly as before — and the
gateway terminates no TLS. The gateway unit in this directory binds
`0.0.0.0:8080` by design, since it is meant to be the box's one entry point, so a
bearer token would cross that hop in cleartext: put TLS in front of the gateway
before `--identity-db` buys anything on a network you do not control. That unit
ships without `--identity-db`.

## When it will not start

The replica fails closed, and the reason is in `lane.err.log`. The refusal is
printed as written, not as an escaped one-line debug string, so read it whole —
it names every offending variable at once, says what each one does, and prints
the four names the replica does accept. The common causes:

- **A `CAMELID_*` variable it does not recognize.** The unit sets none, so this
  means something else in the boot environment does. Admission is deny-by-default
  over the whole `CAMELID_` prefix: a variable is refused because nobody wrote a
  rule admitting it, not because someone listed it as dangerous.
- **A pre-set canonical key**, e.g. `CAMELID_DETERMINISTIC=1` set by hand in good
  faith. Refused *even at the canonical value*: the lane writes its own
  configuration vector, and the published digest means "the vector this replica
  wrote", which stops being checkable the moment anyone else can write it too.
- **`CAMELID_THREADS`.** An engine key, and one the engine reads for *presence*
  rather than value. This replica's worker count is `--threads`, or
  `CAMELID_ENTERPRISE_THREADS`; the refusal says so.
- **`VECLIB_MAXIMUM_THREADS`, `RAYON_NUM_THREADS` or `RAYON_RS_NUM_CPUS`**, the
  three refused names that are not `CAMELID_*`. All three move the arithmetic;
  the last two are the same lever under two spellings. See
  [Configuring the unit](#configuring-the-unit).
- **A missing or unreadable model.** The path is canonicalized and then read
  whole to compute the published digest, so an unreadable file fails before the
  port is bound rather than during the load.
- **The model changed between being hashed and being loaded.** Replacing the file
  by rename in that window is refused, because the digest the replica is about to
  publish would describe bytes it did not load.
- **`--threads 0`**, or a pool that cannot be sized.

Because `KeepAlive` is `SuccessfulExit: false`, a fail-closed start retries every
`ThrottleInterval` (60s) until a human fixes it. That loop is the intended
behaviour: the replica is saying its configuration is wrong, and the alternative
is serving under one it cannot vouch for.
