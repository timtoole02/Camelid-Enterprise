# Platform datastore

**Status:** decided, not built. Resolves the §7 open question *"What datastore
satisfies both 'one box under a desk' and 'data center scale' without an
external managed dependency?"*

**Decision: PostgreSQL, as the single platform datastore, self-hosted inside
the deployment's trust boundary.** SQLite is retained only where exactly one
process owns the file — a condition nothing in this repo currently satisfies.

---

## 1. What has to be true

From the separation principle (§4), a store is only acceptable if the whole set
still "runs inside one trust boundary, on one machine or one cluster, with no
external dependency." That rules out anything hosted. It does **not** rule out
a database server: a Postgres container next to the gateway is no less
local-first than a file next to it. "No external dependency" is about who holds
the data, not about process count.

What Phase 6 has to store:

- durable aggregation of the two append-only JSONL logs (gateway audit,
  gateway usage) joined to replica receipts on `request_id`
- metering rollups derived from that
- shared quota state — the Phase 4 gap: quota counters are per-process and
  in-memory, so an authenticated deployment running the manifest's two replicas
  admits up to `4 x limit` for one organization across a window boundary

Identity data — principals, organizations, tokens — is deliberately **not** on
that list. It has a store already, and §5 explains why moving it is a separate
decision that Phase 6 does not wait on.

These workloads are multi-writer by construction.

## 2. The evidence already in this repo

The decisive argument is not theoretical. It is the identity crate's bug
history.

`crates/identity` uses SQLite, and two processes share the file: the long-lived
`serve` process and the short-lived CLI. That single fact has already produced
three defects, each still visible in the tree as the fix that answers it:

- a migration race, answered by `initialize_schema` taking the write lock at
  `BEGIN` and re-reading `user_version` under it — because a deferred
  transaction's write upgrade fails with `SQLITE_BUSY`, and the busy handler
  does not retry a write upgrade, so `busy_timeout` never applies
- the same read-then-write hazard in seven runtime write paths, answered by the
  `begin_write` helper they now all route through
- an **unfixed** race creating the database concurrently, mitigated only by a
  comment in `deploy/k8s/gateway-deployment.yaml` telling operators to create
  the file before starting pods

(The ablation measurements behind those fixes — failure counts with and without
the lock — are recorded in PR #17's description rather than in the tree.)

That is two processes on one node, and it is already enough to have produced
three defects. The cluster case is worse in kind, not just in degree: the
gateway manifest ships `replicas: 2`, and the authenticated topology it is
heading for would put those pods on shared state. That configuration is not
shipped today — the stock manifest disables identity and its only volume is a
per-pod `emptyDir` at `/tmp` — so this is a statement about where the
deployment is going, not a defect in what is running. But it is the
configuration the identity work exists to enable, and it would put SQLite
behind a `ReadWriteMany` volume, which in practice means a network filesystem,
where SQLite's locking is documented as unreliable.

**SQLite is not failing here because it is a bad database, and it is not a
"single-process" one.** It supports concurrent access from multiple local
processes through file locking, and on a single node that works. Two things
rule it out for this role. The first is the deployment target: multi-pod access
needs a shared volume, which in practice means a network filesystem, where
those locking guarantees do not hold. The second is demonstrated rather than
theoretical — the transaction and initialization mistakes above are ones this
application has already made, and a store whose correctness depends on every
caller getting `BEGIN IMMEDIATE` and initialization ordering right will keep
collecting them.

## 3. Options considered

| Option | Verdict |
|---|---|
| **SQLite on a shared volume** | **Rejected.** Requires `ReadWriteMany`; SQLite locking over network FS is unsafe; already produced three concurrency defects at two processes on one node. Fails the cluster half outright. |
| **Two backends behind a trait** (SQLite one-box, Postgres cluster) | **Rejected.** Two schemas, two migration paths, two sets of isolation semantics, and double the test matrix — where the failure modes that matter (contention, isolation, deadlock) appear in whichever one is exercised least. It also breaks the rule that the same boundaries serve one box and a data center: it makes them *different* boundaries wearing one interface. |
| **Raft-replicated SQLite** (rqlite, dqlite, libSQL) | **Rejected for now.** Genuinely fits the shape, and keeps SQLite ergonomics. But it adds a consensus system to operate, the write path is still single-leader, and the ecosystem and operational familiarity are far thinner than Postgres. The complexity is not smaller — it is less well understood. Revisit only if the Postgres process count becomes the blocking objection to single-box adoption. |
| **Embedded KV** (redb, sled, RocksDB) | **Rejected.** The workload is relational and analytical: joining gateway audit to replica receipts on `request_id`, rolling metering up by organization and window. That is SQL work. No multi-writer story either. |
| **PostgreSQL, self-hosted** | **Chosen.** Real MVCC and concurrent writers; `SELECT ... FOR UPDATE` and atomic upserts give shared quota counters directly; network protocol, so no shared-filesystem semantics; one schema and one migration path for both deployment shapes; entirely inside the trust boundary. |

## 4. What this costs

Honestly stated, because it is the real objection:

- **The single-box install gains a process.** "One binary and a file" becomes
  "compose up" or a supervised bundle. `deploy/` already ships Docker and
  Kubernetes, so this is an addition to the single-box story, not a rewrite of
  it — but it is a genuine regression in setup friction and should not be
  waved away.
- **Postgres must be operated.** Backups, upgrades, and disk are now the
  operator's problem in a way a file was not. The single-box default should
  therefore ship a working configuration, not a blank one.
- **It is heavier on a desk box.** Idle Postgres is tens of megabytes of RAM.
  Against a deployment that loads multi-gigabyte models, this is not the
  constraint it feels like.

## 5. The identity question this forces

Identity is on SQLite today. Two paths, and they are genuinely different
products:

**(a) Migrate identity to Postgres.** One store, one backup, one migration
path. Costs the zero-setup single-box property.

**(b) Restore single-process ownership and keep SQLite for identity.** SQLite
is only wrong here because the CLI reaches into the file behind the running
server's back. If identity operations moved to an authenticated admin API on
the gateway — `create-user` becoming a request rather than a second process
opening the database — then exactly one process owns the file and SQLite
becomes correct again, including the create-race that is currently unfixed.

(b) is architecturally cleaner than it first looks, and it is the option that
preserves the desk-box story. It is also strictly more work, and it adds a new
authenticated HTTP surface, which is its own review burden. **This document
does not decide between them, and Phase 6 does not wait for that decision.**
The platform store is a new database for aggregation, metering and platform
state; identity keeps its own store either way. The ownership boundary is the
decision that matters: the platform store owns aggregated audit and usage
records, metering rollups and shared quota state, and it does *not* own
principals, organizations or tokens until the identity question is settled
separately. What is not an option is "keep SQLite for the platform store
because identity already uses it".

## 6. What this unblocks

- Phase 6 durable store and log aggregation
- Durable, shared quota state — closing the `4 x limit` gap
- Metering built on the usage log's terminal records rather than on head status

## 7. Bounds

- Nothing here is built. This is a decision, not an implementation.
- Postgres does not make the gateway's logs durable on its own. Those writers
  are best-effort by design, and no shutdown behaviour turns a record the
  gateway dropped while running into one an aggregator can read. Aggregation
  reads what survived; it does not retroactively make it complete. The precise
  durability contract lives with the gateway, in `deploy/README.md`, rather
  than being restated here where it would drift.
- Choosing Postgres says nothing about *when* identity migrates. Until it does,
  the single-writer constraints documented in `crates/identity` still apply.
