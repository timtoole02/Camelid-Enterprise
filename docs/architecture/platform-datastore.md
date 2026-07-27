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
- users, organizations, tokens (today in SQLite)
- shared quota state — the Phase 4 gap: quota counters are per-process and
  in-memory, so the shipped two-replica manifest enforces up to `4 x limit`

Three of those four are multi-writer by construction.

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

That is two processes on one node. The shipped Kubernetes manifest runs
`replicas: 2` against a shared volume, which is the same shape with network
storage underneath — and SQLite's locking is documented-unreliable on network
filesystems, which is what most `ReadWriteMany` CSI drivers are.

**SQLite is not failing here because it is a bad database.** It is failing
because it is a library for a single process that owns a file, and it is being
used as a server. The correct fix for "two processes need consistent concurrent
access to shared state" is a database server.

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
does not decide between them**; it records that the choice is now the only
thing standing between here and Phase 6, and that "keep SQLite because it is
already there" is not one of the options.

## 6. What this unblocks

- Phase 6 durable store and log aggregation
- Durable, shared quota state — closing the `4 x limit` gap
- Metering built on the usage log's terminal records rather than on head status

## 7. Bounds

- Nothing here is built. This is a decision, not an implementation.
- Postgres does not make the gateway's logs durable on its own: the audit and
  usage writers are still best-effort and lossy on process exit. Aggregation
  reads what survived; it does not retroactively make it complete.
- Choosing Postgres says nothing about *when* identity migrates. Until it does,
  the single-writer constraints documented in `crates/identity` still apply.
