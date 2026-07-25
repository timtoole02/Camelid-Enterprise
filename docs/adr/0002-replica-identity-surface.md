# 0002 — What a replica publishes about itself

> **Status: accepted.** Every "**Today**" claim below is verified against the
> current tree and names the file, symbol or test that carries it. Anything not
> yet built is labelled **Not yet**, with what is missing.

**Date:** 2026-07-24
**Applies to:** the `deterministic` lane, engine pin `b4e3a9056567ed8145fc4fa29850d6f1f261ac2b`
**Code:** `crates/server/src/lane.rs`, `crates/server/src/attribution.rs`, `crates/server/src/surface.rs`, `crates/server/src/main.rs`

---

## Context

The deterministic lane's published result is that the same greedy request yields
the identical token stream on every run, within one hardware class and
configuration, across process restarts. A result nobody can check is a slogan,
so the lane publishes a configuration digest — `config_sha256` — and clients are
invited to treat two replicas carrying the same digest as interchangeable.

That invitation was broader than the digest could honour, in four independent
ways.

**The digest was under-defended.** Admission was a ban list: ten named
environment variables the lane refused to start under. The engine reads **265
distinct `CAMELID_*` keys** at this pin and writes eight more it never reads
(`crates/server/tests/fixtures/engine-env-keys.tsv`, 282 rows). A ban list of ten
against a namespace of hundreds does not fail loudly when it is incomplete; it
fails silently, because the keys it misses are exactly the ones nobody
classified. Several of the unclassified ones — the normalization epsilon, rotary
pairing and direction, the attention score scale, linear accumulation order,
feed-forward gate/up ordering, the weight-layout selectors — change arithmetic
the result is stated over, and none of them touches the digest. An operator with
one of them exported got a replica that started cleanly, published the same
twelve hex characters a clean replica publishes, and emitted a different token
stream.

Worse, a ban list *cannot* be completed by trying harder. The engine assembles
some of its key names at runtime by appending a role name to a stem
(`CAMELID_RECTANGULAR_LINEAR_LAYOUT_<ROLE>`). The set of names it answers to is
therefore not enumerable from its source, and any list of forbidden names is
provably partial — including for roles that do not exist yet.

**The digest was over-read.** `config_sha256` is a compile-time constant: it
covers the values the lane writes and the engine revision it writes them for. It
says nothing about the machine and nothing about the worker-pool width — and
worker-pool width was not even applied by this binary. `--threads` reached only
the engine's *reported* execution plan; it sized no pool. Two replicas on
different hardware, or started with different `--threads`, published identical
attribution while being free to differ in output.

**Nothing identified the weights.** Two replicas started with identical flags and
a different `--model` published byte-identical `lane`, `config_sha256` and host
summary while returning different tokens for the same greedy request. The
engine's own `model` field did not distinguish them either — it is the GGUF's
`general.name` metadata string, so a copy of a file with every tensor overwritten
still reports the same name. This was the sharpest form of the failure the whole
surface exists to prevent: not "the digest is narrower than you think" but "two
replicas that agree on every published field disagree on every token".

**The control plane was served on the lane's own port.** The engine this
distribution wraps is a complete local inference application; its router carries
unauthenticated `POST /api/models/load`, `/api/models/unload` and
`/api/runtime/gpu` alongside the OpenAI-compatible API. Bound as-is, any caller
could unload the model, load one of their choosing, or flip the process-global
accelerator flag, and every response before and after carried the same lane,
digest and host string. A startup gate on the environment is worth little when
the model itself is one request away, and it made a model digest meaningless
before one was even written: the file the replica hashed at startup was not
necessarily the file it was serving a minute later.

---

## Decision

### 1. `config_sha256` claims one thing, and the admission policy gets its own digest

`config_sha256` is SHA-256 over each `KEY=VALUE\n` of the frozen configuration
vector **in declaration order**, terminated by `engine_pin=<rev>` with no
trailing newline. It means: **this replica wrote this vector, for this engine
build.** Today it is

```
30d77c2608036f8475372ace9ec125ffc5fa16d8d63f0355a08c32c69f4449b7
```

published as its first twelve characters, `30d77c260803`, on headers and in
completion bodies, and in full on serving receipts.

The admission policy is **not** in that preimage. It is hashed separately, as
`admission_sha256`: one `permit_exact=NAME\n` or `permit_prefix=NAME\n` line per
allow-list rule in declaration order, then one `refuse_foreign=NAME\n` line per
out-of-namespace refusal, terminated by `namespace=CAMELID_` with no trailing
newline. Today that is

```
45121fb83fef631f8464c32dada6100b23f0a0af80347031f812803ee9ec2a09
```

published as its first twelve characters, `45121fb83fef`, on headers and in
completion bodies, and in full on serving receipts — the same three places, at
the same two lengths, as its sibling.

The policy has to be hashed *somewhere*, and the reason is the same one that
motivates the rest of this record: under deny-by-default the allow list **is** the
admission surface. A build that gained one row — say a permit for
`CAMELID_ROPE_PAIRING` — would start cleanly with that variable exported, take a
different rotary pairing through the forward pass, and otherwise publish exactly
what a clean replica publishes. Rule *names* are compile-time constants and
identical on every conforming replica, so hashing them closes that without
costing the constancy that makes a published digest worth comparing. The *values*
behind those names are operator-supplied and stay out, or the digest would vary
per deployment.

**Why a second digest rather than more bytes in the first.** Folding the policy
into `config_sha256` gives a client the same protection and was the obvious move;
it was rejected for four reasons, in descending weight.

1. **It contradicts a rule this codebase already wrote down.** Host identity is
   deliberately excluded from the config vector hash because "config identity and
   host identity are different claims"
   (`crates/server/src/attribution.rs`). Admission policy is a third claim —
   *what this replica would have refused* — so by the rule already in force it
   gets its own field.
2. **It makes the field's own documentation false.** `ConfigVector::sha256` is
   documented as identifying *this exact vector*. Under fusion that sentence
   stops being true: the value moves while the vector stays byte-identical.
3. **A split loses a client nothing and gains them something.** What a consumer
   does with either digest is identical — compare it to a value recorded at
   audit, or compare two replicas to each other. The pair
   `(config_sha256, admission_sha256)` distinguishes exactly what a fused digest
   distinguishes, and additionally says *which* of the two moved.
4. **It keeps a scheduled policy change cheap.** One row on the foreign-refusal
   list carries a documented removal condition (§7). Under fusion, adding and
   later removing it would retire the lane's configuration identity twice for
   reasons having nothing to do with the frozen vector.

Shapes rejected: a rule *count* (`permits=4;foreign=2`) fails the exact attack
that motivates the digest, because swapping one permit for another preserves the
count; a hand-maintained policy version integer is a claim rather than a
measurement and goes stale in silence; publishing the rule names verbatim on the
wire grows unboundedly on every response including streams, and the useful
operation is comparison, which a digest already serves — the full list already
reaches the operator where they need it, inside the refusal message.

**Today.** `admission_sha256` is computed at startup, pinned by
`lane::tests::admission_sha256_is_pinned`, printed in the startup banner, and
carried on `x-camelid-admission-sha256`, in `camelid_admission_sha256` and on
every serving receipt.
`attribution::tests::two_admission_policies_render_differently_everywhere_they_are_published`
is what makes it load-bearing rather than decorative: it drives two contexts that
differ in this field alone, through all three surfaces, and asserts every other
published field stays identical. A digest that is computed and withheld protects
nothing — it lets a client compare two builds only by reading their source, which
is the position the digest exists to improve on.

`config_sha256` deliberately does **not** cover:

| Not covered | Why not |
|---|---|
| Host CPU, ISA extensions, core count | Not knowable at compile time; folding them in destroys the published constant (§5) |
| Worker-pool width | Same reason; see §5 |
| The admission policy | A different claim, with its own digest (above) |
| The *values* the policy admits | Operator-supplied; folding them in would make the digest vary per deployment |
| The model file | Published separately, and varies per deployment by nature; see §2 |
| Anything the engine's planner writes at model load | Structurally unobservable at startup; see §8 |

Falsifiable: `config_sha256` is a pure function of two `const`s and the framing
bytes between them, reading no environment, no host probe, no clock.
`lane::tests::config_sha256_is_pinned` asserts the exact 64-hex value;
`engine_pin_matches_the_dependency_revision` asserts the pin constant still
matches the dependency it names; `the_two_digests_are_independent_claims` fails
if the admission policy is ever folded back into the configuration preimage.

### 2. The weights are published, and then made unable to move

Every response carries `x-camelid-model-sha256`: a plain SHA-256 of the GGUF this
replica loaded, published as twelve hex characters in headers and completion
bodies and as all sixty-four on receipts.

Four properties, each chosen against a cheaper alternative.

**It is over content, not path or name.** `/models/model.gguf` means a different
file on every host, and `general.name` is metadata that copies between files
without a tensor changing. Only the bytes distinguish two sets of weights.

**It is over the whole file, not the tensor block alone.** Tensor-only would be
smaller and is wrong in the direction that matters: the GGUF metadata block
carries the tokenizer vocabulary, the rotary base and scaling, and the
normalization epsilon, all of which change output. A tensor-only digest would
publish one identity for two files that emit different token streams — this
field's purpose, inverted — and would need a canonical serialization no standard
tool can reproduce. Whole-file over-triggers only on inert metadata edits, which
is the safe direction of error: a false "these differ" prompts a look, a false
"these are the same" is an undetected substitution.

**It is a standard digest.** An operator checks it with `shasum -a 256`. A
faster construction — chunked and combined — would have made a several-gigabyte
startup read cheaper to parallelize, at the cost of a value nobody could verify
with a tool they already have.

**It is computed before the port is bound.** There is no window in which the
replica answers a request and cannot say what produced it. Failing there also
fails before a port is claimed rather than leaving a half-started replica holding
one, and the read warms the page cache the model load is about to fault through.

One hardening rides along. The engine loads by path string rather than by handle,
so between hashing and loading there is a window in which a rename could hand the
loader different bytes than the ones about to be published. The digest routine
captures a fingerprint — length, plus device and inode on unix — from the *same
open handle* it hashed, and re-checks it after the load returns. It is not the
primary control: the primary controls are a read-only model mount and §3. It is
here because it is nearly free and because replace-by-rename is the accident a
build pipeline actually produces.

Its limits, stated rather than left to be inferred, because "the digest describes
the bytes we serve" is exactly the sentence a reader will over-extend. It
compares length and, on unix, device and inode, so a **same-length in-place
rewrite** passes it — `dd conv=notrunc` over a live GGUF is not detected. It runs
**once**, immediately after the startup load, so it says nothing about the rest of
the process's life. And off unix it degrades to length alone. Against a writer who
can reach the file, the answer is the read-only mount, not this.

Falsifiable: `attribution::tests::the_model_digest_is_the_sha256_of_the_file_contents`
pins the value against a digest anyone can reproduce with `shasum`;
`the_model_digest_follows_content_not_path` asserts that identical bytes at
different paths are one identity and one changed byte is not;
`two_model_digests_render_differently_everywhere_they_are_published` constructs
the failure this field exists for — everything else about the two replicas
identical; `a_model_swapped_after_it_was_identified_is_refused` swaps a file of
equal length by rename and asserts the refusal, `#[cfg(unix)]` because catching a
same-length replacement is precisely what the inode buys;
`a_model_replaced_at_a_different_length_is_refused_on_every_platform` covers the
part that is not platform-specific; and
`a_same_length_rewrite_in_place_is_not_detected` asserts the limit above, so it
cannot quietly become a claim.

### 3. The replica serves an allow list of routes, and an allow list of model names

`crates/server/src/surface.rs` filters the engine's router down to six request
shapes:

| Method | Path |
|---|---|
| `GET`, `HEAD`, `OPTIONS` | `/health` |
| `GET`, `HEAD`, `OPTIONS` | `/v1/health` |
| `GET`, `HEAD`, `OPTIONS` | `/v1/models` |
| `GET`, `HEAD`, `OPTIONS` | `/v1/models/<one segment>` |
| `POST`, `OPTIONS` | `/v1/completions` |
| `POST`, `OPTIONS` | `/v1/chat/completions` |

Everything else is `403` with an OpenAI-shaped error body carrying
`"code":"route_not_served"`. `403` and not `404`: the route exists and is
withheld, and a `404` sends an operator hunting a version mismatch that is not
there.

An allow list rather than a block list, for the same structural reason admission
is one: the engine's router registers **61 routes resolving to 69 method/path
handler pairs** at this pin (28 `get`, 38 `post`, 3 `delete`), and a later pin
adds more. A list of paths to block goes stale in silence; a list of paths to
serve stays correct across a pin bump by refusing what it has never heard of. A
route a later revision invents arrives refused rather than arriving served and
waiting to be noticed.

Method and path are **one rule**, because the engine overloads paths by method:
`/api/runtime/gpu` is `get(gpu_runtime).post(set_gpu_runtime)` on a single line of
its router, so a path-only filter cannot tell reading the accelerator state from
mutating it.

Three details are load-bearing rather than incidental.

**`HEAD` is admitted wherever `GET` is**, because the engine's `get(...)` routes
answer it and a probe or gateway that issues one is making a request the replica
can serve.

**`OPTIONS` is admitted on the six allowed paths.** The engine applies a
permissive CORS layer *inside* this filter, so the filter sees a browser's
preflight first; refusing it fails the preflight and the real request is never
issued. Admitting it concedes nothing — preflight for a withheld path is still
refused, and the write it precedes is refused on its own rule regardless.

**The one prefix rule admits exactly one further segment.** The engine's route is
`/v1/models/:model`, and its router falls back to an embedded web application. A
looser prefix admits `/v1/models/a/b`, which matches no engine route, reaches the
fallback, and is answered by the application shell with HTTP 200 — a served page
from the surface this filter exists to withhold.

**Withholding routes is necessary and is not sufficient.** One of the routes that
has to stay is itself a weights-loading control. The engine resolves a generation
request's `model` field against the filesystem *first* — `resolve_model_path`
opens with `PathBuf::from(model_id).exists()` — so a request body naming an
existing file loads it on demand, makes it the process's active model, and
answers every later request that names no model at all. Nothing published moves:
the model digest, the host summary and the receipts go on describing the file
hashed at startup. It is also unbounded, and the shipped Kubernetes manifest
mounts the whole model pool read-only at `/models` with a memory limit and
Guaranteed QoS, so naming a handful of files off that mount walks the pod into an
OOM kill over a route the path filter admits.

A path-and-method filter structurally cannot see this, because the request it
arrives in is one the replica has to serve. So there is a second filter,
`pin_generation_to_the_served_model`, over the body of the two admitted
generation routes.

It is an allow list of *names for one file*: the id the engine filed the weights
under, the path the replica was started with, the path it canonicalized and
hashed, and that file's name and stem. Absent or `null` passes through
untouched — the engine then answers from its active model, which with the control
plane withheld is the one this replica hashed. An admitted name is **rewritten to
the engine's own key** before the request continues, and that rewrite is what
makes the property structural rather than a race against the engine's resolution
order: after this filter the field is either absent or an exact key in the
engine's loaded-model map, so the branch that touches the filesystem is
unreachable from a request body at all. Anything else is `404` with
`"code":"model_not_served"`.

Three details are deliberate. **The key is read back from the load's own reply,
never derived here** — the engine picks it from the GGUF's `general.name`, or the
file stem when that metadata is absent, and a copy of that logic in this
repository would go stale at a pin bump and refuse every request naming the model
the replica is serving. **The refusal is byte-identical whether or not the named
file exists**, so the field does not become a way to ask a replica what is on its
disk. **A body too large to buffer is refused** (`413`, `request_too_large`)
rather than forwarded: a body this filter cannot read is one whose `model` field
it cannot check.

The alternative — accepting only the engine's exact key and refusing every other
spelling — was rejected. It refuses the path an operator started the replica
with, which is the first thing they will try, and it buys nothing the rewrite
does not already give.

**How the startup model is loaded.** The engine exposes model loading over HTTP
and nowhere else: its loader is module-private and its state exposes no
in-process entry point, and the engine is pinned read-only. So the load is an
HTTP request — but not one over a socket. The unfiltered router is a `tower`
service, so the load is dispatched into it in-process with
`ServiceExt::oneshot`: same handler, same middleware stack, same shared state, and
no listener anywhere that could accept a connection from anything else.

That is stronger than the alternative considered first, a private ephemeral
loopback listener carrying the unfiltered router and closed before serving
begins. A private listener *shrinks* the window in which the control plane is
reachable; in-process dispatch removes it. It also removes a bind that can fail,
an ephemeral port, a retry loop for the gap before the listener is polled, and a
hand-rolled HTTP parse.

Source-restricting `/api/models/load` to loopback instead was rejected as
strictly weaker exactly where it matters: in a pod every sidecar shares the
network namespace and is "loopback", and on a host bound to `0.0.0.0` so is every
local process. It weakens "nothing can swap the weights" to "no remote client
can", and it leaves the control plane permanently mounted rather than never
mounted.

The load **blocks** before `axum::serve`, so "serving" implies "loaded". The
listener binds *before* the load so an address collision fails in milliseconds
rather than after a multi-gigabyte read. Between bind and serve, connections sit
unanswered in the listen backlog — see §Consequences. The served router is
composed only after the load returns, because the body filter needs the key the
load reports; the unfiltered router is what the loader is handed, and it is never
bound to anything.

What this section does **not** claim: the allow list bounds *what* a caller may
ask for, never *who* may ask. The engine's permissive CORS policy still applies
to every admitted route, so each is readable and postable from any web origin by
anyone who can reach the port. There is no authentication here and none is
implied; keeping the port private remains the deployment's job.

Falsifiable: `surface::tests::the_control_plane_is_not_served` and
`no_unattributed_generation_route_is_served` name the withheld routes;
`the_prefix_rule_admits_exactly_one_segment` pins the segment rule;
`main::tests::every_allow_listed_route_reaches_the_engine_and_is_attributed`
drives the allow list against the engine's **real** router composed exactly as
`serve` composes it, which is what catches a rule naming a route the engine does
not have; `the_embedded_web_application_is_not_reachable_through_the_filter`
asserts the fallback is unreachable, `/v1/models/a/b` included; and
`the_startup_load_reaches_the_engine_handler_without_a_socket` asserts the
in-process dispatch reaches the engine's private loader rather than a router
404 or a panic.

For the body filter, against the engine's real router in the same composed stack:
`main::tests::a_generation_request_cannot_name_another_file_on_disk` posts a real,
readable path — the engine's resolver branches on `exists()`, so a missing path
would pass for the wrong reason — and asserts the refusal;
`a_missing_and_a_present_file_are_refused_identically` compares the two responses
byte for byte; and `every_name_for_the_served_model_reaches_the_engine` asserts
each admitted spelling reaches the engine, pinned to a code only the engine
emits, because "not refused here" is also true of a response that never had a
body. In `surface.rs`,
`an_admitted_alias_is_rewritten_to_the_engines_key` reads the request the engine
would have received, `a_request_naming_the_engines_key_is_forwarded_unchanged`
pins that an exact match is not re-serialized, and
`bodies_this_filter_cannot_check_are_forwarded_for_the_engine_to_reject` fixes
what the filter does *not* claim.

### 4. Machine state is reported, next to the digests and never inside them

Two fields carry what the digests structurally cannot.

**`x-camelid-host`** is the hardware class the guarantee is scoped to:
`os/arch cores=<logical> simd=<features>`, e.g.
`macos/aarch64 cores=8 simd=dotprod+i8mm+neon`. It is produced by the platform
crate's `probe()` and rendered by `HostCapabilities::summary()`.

**`x-camelid-worker-threads`** is the resolved width of the process-global
data-parallel worker pool, as a decimal integer.

"Resolved" is the whole point: the width is **read back from the pool** after
sizing it, never echoed from the flag. A replica that echoed `--threads` would
publish a number that is wrong in exactly the case anyone would check it — when
the flag was absent. Sizing happens immediately after the environment freeze and
before the engine's state is constructed, and both halves of that ordering are
load-bearing: after the freeze because the pool reads its own environment as it
is built, and before the engine state because constructing it spawns the engine's
handle, and the first read of the pool's width fixes it — a sizing call after
that point fails rather than resizes.

The number is the width of the *global* pool. The engine derives narrower or
wider phase-specific pools from it at first generation, so this is not a promise
that every kernel runs at this width; it is the one number those are derived from,
and the number that differs between two replicas started differently.

Two shapes were considered and rejected. A **composite runtime string**
(`threads=8;cores=8;os=…;arch=…;simd=…`) would answer "are these two replicas the
same?" in one string comparison, which is genuinely attractive — but
`x-camelid-host` already publishes four of its five pairs and its exact string is
pinned by tests here and in the gateway contract suite. Replacing a header this
distribution had just shipped, to add one integer, is a restructure without cause,
and it is the same objection that keeps `config_sha256` where it is. A **second
short digest over the machine state** was rejected because it gives one-token
comparison while hiding *which* property differs, and invites confusion with the
configuration digest; the readable fields already compare in one operation each
and say what moved.

`simd` is the field that says which kernels could have run, so its vocabulary has
to be a superset of what the engine actually branches on. A feature the engine
gates a kernel on and the platform probe does not report gives two hosts an
identical string while one of them takes a different code path. `avx512bw` was
exactly that: the engine's AVX-512 VNNI decode kernel requires it alongside
`avx512f` and `avx512vnni`, at six separate gate sites, and the Linux and Windows
probes reported neither it nor `avx` nor `f16c`. The vocabulary is now declared by
a fixture at the pin, `crates/engine-linux/tests/fixtures/engine-kernel-features.tsv`,
which both the Linux and Windows probes assert against — that shared file, and
not any one crate's constant, is what keeps two platform crates on the same
architecture from drifting apart. The fixture classifies each detected feature as
`routes_by_default`, `routes_only_under_a_refused_key` or `reporting_only`, and
records what must **not** be reported and why: the AMX features are read from the
kernel's CPU-flag file rather than from cpuid and drive only the execution plan's
description, so publishing one would widen the identity with a name no kernel
branches on.

**Not yet:** `crates/engine-macos` is not asserted against that fixture. Its
declared set satisfies both bounds the fixture imposes today, but nothing enforces
it, and macOS is the platform actually serving.

Falsifiable: `attribution::tests::headers_on_every_response_body_untouched_off_completion_paths`
pins all five headers; `the_worker_width_renders_the_same_in_both_encodings`
pins the number/string pair; `main::tests::an_unsized_pool_still_publishes_a_resolved_width`
asserts the published width is read from rayon in the no-flag case too, and
`a_zero_width_pool_is_refused_before_anything_is_built` asserts `--threads 0` is
refused rather than silently reinterpreted as "choose for me";
`crates/engine-linux/tests/kernel_feature_coverage.rs` and the test module inside
`crates/engine-windows/src/lib.rs` assert **both** architectures' vocabularies
against the fixture, and neither is gated on the host it describes — they compare
compile-time constants against a checked-in table, so they run on every CI runner
rather than only on the platform they are about. Within that,
`required_is_exactly_the_default_reachable_class` is the guard on the guard: it
fails the one edit that would defeat the file, quietly demoting a routed feature
to `optional` so the coverage assertion stops asking for it.

### 5. Thread count and host are reported, not hashed

The reason is cross-host comparability, and it cuts the way that is easy to get
backwards.

A digest is worth publishing only if it is *constant across every conforming
replica*. That constancy is what lets a client say "digest `30d77c260803` means
the configuration I audited" without knowing anything about the fleet. Fold in
the pool width or the host and the digest becomes `f(configuration, machine)`: an
8-core and a 16-core replica running an identical, audited configuration publish
different digests, and the field stops answering "is this the audited
configuration?" for anyone. Every consumer would need the per-hardware table the
field exists to eliminate.

So the questions get separate fields, and none impersonates another:

- **`config_sha256`** — *is this the configuration I audited?* Constant across a
  conforming fleet. Diffing it detects a configuration or engine-pin change.
- **`admission_sha256`** — *would this replica have refused what I think it
  refuses?* Constant across a conforming fleet. Not yet published (§1).
- **`model_sha256`** — *are these the weights I audited?* Varies per deployment
  by nature.
- **`host`** and **`worker_threads`** — *is this the same machine, at the same
  width?* Vary across a fleet by design. Diffing them detects what the digests are
  deliberately blind to.

The honest consequence, stated plainly because a reader will otherwise assume it
away: **two replicas may publish the same `config_sha256` and still emit
different tokens.** That is not a defect in the digest; it is the digest's scope.
A client that cares about reproducibility compares *all* the fields, not one.

Thread count in particular is reported and never hashed for a second reason worth
recording: the engine documents its prefill matmul as parallelizing over
independent output rows with serial per-row accumulation, so the thread count does
not change that kernel's numeric result. Width is an operational identity, not an
input to the configuration vector. That claim is the engine's, about one kernel,
and this record does not extend it to the whole forward pass — see §Open
questions.

### 6. Declaration order of the configuration vector is load-bearing

Both digests hash their inputs in **declaration order**, never sorted. Sorting the
canonical vector — identical keys, identical values, identical framing — yields
`42c63ead830c…` against the published `30d77c260803…`.

That asymmetry is the whole risk. A tidy-up that sorts a list, changing nothing
about what any replica applies, would silently mint a new public identity for
every replica in the fleet, and clients pinning the old digest would see a
mismatch they cannot explain from any behavioural change, because there is none.

So the order is part of the contract, and the code says so at the hashing site and
on the vector itself. `config_sha256_is_declaration_order_not_sorted` is what
enforces it, and it asserts the sorted counter-value explicitly so a reviewer
reading a failure can tell an intentional pin bump from an accidental sort.

The doc comment on the vector once said the hash was over the sorted list. The
code was right and the comment was wrong; the comment was corrected. **The order
is never changed to match a comment.**

### 7. Admission is deny-by-default

The lane refuses to start if **any** `CAMELID_*` variable is set that is not on an
explicit allow list. Four names are permitted, each carrying a written reason,
each naming a consumer inside this binary or its own test suite:

| Permitted | Consumer |
|---|---|
| `CAMELID_ENTERPRISE_MODEL` | `--model`; the only way to supply a model without argv |
| `CAMELID_ENTERPRISE_ADDR` | `--addr`; container and orchestrator entrypoints own argv |
| `CAMELID_ENTERPRISE_THREADS` | `--threads` |
| `CAMELID_ENTERPRISE_TEST_MODEL` | this workspace's gated tests; read only under `#[cfg(test)]`, never on the serving path |

**Admission is not a claim that a permitted variable cannot move the numerics**,
and one row plainly does: `CAMELID_ENTERPRISE_THREADS` sizes the data-parallel
pool, and several engine kernels reduce differently above width one. Two replicas
started with different values of it emit different tokens while publishing the
same `config_sha256`. That is by design of the list, not a gap in it, and the
rule that decides it is the same one the foreign-refusal bar states from the
other side: **a lever this replica publishes is a lever a client can check; a
lever it neither publishes nor refuses is the hole.** The resolved width is read
back from the pool and published on every response, so this one is checkable, and
refusing an operator the single width control the binary documents would buy no
visibility that the published width does not already give. Anyone tempted to
strengthen "deny-by-default" into "nothing admitted can change the output" should
read §Scope in `README.md` first: reproducibility is scoped per worker-pool
width, and this is why.

The entry bar is a named consumer in a namespace the engine does not touch.
`CAMELID_ENTERPRISE_` is free at the pin, and
`no_permit_rule_overlaps_an_engine_key` is what keeps it that way — it fails if a
permit rule ever names a key the engine reads or writes. No engine key may ever be
admitted: doing so puts this distribution in the business of re-deriving engine
semantics at every pin bump, which is precisely the work deny-by-default was
adopted to stop.

An allow list replaced a ban list for one structural reason: **a ban list cannot
cover key names that do not exist as literals.** The engine builds
`CAMELID_RECTANGULAR_LINEAR_LAYOUT_<ROLE>` from a role string at runtime, so no
enumeration of forbidden names can be complete, and no amount of care makes it
complete. Under deny-by-default that family is closed *by construction* — every
member fails the "is it permitted?" test because no rule was written for it,
including roles a later engine revision invents. The refusal side needs no
pattern arm at all. `the_computed_key_family_is_refused_without_a_rule_for_it`
asserts this against a role that exists nowhere in the engine.

The second reason is maintenance. A ban list must be re-derived against the
engine's key set at every pin bump — 265 read keys to reclassify, with silence as
the failure mode. An allow list survives a bump untouched.

Four consequences of the posture, each deliberate:

- **Canonical keys are refused even at the canonical value.** The lane is the
  sole writer of its own vector. Sole-writership is checkable at startup; "the
  operator happened to agree with us today" is not. It also avoids teaching
  operators to hard-code the vector into unit files, where it becomes a second,
  unversioned copy that drifts at the next pin bump.
- **There is no escape hatch.** No `--allow-env`, no override variable. An
  override that left the published digests unchanged would restore exactly the
  falsification this work exists to prevent. If one is ever demanded, the only
  acceptable form mixes the overridden names into a published digest — and that
  is a separate decision, not a flag.
- **A refusal is one message naming every offender, sorted, names only.** Five
  stray variables must not cost five restarts; a determinism product's own
  diagnostics should diff cleanly; and an unrecognized variable may hold anything
  — a tenant's draft-model path, a token — while refusal text lands in journals
  that ship off-box.
- **The decision is made on presence, never on value.** Several engine overrides
  fire on the variable merely existing, so an empty value still moves the
  numerics. This also removed a latent unsoundness: reading a variable with
  `env::var` returns an error for a non-UTF-8 *value*, so a read-and-compare scan
  silently admitted a variable that was plainly set. The scan reads names as OS
  strings and decodes them lossily, so a name that is not UTF-8 is refused rather
  than panicked on.

**Three exceptions to the one-namespace claim.** A foreign name earns a row only
by clearing all four parts of a bar: it is outside the namespace, so the general
rule does not already cover it; it moves arithmetic the guarantee is stated over;
nothing this replica publishes would reveal that it was set; and it is
*configuration* — a documented input to a library this process already calls —
rather than a way to substitute code.

The third part is what makes a row necessary rather than merely tidy. The fourth
is what stops the list growing without bound, and it is why `LD_PRELOAD` and
`DYLD_INSERT_LIBRARIES` are absent although they clear the first three: whoever
can set one of those in this process's environment can equally replace the
binary, the dependency tree or the model file, so refusing the name moves no
boundary. These rows exist for the operator who trips a lever in good faith. A
refusal aimed at an actor who already controls the process image would buy
nothing and would invite the reader to mistake a startup scan for a sandbox.

- **`VECLIB_MAXIMUM_THREADS`** sizes Apple vecLib's internal BLAS pool. Accelerate
  is the only native library the engine links, and it is reached from several
  macOS sites — `cblas_sgemv` for dense f32 linear rows above a size floor, and
  `cblas_sgemm` from the CPU tensor matmul with no floor and no environment gate
  in front of it. vecLib's documented response to this variable is to change how
  it blocks and parallelizes those calls, which moves f32 summation order. That
  last step is a property of the platform's BLAS, not something the engine pin
  proves; what the pin establishes is that the engine calls into vecLib. Nothing
  this replica publishes reports vecLib's pool width.
- **`RAYON_NUM_THREADS`** sizes the process-global data-parallel pool, and several
  engine kernels take a different reduction structure when the resolved width is
  greater than one.
- **`RAYON_RS_NUM_CPUS`** is the deprecated spelling of that name, and a live one
  at the pinned `rayon-core`: `get_num_threads` falls through to it whenever
  `RAYON_NUM_THREADS` is unset or does not parse as a positive integer. It sizes
  the same pool by the same rule and reaches the same kernels.

**Why both spellings, and why the pair is now kept on a different ground than it
was added on.** Refusing only the current name was a defect rather than a
simplification: the deprecated one bypassed the whole refusal, and a replica
started under it came up at a width a clean environment would not have chosen. A
list that names one spelling of a lever and not another is a hint, not a refusal.
Whether a name has a live synonym is a property of somebody else's crate at the
version this workspace pins, so it is established by reading that crate, and each
spelling it honours gets its own row, its own digest line and its own test.

Part three of the entry bar has meanwhile stopped holding for both rows: §4
ships, the resolved width **is** published on every response, and it is read back
from the pool, so a replica running under either name no longer looks identical
to one that is not. Part three is an entry bar, not a survival condition, and the
rows now stand on a narrower ground stated in their own text — this replica
declares its worker width through one documented flag, and an undocumented second
spelling of that same setting is refused for the reason `CAMELID_THREADS` is.
That is an ergonomic rule, not a soundness one, and saying so is the point: a row
kept for a reason its own bar does not state is a row nobody can audit. The
earlier plan to delete the pair before publishing the digest is therefore
withdrawn rather than deferred; both rows are inside `45121fb83fef`, and removing
them later would land it on
`f3a2b47d322b056e670079125df670c50c25da57e05abccbdc4d76c6f0fa3653`.

Widening the *scanned namespace* remains rejected. A scan that claims one prefix
is easy to reason about; a handful of exact foreign names with written reasons is
not the same thing as claiming a second one.

The classification work done on the old ban list was not discarded: it survives as
an advisory table whose only consumer is the refusal message, so an operator who
trips over `CAMELID_ROPE_PAIRING` is told what it does rather than only that it is
unwelcome. That table decides nothing, is incomplete by construction, and says so
— `every_advisory_row_annotates_a_name_the_scan_refuses` keeps it from acquiring
rows that never print, and `no_advisory_row_annotates_a_permitted_key` keeps it
from warning about a name the replica is telling the operator to use.

The scan runs **exactly once**, before the lane writes anything, and can never be
re-run — see §8.

### 8. What deny-by-default does **not** close

This posture is strong enough that a reader will assume it closes more than it
does. It does not close these, and no future edit should imply otherwise.

**(a) The engine's planner writes environment keys after the scan has run.** At
model load the engine's execution planner sets and removes **42 managed keys** in
the process environment, choosing values from detected host CPU features. Four of
the forty-two are additionally read back as passthrough — they are a subset, not
a further four. A startup scan structurally cannot observe this: it
happens later, and it is the engine configuring itself, which is legitimate. One
of those managed keys is a key the lane's own advisory table warns about, so a
post-load re-scan would refuse on the engine's own writes.

This is why the scan is single-shot, and why that precondition is a doc comment on
the function rather than tribal knowledge. **Do not add a periodic re-scan, a
readiness probe that calls the scan again, or a "verify environment" route.**

The timing is now sharper than it used to be, and in the right direction: §3 made
the startup load blocking, so the planner's writes land during startup, strictly
before the replica serves anything. They are still after the scan.

**(a′) — and the routing those writes select is *not* invisible.** This is the
correction to make before someone reads (a) as "the planner is a black box". The
env writes themselves are not serialized anywhere. But the engine's health
response carries its execution plan, and that plan publishes the detected CPU
features, thread count, selected backend, quantized path, prefill path, decode
path, fallback path and CPU model — on `/v1/health`, which §3 admits. So the
accurate claim is: **the planner's environment writes are unobservable; the kernel
routing they select is reported on a served route.** Two things follow. Refusing
`/api/execution-plan` while serving `/v1/health` hides nothing, so the read-only
carve-out question in §Open questions is about ergonomics, not exposure. And an
operator pointing a probe at `/v1/health` should know it also publishes the host
CPU model string.

**(b) The guarantee is scoped to a hardware class, not to a digest.** The
planner's inputs are host CPU features, and host CPU features are not in
`config_sha256`. The digest identifies the vector the lane applied; it does not
identify the arithmetic the host will produce. §4 is what makes that scope
externally checkable instead of a footnote.

**(c) The scan is a startup gate, not an invariant.** Nothing prevents the engine,
or any dependency, from writing to the environment later in the process. The lane
holds the environment at the one moment it can prove it holds it, and reports
resolved state thereafter.

What (c) used to leave open and no longer does: the *model* was also mutable after
the gate, over `POST /api/models/load` on the replica's own port — **and, after
that route was withheld, over the `model` field of an ordinary
`POST /v1/chat/completions`.** §3 closes both, the first by removing the route and
the second by checking the field. The second is worth recording as its own
failure rather than folding into the first, because it is the shape the mistake
takes: the route filter looked complete, every test of it passed, and the
property it was written to establish was still false. Read (c) as being about
environment writes specifically. It is not a general licence for the replica's
state to drift after startup, and anything else that turns out to be mutable
after the gate deserves the same treatment rather than a footnote here.

---

## Consequences

- **A clean environment is a deployment requirement.** An operator with
  `CAMELID_*` variables exported for the engine's own CLI can no longer start a
  replica from that shell. This is correct, and the "false positive" framing is
  wrong: the alternative is not "it works", it is a replica publishing
  `30d77c260803` while emitting different tokens. The costs are asymmetric — a
  refusal costs one failed start and an error naming the variable; the other
  outcome costs a published result that is quietly untrue.
- **A pre-set canonical key is now refused even at the canonical value.** Nothing
  in this repository relied on the old equality rule, but an operator with
  `CAMELID_DETERMINISTIC=1` in a unit file breaks. The refusal explains who writes
  the vector rather than only that the variable is unwelcome.
- **Two of this workspace's own test variables became unusable in a serving
  shell.** `CAMELID_LLAMA3_MISSING_PRE_GGUF` and `CAMELID_LLAMA3_REFERENCE_GGUF`
  are read by this repository's tokenizer tests *and* by the engine at the pin, so
  permitting them would put an engine key on the allow list. Both get advisory
  rows telling the operator to unset them or use a different shell. Renaming them
  into `CAMELID_ENTERPRISE_` is the real fix and is not done.
- **CLI surface change: `--threads` binds `CAMELID_ENTERPRISE_THREADS`.** The old
  binding was `CAMELID_THREADS`, which the engine also reads — and reads for
  *presence*, not value, so setting it changed the engine's phase-threading shape
  rather than any thread count. The binary's own documented flag must not be a
  variable its own scan refuses; the refusal for `CAMELID_THREADS` names the
  replacement.
- **`--threads` now sizes a pool.** It previously reached only the reported
  execution plan. A flag that names a thread count and sizes nothing is worse than
  no flag, and a published width would have been honest about a width nobody
  chose.
- **Orchestrator break, invisible on the first deploy.** Kubernetes injects
  Docker-link-style variables named from Service names, and this repository ships
  two `camelid-`prefixed Services, so a pod scheduled after both receives
  **sixteen** `CAMELID_`-prefixed variables. The first rollout, whose pods precede
  the Services, starts fine; every rollout after it refuses. The fix is
  `enableServiceLinks: false` on the replica pod spec, and it is now in
  `deploy/k8s/deployment.yaml`. The refusal recognizes the shape and names the fix,
  so the same failure on a Service this repository does not ship is
  self-diagnosing. A suffix allowance was rejected: it is the unbounded-pattern
  reasoning this decision exists to reject, and it would hide a real deployment
  defect. `no_engine_key_is_mistaken_for_a_service_link` checks the heuristic
  against the engine's whole key set, because a false positive here would send an
  operator to fix a pod spec when what they set was an engine override.
- **Receipt lines grew** by the model digest, host and worker width. Accepted: a
  receipt line that does not identify its own replica is a line that has to be
  joined against a log nobody kept.
- **Startup got slower by one full read of the model**, before the port is bound.
  Partly repaid — the read warms the page cache the load then faults through — and
  the digest uses the platform's hardware SHA-256 where one exists.
- **Startup is now visibly two-phase, and this will be reported as a regression if
  it is not documented.** The process prints that it is listening, then goes quiet
  for the length of the load, then serves. That is the point of §3, but a log
  watcher reads the pause as a hang. Note also that between bind and serve the
  port sits in the listen backlog: an HTTP probe fails, correctly, but a
  `tcpSocket` readiness probe added later against the replica would go ready too
  early.
- **A generation request may no longer name an arbitrary model.** A client that
  sends `"model": "gpt-4"`, or any name that is not this replica's, now reads
  `404 model_not_served` where it previously got a generation from whatever the
  replica had loaded — or, if the name happened to be a readable path, a
  generation from *that* file and a replica quietly repointed for the rest of its
  life. Routing by model name belongs above the fleet, not inside a replica that
  serves one model; the refusal names the model that is served so a gateway can
  act on it.
- **Every generation request body is now buffered before it is served.** It was
  already buffered by the engine's own JSON extractor, so the cost is a second
  pass over the bytes rather than a new one, and a body above 64 MiB — the same
  ceiling the attribution middleware applies to responses — is refused with
  `413 request_too_large` instead of being forwarded unchecked. A request that
  names the engine's own key, or no model at all, is forwarded byte for byte;
  only an admitted *alias* is re-serialized.
- **Receipts became trustworthy rather than merely present.** A receipt records
  the startup model digest and nothing re-derives it per request, so before §3's
  body filter a generation served from swapped weights was attributed in the
  audit log to a file that did not produce it — the opposite of what a receipt
  exists for. The fix is upstream of the receipt, in the filter.
- **The startup banner and the ready line grew.** The banner carries
  `admission sha256`, and the ready line names the key the engine filed the
  weights under, because that key is what a request's `model` field may use. Log
  scrapers matching those lines exactly will need updating.
- **The engine's typed "unsupported" replies became refusals.** `/v1/embeddings`,
  `/v1/responses`, `/v1/messages` and the rerank spellings answer `403
  route_not_served` rather than the engine's own answer. A client SDK probing them
  for capability detection reads the refusal instead. That is the consistent
  outcome — this replica does not serve them — but it is a decision, not a side
  effect of the list being short.
- **Some genuinely useful diagnostics went with the control plane.**
  `/api/capabilities`, `/api/execution-plan` and the telemetry stream are refused
  along with the routes that had to go. A read-only carve-out was rejected for
  now; see §Open questions, and note (a′) — the routing information itself is
  still on `/v1/health`.
- **A stop during startup no longer waits out the model read.** The load used to
  be posted to the replica's own serving listener, which made it in-flight work
  there: a stop signal three seconds into startup blocked for the entire
  multi-gigabyte read while the documented drain probe reported an empty queue the
  whole time. In-process dispatch removes that, and it also removes a race in
  which the detached loader could exit the process while the listener was already
  accepting.
- **Errors print with `Display`.** Returning an error from `main` renders it with
  `Debug`, which turns the environment scan's multi-line refusal into one line of
  escaped `\n` inside a wrapper type. Those messages are the operator's entire
  diagnostic.
- **The published `simd=` string widened on x86-64.** Linux and Windows hosts now
  report `avx`, `avx512bw` and `f16c` where present. That is the defect being
  closed rather than a side effect, and no aarch64 string moved — but two replicas
  built either side of this change publish different host strings on the same
  x86-64 hardware.

---

## Pin bump checklist

Bumping the engine dependency retires the lane's public identity. It does not
inherit it. Seven things move in the same change, or the replica misattributes its
own output:

1. **`crates/server/Cargo.toml`** — the dependency `rev`.
2. **`lane::ENGINE_PIN`** — asserted against (1) by
   `engine_pin_matches_the_dependency_revision`. Bumping one without the other
   leaves the digest valid while the lane names an engine it is not running.
3. **`crates/server/tests/fixtures/engine-env-keys.tsv`** — regenerated at the new
   revision. It is a snapshot; against a stale one, both
   `every_canonical_key_is_read_by_the_engine` and
   `no_permit_rule_overlaps_an_engine_key` assert about an engine nobody is
   running. Its `class` column is swept and then corrected by hand, because the
   sweep cannot see a key read through a helper or assembled from a role string.
4. **`crates/engine-linux/tests/fixtures/engine-kernel-features.tsv`** — likewise,
   and its `routing` column specifically cannot be derived from the call site: the
   environment gate that makes a branch unreachable usually sits inside the same
   predicate, and reporting-only calls look identical to routing ones. A bump that
   gives a feature a live gate must move it to `routes_by_default`, which the
   `required_is_exactly_the_default_reachable_class` invariant then promotes to
   `required`.
5. **The served route allow list in `crates/server/src/surface.rs`** — a bump that
   moves a documented client route renames something a probe depends on, and a
   bump that adds a control-plane route adds it *refused*, which is the intended
   default but worth confirming rather than assuming.
6. **`config_sha256`** — it *will* change, because the pin is inside the preimage.
   Update the constant in `config_sha256_is_pinned`, then every published copy:
   this record, `README.md`, `deploy/macos/README.md`. `admission_sha256` does
   **not** move at a pin bump; that is the point of §1. When the admission policy
   itself changes, its published copies are the constant in
   `admission_sha256_is_pinned`, this record, `README.md` and `deploy/README.md`.
7. **The frozen vector itself** — a canonical key the new engine no longer reads
   is dead weight inside a public digest, which is what
   `every_canonical_key_is_read_by_the_engine` exists to catch.

---

## Open questions

**Whether the published copies of the digests should be checked by a test.** Both
values are maintained by hand across this record, `README.md`, `deploy/README.md`
and `deploy/macos/README.md`, and a pin bump or a policy edit is exactly the
moment a hand-maintained copy gets missed. A test that `include_str!`s each file
and asserts the current digest appears in it is cheap and is the idiom this
codebase already uses everywhere else. Note that any such test must cope with a
truncated form in prose. This is more pressing now that there are two digests and
four documents rather than one and three.

**Whether the served surface should carry the admission policy in full.** The
digest answers "is this the policy I audited?" and nothing else; an operator who
gets an unexpected value has to read the source to learn what moved. The full
list already reaches the operator where they need it, inside the refusal message,
but that is only visible to someone who trips the scan. A read-only route
returning the permit and refusal names was rejected with the rest of the
diagnostics surface (below) rather than on its own merits, and is worth
reconsidering separately.

**Which engine keys remain unclassified.** All of them, and that is the design.
Deny-by-default does not classify keys — it refuses everything unnamed, so
exhaustive classification stopped being a prerequisite for safety. The advisory
table annotates a few dozen of the 265 read keys purely so refusals are legible.
It is not coverage, does not aspire to coverage, and growing it is a diagnostics
improvement rather than a safety one. **Do not turn it back into an enforcement
mechanism.**

**Whether the served surface should carry read-only diagnostics.**
`/api/capabilities` and `/api/execution-plan` mutate nothing and would help an
operator debugging a replica. They are refused today because the allow list is
easier to defend when every row is on a documented client or probe path.
Reconsider one route at a time, with the method pinned, and never as "read-only
`/api/*`" — that is a category whose membership the engine controls, not this
distribution.

**Whether reproducibility across thread counts holds beyond the prefill matmul.**
`README.md` scopes the guarantee away from differing thread counts; the engine
documents its prefill matmul as bit-exact across them. Both can be true if the
decode path or another kernel is not, but the two statements should not be left
standing side by side now that the width is a published field. Resolve it by
checking the decode path, not by assertion.

**Whether the model digest should be re-verified while serving.** It is computed
once, before the port binds; §3 removes the routes *and the request field* that
could change which weights answer, and §2 closes replace-by-rename across the
load. Nothing stops the *file* from being rewritten in place underneath a running
replica — a same-length rewrite is not even caught by the load-time fingerprint —
and memory-mapped weights would follow it. The mitigation today is deployment
posture: the macOS layout owns the model as `root` and gives the service account
read only, and the Kubernetes manifests mount it read-only. A periodic re-hash
would cost a full read per interval and still only narrow the window, so it is not
obviously worth it. Recorded because "the digest is a property of the process" is
true only given that posture.

**Whether `crates/engine-macos` should be asserted against the kernel-feature
fixture.** It is the platform actually serving, its probe is conformant today, and
nothing enforces that. See §4.
