//! Lane identity and the frozen configuration vector.
//!
//! A deterministic-lane replica's output guarantee is stated against the engine
//! at a pinned revision *under a frozen configuration vector*. That vector is
//! enforced here at startup: the process environment is admitted by allow list,
//! canonical values are applied, and anything unrecognized is a startup error —
//! the lane fails closed rather than serving under a config it cannot vouch for.
//!
//! Admission is **deny by default**, not a ban list, and the reason is
//! structural rather than a matter of coverage. The engine assembles some of its
//! key names at runtime from a role string, so the set of names it answers to is
//! not enumerable from its source at all — no list of forbidden names can be
//! complete, including for roles a later revision invents. Every name such a
//! list misses is a knob on the forward pass that leaves the published
//! configuration digest unchanged, which is the one failure this module exists
//! to prevent. Listing what is *allowed* inverts that: the list is short, every
//! row names a consumer, and a key nobody thought about is refused because
//! nobody wrote a rule for it.
//!
//! Deny-by-default claims one namespace, and not every lever lives in it: the
//! engine calls into system libraries that read their own environment. Those are
//! handled by a second, deliberately tiny list of exact foreign names —
//! [`REFUSED_FOREIGN`] — whose entry bar is stated there and is stricter than
//! "it changes something".

use sha2::{Digest, Sha256};
use std::borrow::Cow;
use std::collections::BTreeSet;
use std::ffi::OsString;

pub const ENGINE_PIN: &str = "b4e3a9056567ed8145fc4fa29850d6f1f261ac2b";

/// Namespace the engine and this distribution both draw their keys from. The
/// scan claims the whole prefix: everything under it is refused unless
/// [`PERMITTED`] names it.
const NAMESPACE: &str = "CAMELID_";

/// Variables *outside* [`NAMESPACE`] that the scan also refuses, by exact name.
///
/// This is not a widening of the namespace claim — widening it is unbounded work
/// and would refuse half a developer's shell. It is a short list with a hard
/// entry bar, and a name has to clear all four parts of it:
///
///   1. it is outside [`NAMESPACE`], so the general rule does not already cover
///      it — an in-namespace row would be dead weight that still sits in the
///      admission digest, implying a policy the scan does not have;
///   2. it moves arithmetic the output guarantee is stated over;
///   3. nothing this replica publishes would reveal that it was set; and
///   4. it is *configuration* — a documented input to a library this process
///      already chose to call — rather than a way to substitute code.
///
/// The third part is what makes a row necessary rather than merely tidy. A lever
/// that is neither published nor refused is the exact hole deny-by-default was
/// adopted to close; a lever that is published is one a client can check, and
/// refusing one of those costs an operator a working shell to buy nothing.
///
/// The fourth part is what stops the list growing without bound, and it is why
/// `LD_PRELOAD` and `DYLD_INSERT_LIBRARIES` are absent although they clear the
/// first three: they interpose arbitrary code, so whoever can set one in this
/// process's environment can equally replace the binary, the engine's dependency
/// tree or the model file, and refusing the name moves no boundary. These rows
/// exist for the operator who trips a lever in good faith. A refusal aimed at an
/// actor who already controls the process image would buy nothing and would
/// invite the reader to mistake a startup scan for a sandbox.
///
/// Part 3 is an entry bar, not a survival condition, and one row has already
/// outlived it: the resolved pool width *is* published now, so the `RAYON_*`
/// rows no longer clear it. They stay on a narrower ground stated in their own
/// text — this replica declares its worker width through one documented flag, and
/// an undocumented second spelling of that same setting is refused for the reason
/// `CAMELID_THREADS` is. That is an ergonomic rule rather than a soundness one,
/// and saying so is the point: a row kept for a reason its own bar does not state
/// is a row nobody can audit.
///
/// A name and each of its live aliases earn a row apiece, never one row for the
/// family. Whether a name has a synonym is a property of somebody else's crate at
/// the version this workspace pins, so it is established by reading that crate,
/// and every spelling it honours gets its own row, its own digest line and its
/// own test. A list that names one spelling of a lever and not another is not a
/// refusal, it is a hint.
const REFUSED_FOREIGN: &[(&str, &str)] = &[
    (
        "VECLIB_MAXIMUM_THREADS",
        "sizes Apple vecLib's internal BLAS pool. The engine calls into Accelerate on this \
         platform — cblas_sgemv for dense f32 linear rows above a size floor, and cblas_sgemm \
         from the CPU tensor matmul with no floor and no flag in front of it — and vecLib's \
         documented response to this variable is to change how it blocks and parallelizes those \
         calls, which moves f32 summation order. Nothing this replica publishes reports vecLib's \
         pool width, so a replica running under it looks identical to one that is not",
    ),
    (
        "RAYON_NUM_THREADS",
        "sizes the process-global data-parallel worker pool, and several engine kernels take a \
         different reduction structure when the resolved width is greater than one. The width it \
         sets is published — x-camelid-worker-threads is read back from the pool, so a replica \
         running under this looks different from one that is not — and it is refused anyway, \
         because this replica declares its width through one documented flag. Use --threads, or \
         CAMELID_ENTERPRISE_THREADS",
    ),
    (
        "RAYON_RS_NUM_CPUS",
        "the deprecated spelling of RAYON_NUM_THREADS, and a live one at the pinned rayon-core: \
         the pool falls through to it whenever the current name is unset or does not parse as a \
         positive integer, so it sizes the same pool by the same rule and reaches the same \
         kernels. Refused for the same reason, and named separately because refusing only the \
         current spelling would leave the refusal bypassable by a synonym this replica never \
         mentions. Use --threads, or CAMELID_ENTERPRISE_THREADS",
    ),
];

/// Keys the deterministic lane sets to canonical values before the engine
/// reads any of them. The engine pins its forward pass to the order-stable
/// CPU path when `CAMELID_DETERMINISTIC` is on; the explicit `false` entries
/// mirror the engine CLI's own deterministic-mode behavior so the vector is
/// complete even if a key's default changes upstream.
///
/// Editing this list mints a new public lane identity, and a pure reorder does
/// it just as surely as a value change — see [`ConfigVector::sha256`] before
/// touching either the contents or the order.
pub(crate) const CANONICAL: &[(&str, &str)] = &[
    ("CAMELID_DETERMINISTIC", "1"),
    ("CAMELID_NO_GPU_SAMPLE", "1"),
    ("CAMELID_METAL_RESIDENT_DECODE", "false"),
    ("CAMELID_METAL_F32Y", "false"),
    ("CAMELID_METAL_WIRE", "false"),
    ("CAMELID_METAL_WIRE_NSG8", "false"),
    ("CAMELID_METAL_ATTN2", "false"),
    ("CAMELID_METAL_RESIDENT_PREFILL", "false"),
    ("CAMELID_METAL_MM", "false"),
    ("CAMELID_METAL_LINEAR", "false"),
    ("CAMELID_METAL_Q8", "false"),
    ("CAMELID_METAL_Q8_RETAINED", "false"),
    ("CAMELID_HYBRID_Q8_RETAINED", "false"),
    ("CAMELID_METAL_NOCOPY", "false"),
];

/// How a [`Permit`] recognizes a variable name.
enum Match {
    /// The whole name, exactly.
    Exact(&'static str),
    /// Every name under a prefix this distribution owns end to end.
    ///
    /// Unused today, and that is a position rather than an oversight. A prefix
    /// permits names nobody has written yet, including typos, which is the
    /// opposite of a list whose every row names a consumer. It exists because
    /// the *engine* builds key names at runtime, and a distribution that ever
    /// needs to do the same must be able to say so in the data rather than by
    /// bolting a second mechanism onto the scan.
    #[allow(dead_code)]
    Prefix(&'static str),
}

impl Match {
    /// The literal the rule is written with.
    fn literal(&self) -> &'static str {
        match self {
            Match::Exact(rule) | Match::Prefix(rule) => rule,
        }
    }

    /// The tag this rule hashes under in [`compute_admission_sha256`].
    ///
    /// `Exact` and `Prefix` are tagged apart because a prefix admits strictly
    /// more than the exact rule spelled the same way, and two policies that
    /// differ in what they admit must not hash alike.
    fn digest_tag(&self) -> &'static str {
        match self {
            Match::Exact(_) => "permit_exact=",
            Match::Prefix(_) => "permit_prefix=",
        }
    }

    /// Whether the refusal message should print this rule with a trailing `*`.
    /// An operator reading the allow list has to be able to tell "this exact
    /// name" from "anything under this name" without reading the source.
    fn is_prefix(&self) -> bool {
        matches!(self, Match::Prefix(_))
    }
}

/// One admitted variable, and the reason it is admitted.
///
/// The reason is data, not a comment, because the refusal message prints this
/// table: an operator who trips the scan gets the whole answer to "what may I
/// set, then?" from the process that refused them.
struct Permit {
    rule: Match,
    reason: &'static str,
}

/// Every `CAMELID_*` variable this replica tolerates in its startup environment.
///
/// The entry bar is a named consumer inside this binary or its own test suite,
/// in a namespace the engine does not touch — `CAMELID_ENTERPRISE_` is free at
/// the pin, and the fixture-backed overlap test is what keeps it that way. That
/// is why the list is four rows against the engine's hundreds, and why none of
/// the engine's own keys appear on it: admitting an engine key would put this
/// distribution in the business of re-deriving engine semantics at every pin
/// bump, which is the work deny-by-default was adopted to stop doing.
///
/// Admission is **not** a claim that a permitted variable cannot move the
/// numerics, and one row here plainly does: `CAMELID_ENTERPRISE_THREADS` sizes
/// the data-parallel pool, and several engine kernels take a different reduction
/// structure above width one. It is permitted because the width it produces is
/// read back from the pool and published on every response, which is the same
/// test [`REFUSED_FOREIGN`]'s part 3 applies from the other side. The rule the
/// two tables share, stated once: **a lever this replica publishes is a lever a
/// client can check; a lever it neither publishes nor refuses is the hole.**
/// Refusing an operator the one width control this binary documents would buy no
/// visibility that its own published width does not already give.
///
/// These names are inside the published admission digest, **in this order** —
/// see [`ConfigVector::admission_sha256`] before adding, removing or reordering
/// a row. They are deliberately *not* inside the configuration digest: what a
/// replica applies and what it would have refused are two different claims, and
/// fusing them would make a digest documented as "this exact vector" move while
/// the vector stayed byte-identical.
///
/// Absent by deliberate decision, each refused by the general rule with no
/// special case anywhere in the scan:
///   * the [`CANONICAL`] keys — the lane is the sole writer of its own vector,
///     so a pre-set canonical key is refused *even at the canonical value*;
///   * `CAMELID_THREADS` — the engine reads that name too, and reads it for
///     *presence*, so `--threads` must bind `CAMELID_ENTERPRISE_THREADS`;
///   * `CAMELID_MODELS_DIR` — the serving path passes no models directory, so
///     setting it is a silent no-op today; refusing turns silence into an error
///     that says so;
///   * `CAMELID_GATEWAY_ADDR` and `CAMELID_GATEWAY_UPSTREAM` — their consumer is
///     the gateway binary, which is a different process running no scan.
const PERMITTED: &[Permit] = &[
    Permit {
        rule: Match::Exact("CAMELID_ENTERPRISE_MODEL"),
        reason: "GGUF loaded at startup (--model); the only way to supply a model without argv",
    },
    Permit {
        rule: Match::Exact("CAMELID_ENTERPRISE_ADDR"),
        reason: "bind address (--addr); container and orchestrator entrypoints own argv, so \
                 this is how they configure it",
    },
    Permit {
        rule: Match::Exact("CAMELID_ENTERPRISE_THREADS"),
        reason: "worker-thread count (--threads). It moves the numerics — kernels reduce \
                 differently above width one — and is admitted because the width it resolves to \
                 is read back from the pool and published on every response, so two replicas \
                 started differently are told apart from the outside",
    },
    Permit {
        rule: Match::Exact("CAMELID_ENTERPRISE_TEST_MODEL"),
        reason: "local GGUF for this workspace's gated tests; read only under #[cfg(test)] and \
                 never on the serving path — permitted so one shell can run both the tests and \
                 a replica",
    },
];

/// Keys worth explaining when an operator trips over one, keyed by name.
///
/// This table decides nothing. Admission is [`PERMITTED`] and only [`PERMITTED`];
/// what these rows add is the sentence an operator needs to understand *why*
/// their variable was refused, which an allow list structurally cannot express —
/// a deny-by-default rule knows a name is absent, never what setting it would
/// have done.
///
/// It is incomplete by construction and must stay comfortable being incomplete:
/// it annotates a few dozen of the engine's hundreds of keys. An offender with
/// no row here prints as a bare name, which is the correct outcome — the list of
/// refused keys is the engine's key set, and this repository is not the place to
/// mirror it.
const ADVISORY: &[(&str, &str)] = &[
    // Numeric overrides confirmed reachable on the CPU serving path: each one
    // moves arithmetic the guarantee is stated over. `CAMELID_DETERMINISTIC=1`
    // does not gate them — that flag selects the GPU/Metal route, nothing more.
    ("CAMELID_RMS_NORM_EPSILON", "overrides a normalization constant in the forward pass"),
    ("CAMELID_ROPE_PAIRING", "changes rotary-embedding pairing; takes effect on presence alone, empty value included"),
    ("CAMELID_ROPE_DIRECTION", "changes rotary-embedding direction"),
    ("CAMELID_ROPE_POSITION_MODE", "changes how rotary positions are derived"),
    ("CAMELID_ATTENTION_SCORE_SCALE", "overrides the attention score scale"),
    ("CAMELID_GQA_HEAD_MAPPING", "changes grouped-query head mapping"),
    ("CAMELID_LINEAR_ACCUMULATION", "changes linear-layer accumulation, and so reduction order"),
    ("CAMELID_FFN_GATE_UP_ORDER", "reorders the feed-forward gate/up product"),
    ("CAMELID_SQUARE_LINEAR_LAYOUT", "selects a different square-linear weight layout and kernel"),
    ("CAMELID_OUTPUT_PROJECTION_LAYOUT", "selects a different output-projection layout and kernel"),
    ("CAMELID_RECTANGULAR_LINEAR_LAYOUT", "selects a different rectangular-linear layout and kernel; the engine also derives per-role names from this one at runtime, which is why no ban list could have covered the family"),
    ("CAMELID_DEBUG_DISABLE_QK_NORM", "disables QK normalization; takes effect on presence alone"),
    ("CAMELID_PARALLEL_LINEAR", "switches linear layers between serial and parallel reduction"),
    ("CAMELID_PROFILE", "selects an engine profile, which drives what the planner writes into its own managed keys"),
    ("CAMELID_ZERO_ATTENTION_DELTA", "zeroes an attention contribution — a diagnostic, not a serving mode"),
    ("CAMELID_ZERO_FFN_DELTA", "zeroes a feed-forward contribution — a diagnostic, not a serving mode"),
    // Thread-pool overrides. A phase-specific pool underneath the replica's
    // worker count makes any single declared thread count untrue.
    ("CAMELID_PREFILL_THREADS", "installs a prefill-specific thread pool no single declared thread count describes"),
    ("CAMELID_DECODE_THREADS", "installs a decode-specific thread pool no single declared thread count describes"),
    ("CAMELID_THREADS", "engine-owned, and read for its presence rather than its value — setting it changes the engine's phase-threading shape. For this replica's worker count use CAMELID_ENTERPRISE_THREADS"),
    // Route boundaries, batch shape, and features the lane excludes. The
    // guarantee holds at the engine defaults, so each of these is an override
    // of something the vector froze by declining to set it.
    ("CAMELID_PREFILL_CHUNK_TOKENS", "changes prefill batch shape, and so reduction order"),
    ("CAMELID_PREFILL_LAYER_MAJOR_CHUNK_TOKENS", "changes layer-major prefill batch shape, and so reduction order"),
    ("CAMELID_MAC_Q8_PREFILL_I8MM", "moves a quantized prefill kernel boundary; the engine's planner also writes this key itself at model load"),
    ("CAMELID_MAC_Q8_I8MM_SMALL_M_MAX_ROWS", "moves a quantized kernel row threshold, and so which kernel runs"),
    ("CAMELID_X86_Q8_PARALLEL_INPUT_QUANTIZE", "parallelizes input quantization, changing reduction order"),
    ("CAMELID_ATTN_COALESCED", "selects a coalesced attention path"),
    ("CAMELID_SPEC_DECODE", "enables speculative decoding, which this lane excludes"),
    ("CAMELID_SPEC_DRAFT_MODEL", "names a speculative draft model; this lane serves one model"),
    ("CAMELID_SPEC_DRAFT_TOKENS", "sizes speculative drafting, which this lane excludes"),
    ("CAMELID_QUEUE_DEPTH", "resizes the engine queue, changing which requests fail closed"),
    ("CAMELID_GENERATION_TIMEOUT_MS", "changes the per-generation wall-clock ceiling, and at 0 removes it"),
    ("CAMELID_APPLE_ACCELERATE", "turns the Accelerate BLAS route on or off, which changes f32 summation order"),
    ("CAMELID_MODELS_DIR", "not consumed by this replica — the serving path passes no models directory, so setting this changes nothing"),
    // Keys whose consumer is a different process or a different build. Refused
    // here anyway: the scan cannot tell a variable meant for a neighbour from
    // one meant for this replica, and guessing is how a lever gets through.
    ("CAMELID_GATEWAY_ADDR", "read by the gateway binary, not by this replica"),
    ("CAMELID_GATEWAY_UPSTREAM", "read by the gateway binary, not by this replica"),
    // Read by this workspace's own tokenizer tests — and by the engine, at the
    // pin, under the same names. That overlap is why they cannot be permitted:
    // a rule for them would be a rule for an engine key.
    ("CAMELID_LLAMA3_MISSING_PRE_GGUF", "used by this workspace's tokenizer tests, but the engine reads the same name at the pin, so admitting it would put an engine key on the allow list. A shell with it exported cannot start a replica; unset it there, or run the tests in a different shell"),
    ("CAMELID_LLAMA3_REFERENCE_GGUF", "used by this workspace's tokenizer tests, but the engine reads the same name at the pin, so admitting it would put an engine key on the allow list. A shell with it exported cannot start a replica; unset it there, or run the tests in a different shell"),
];

/// The applied, frozen configuration vector.
#[derive(Debug)]
pub struct ConfigVector {
    /// SHA-256 over the canonical `KEY=VALUE` list **in declaration order**,
    /// plus `engine_pin=<rev>`, identifying this exact vector in attribution
    /// headers and serving receipts. Declaration order is load-bearing:
    /// reordering `CANONICAL` changes the digest (see `compute_config_sha256`).
    ///
    /// The admission policy is deliberately **not** in this preimage. Config
    /// identity and admission policy are different claims — "the vector this
    /// replica applied" against "what this replica would have refused" — and
    /// fusing them would make the sentence above false: the value could move
    /// while the vector stayed byte-identical. It is published separately, as
    /// [`ConfigVector::admission_sha256`], which gives a client the same
    /// protection and additionally tells them *which* of the two moved.
    pub sha256: String,

    /// SHA-256 over the admission policy's rule **names**, in declaration order:
    /// one `permit_exact=NAME\n` or `permit_prefix=NAME\n` line per [`PERMITTED`]
    /// rule, then one `refuse_foreign=NAME\n` line per [`REFUSED_FOREIGN`] row,
    /// terminated by `namespace=<NAMESPACE>` with no trailing newline.
    ///
    /// This exists because under deny-by-default the allow list *is* the
    /// admission surface. A build that gained one row — say
    /// `CAMELID_ROPE_PAIRING` — would start cleanly with that variable exported,
    /// take a different rotary pairing through the forward pass, and otherwise
    /// publish exactly what a clean replica publishes. Rule names are
    /// compile-time constants and identical on every conforming replica, so
    /// hashing them closes that without costing the constancy that makes a
    /// published digest worth comparing. The *values* behind those names are
    /// operator-supplied and stay out; folding them in would make the digest
    /// vary per deployment.
    ///
    /// [`ENGINE_PIN`] is deliberately absent from this preimage: the admission
    /// policy is this distribution's, not the engine's, and the configuration
    /// digest already carries the pin.
    ///
    /// Published exactly where its sibling is — `x-camelid-admission-sha256` and
    /// `camelid_admission_sha256` at twelve hex, `admission_sha256` at all
    /// sixty-four on receipts. A digest that is computed and then withheld
    /// protects nothing: it would let a client compare two builds only by
    /// reading their source, which is the position the digest exists to improve
    /// on.
    pub admission_sha256: String,
}

impl ConfigVector {
    pub fn short(&self) -> &str {
        &self.sha256[..12]
    }

    /// The admission digest's published short form, at the same length as the
    /// configuration digest's, so a client comparing the two headers is
    /// comparing two values of one shape.
    pub fn admission_short(&self) -> &str {
        &self.admission_sha256[..12]
    }
}

/// Compare a variable name against a rule string.
///
/// Case-sensitive on unix, where `camelid_foo` is a genuinely different variable
/// that the engine never reads. ASCII-case-insensitive under Windows, mirroring
/// that platform's own environment semantics, so `Camelid_Rope_Pairing` cannot
/// walk past the scan there.
#[cfg(not(windows))]
fn name_eq(name: &str, rule: &str) -> bool {
    name == rule
}

#[cfg(not(windows))]
fn name_starts_with(name: &str, rule: &str) -> bool {
    name.starts_with(rule)
}

#[cfg(windows)]
fn name_eq(name: &str, rule: &str) -> bool {
    name.eq_ignore_ascii_case(rule)
}

/// Compared as bytes, not as a string slice: a lossily-decoded name can carry
/// multi-byte characters, and `&name[..rule.len()]` would panic when the rule
/// length lands mid-character. A scan whose entire purpose is a typed refusal
/// must not have a panic in it — least of all on the leg CI does not build.
#[cfg(windows)]
fn name_starts_with(name: &str, rule: &str) -> bool {
    let (name, rule) = (name.as_bytes(), rule.as_bytes());
    name.len() >= rule.len() && name[..rule.len()].eq_ignore_ascii_case(rule)
}

fn in_namespace(name: &str) -> bool {
    name_starts_with(name, NAMESPACE)
}

/// Dispatch one rule against one name.
///
/// Factored out so [`permitted_by`] and its tests drive the same arm selection.
/// A test that re-implements this match tests its own copy, which is how a
/// production `Prefix` arm can stay unexercised while a test claims to cover it.
fn matches(rule: &Match, name: &str) -> bool {
    match rule {
        Match::Exact(rule) => name_eq(name, rule),
        Match::Prefix(rule) => name_starts_with(name, rule),
    }
}

/// Whether any rule in `rules` admits `name`.
///
/// Takes the table as an argument for one reason: nothing ships a `Prefix` rule
/// today, so a test that wants to exercise that arm end to end has to supply one
/// — and it must reach it through this function, not a copy of it, or the
/// coverage is imaginary.
fn permitted_by(rules: &[Permit], name: &str) -> bool {
    rules.iter().any(|permit| matches(&permit.rule, name))
}

fn permitted(name: &str) -> bool {
    permitted_by(PERMITTED, name)
}

/// The reason an out-of-namespace name is refused, if it is one of the few.
fn foreign_refusal(name: &str) -> Option<&'static str> {
    REFUSED_FOREIGN
        .iter()
        .find(|(key, _)| name_eq(name, key))
        .map(|(_, reason)| *reason)
}

/// Whether this replica refuses to start with `name` set.
pub(crate) fn refused(name: &str) -> bool {
    (in_namespace(name) && !permitted(name)) || foreign_refusal(name).is_some()
}

/// The one-line explanation attached to an offender, if there is one to give.
///
/// A canonical key is answered from [`CANONICAL`] rather than from a row in
/// [`ADVISORY`], so the two can never drift into disagreeing about the lane's
/// own vector. It is also the likeliest way a careful operator trips this scan —
/// setting `CAMELID_DETERMINISTIC=1` by hand, in good faith — and that operator
/// is owed the specific answer, not the general one.
///
/// A [`REFUSED_FOREIGN`] name always has one, and must: it is outside the
/// namespace the scan otherwise claims, so an operator has no way to guess why a
/// replica objected to it without being told.
fn advice(name: &str) -> Option<&'static str> {
    if CANONICAL.iter().any(|(key, _)| name_eq(name, key)) {
        return Some(
            "part of the vector the lane applies itself, so it is refused even at the canonical \
             value — the published configuration digest means \"the vector this replica wrote\", \
             which stops being checkable the moment anyone else can write it too",
        );
    }
    foreign_refusal(name).or_else(|| {
        ADVISORY
            .iter()
            .find(|(key, _)| name_eq(name, key))
            .map(|(_, reason)| *reason)
    })
}

/// Recognize the environment an orchestrator injects for a service whose name
/// starts with this namespace.
///
/// Kubernetes' legacy Docker-link variables are derived from a Service's name,
/// so a Service called `camelid-…` puts eight `CAMELID_*` variables into every
/// pod scheduled after it exists — and this distribution ships two such
/// Services, so a pod scheduled after both receives sixteen. The replica uses
/// none of them, and they are not permitted: a suffix allowance would be exactly
/// the unbounded pattern this design rejects, and would hide the deployment
/// defect instead of naming it. The refusal is correct; what it owes the
/// operator is the fix, because the timing is vicious — the first rollout, whose
/// pods precede the Services, starts fine, and every rollout after it does not.
fn looks_like_a_service_link(name: &str) -> bool {
    let Some(rest) = name.rsplit_once("_SERVICE_").map(|(_, rest)| rest) else {
        return name.ends_with("_PORT")
            || name
                .rsplit_once("_PORT_")
                .is_some_and(|(_, rest)| rest.contains("_TCP") || rest.contains("_UDP"));
    };
    rest == "HOST" || rest == "PORT" || rest.starts_with("PORT_")
}

/// Decide admission for one environment, without touching the process's own.
///
/// Split out from [`apply_deterministic`] so the policy is testable: `set_var`
/// is process-global and `cargo test` runs tests as threads of one process, so a
/// policy exercised only through the real environment can only be tested in ways
/// that flake. Everything here is a pure function of its argument.
///
/// Presence, not value, is the trigger. Several engine overrides fire on the
/// variable merely existing, so `CAMELID_ROPE_PAIRING=` with an empty value
/// still moves the numerics, and the name is what the operator has to act on
/// regardless. Names are also all this reports: an unrecognized variable may
/// hold anything — a tenant's draft-model path, a token — and refusal text lands
/// in journals that ship off-box.
fn refuse_unpermitted<I, K, V>(vars: I) -> Result<(), String>
where
    I: IntoIterator<Item = (K, V)>,
    K: Into<OsString>,
{
    let mut offenders = BTreeSet::new();
    for (name, _) in vars {
        let name: OsString = name.into();
        // Lossy is deliberate, and it is also a correctness fix rather than a
        // convenience: `env::var` returns `Err` for a non-UTF-8 *value*, so a
        // read-and-compare scan silently admits a variable that is plainly set.
        // A non-UTF-8 name is still a name the engine may match, and refusing to
        // start beats panicking on the way to saying so.
        let name = match name.to_string_lossy() {
            Cow::Borrowed(name) => name.to_owned(),
            Cow::Owned(name) => name,
        };
        if refused(&name) {
            offenders.insert(name);
        }
    }
    if offenders.is_empty() {
        return Ok(());
    }

    // Every offender at once, sorted and deduplicated: an operator with five
    // stray variables must not need five restarts to find them, and a
    // determinism product's own diagnostics should diff cleanly.
    let mut message = String::from(
        "deterministic lane refuses to start: the output guarantee is stated against a frozen \
         configuration vector, and an unrecognized engine variable can move the numerics without \
         changing the published configuration digest — so the lane fails closed rather than \
         serve under a configuration it cannot vouch for.\n\n\
         Unset these, then start again:\n",
    );
    for name in &offenders {
        message.push_str("  ");
        message.push_str(name);
        if let Some(reason) = advice(name) {
            message.push_str(" — ");
            message.push_str(reason);
        } else if looks_like_a_service_link(name) {
            message.push_str(
                " — looks like a Kubernetes service-link variable, injected from a Service whose \
                 name shares this prefix; set enableServiceLinks: false on the pod spec",
            );
        }
        message.push('\n');
    }
    message.push_str(
        "\nThis replica reads exactly these, and nothing else in the namespace is admitted \
         — including the lane's own canonical keys, which the lane sets itself:\n",
    );
    for permit in PERMITTED {
        message.push_str("  ");
        message.push_str(permit.rule.literal());
        if permit.rule.is_prefix() {
            message.push('*');
        }
        message.push_str(" — ");
        message.push_str(permit.reason);
        message.push('\n');
    }
    message.push_str(
        "\nRefused from outside that namespace as well, because they move the numerics and this \
         replica is not the one that set them:\n",
    );
    for (name, _) in REFUSED_FOREIGN {
        message.push_str("  ");
        message.push_str(name);
        message.push('\n');
    }
    Err(message)
}

/// Compute the configuration-vector digest: SHA-256 over each `KEY=VALUE\n` of
/// `CANONICAL` **in declaration order**, then `engine_pin=<ENGINE_PIN>`.
/// Declaration order is part of the contract — reordering `CANONICAL` silently
/// changes the published hash. Kept free of environment access so the digest can
/// be pinned by a test without the fail-closed startup side effects.
pub(crate) fn compute_config_sha256() -> String {
    let mut hasher = Sha256::new();
    for (key, value) in CANONICAL {
        hasher.update(key.as_bytes());
        hasher.update(b"=");
        hasher.update(value.as_bytes());
        hasher.update(b"\n");
    }
    hasher.update(b"engine_pin=");
    hasher.update(ENGINE_PIN.as_bytes());
    format!("{:x}", hasher.finalize())
}

/// Compute the admission-policy digest — see [`ConfigVector::admission_sha256`]
/// for the preimage and for why it is a separate published value rather than
/// more bytes fed into [`compute_config_sha256`].
///
/// Env-free for the same reason its sibling is: the policy is a compile-time
/// constant, so its identity must be pinnable by a test that does not need the
/// process environment to be in any particular state.
pub(crate) fn compute_admission_sha256() -> String {
    let mut hasher = Sha256::new();
    // Declaration order throughout, never sorted — the same rule, and the same
    // reason, as CANONICAL's.
    for permit in PERMITTED {
        hasher.update(permit.rule.digest_tag().as_bytes());
        hasher.update(permit.rule.literal().as_bytes());
        hasher.update(b"\n");
    }
    for (name, _) in REFUSED_FOREIGN {
        hasher.update(b"refuse_foreign=");
        hasher.update(name.as_bytes());
        hasher.update(b"\n");
    }
    // The namespace claim is part of the policy: the same rules over a different
    // prefix would refuse a different world.
    hasher.update(b"namespace=");
    hasher.update(NAMESPACE.as_bytes());
    format!("{:x}", hasher.finalize())
}

/// Apply the deterministic lane's configuration vector, failing closed on any
/// `CAMELID_*` variable the replica does not admit.
///
/// **Call this exactly once, before anything else writes to the environment.**
/// The scan's precondition is that the lane has not yet written, which is true
/// exactly once per process and cannot be re-established. Two independent
/// reasons, both of which make a re-scan refuse on writes that are entirely
/// legitimate:
///
///   1. this function itself sets the canonical keys, which are not permitted —
///      the lane is the sole writer of its own vector, so the values it writes
///      are not values it accepts from anyone else;
///   2. at model load, long after the scan, the engine's execution planner sets
///      and removes several dozen of its own managed keys from detected host CPU
///      features. One of them is a key this module warns about by name.
///
/// So a startup scan structurally cannot observe what the planner does, and must
/// not pretend to. This is a stated limit of the startup gate, not a gap to be
/// closed here by adding a periodic re-scan, a readiness probe that calls this
/// again, or a "verify environment" route.
pub fn apply_deterministic() -> Result<ConfigVector, String> {
    // Snapshot before writing anything: this is the operator's environment, and
    // it is the only moment at which it can still be seen. If the canonical loop
    // below ran first, the lane's own writes would be indistinguishable from an
    // operator's, and "refused even at the canonical value" could not be
    // enforced at all.
    let snapshot: Vec<_> = std::env::vars_os().collect();
    refuse_unpermitted(snapshot)?;

    // Only now, with admission settled, does the lane write its own vector.
    for (key, value) in CANONICAL {
        std::env::set_var(key, value);
    }
    Ok(ConfigVector {
        sha256: compute_config_sha256(),
        admission_sha256: compute_admission_sha256(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The engine's key set at the pin.
    ///
    /// A compile-time include, so a missing or renamed fixture is a build error
    /// rather than a test that quietly passes over an empty table.
    const ENGINE_KEYS_TSV: &str = include_str!("../tests/fixtures/engine-env-keys.tsv");

    struct EngineKey {
        name: &'static str,
        class: &'static str,
    }

    fn engine_keys() -> Vec<EngineKey> {
        ENGINE_KEYS_TSV
            .lines()
            .map(str::trim_end)
            .filter(|line| !line.is_empty() && !line.starts_with('#'))
            .skip(1) // column header
            .map(|line| {
                let mut fields = line.split('\t');
                let name = fields.next().expect("fixture row has a key");
                let class = fields.next().expect("fixture row has a class");
                EngineKey { name, class }
            })
            .collect()
    }

    /// Names the engine has spoken for: anything it reads or writes, plus names
    /// it only mentions, which include the fragments it assembles keys from.
    fn engine_owned() -> Vec<&'static str> {
        engine_keys()
            .into_iter()
            .filter(|key| key.class != "DOC_ONLY")
            .map(|key| key.name)
            .collect()
    }

    fn env_of(names: &[&str]) -> Vec<(OsString, OsString)> {
        names
            .iter()
            .map(|name| (OsString::from(*name), OsString::from("1")))
            .collect()
    }

    /// The offenders block of a refusal, without the two tables printed after
    /// it.
    ///
    /// Every test that asserts "the refusal explains this offender" has to look
    /// here and nowhere else. The trailing sections echo `PERMITTED` and
    /// `REFUSED_FOREIGN` unconditionally, reasons included, so a `contains`
    /// against the whole message can be satisfied by text that has nothing to do
    /// with the variable the operator set — which is how a test can claim to
    /// cover an annotation that never prints.
    fn offenders_of(err: &str) -> &str {
        err.split("\nThis replica reads exactly these").next().expect("split yields a head")
    }

    /// The tables printed after the offenders: the allow list, then the foreign
    /// refusals.
    fn tables_of(err: &str) -> &str {
        err.split_once("\nThis replica reads exactly these")
            .expect("the refusal echoes the allow list")
            .1
    }

    /// Pins the deterministic lane's configuration-vector digest. If this value
    /// changes, the config vector changed: every replica's
    /// `x-camelid-config-sha256` header and every serving receipt move with it,
    /// and comparisons against any prior baseline (including cross-host) break.
    /// Update this deliberately, never by accident.
    #[test]
    fn config_sha256_is_pinned() {
        assert_eq!(
            compute_config_sha256(),
            "30d77c2608036f8475372ace9ec125ffc5fa16d8d63f0355a08c32c69f4449b7",
        );
    }

    /// The digest is over `CANONICAL` in DECLARATION order, not sorted order.
    /// Guards against a well-meaning "sort the keys for stability" refactor,
    /// which would silently republish a different hash under the same intent.
    #[test]
    fn config_sha256_is_declaration_order_not_sorted() {
        let mut sorted = CANONICAL.to_vec();
        sorted.sort();
        let mut hasher = Sha256::new();
        for (key, value) in &sorted {
            hasher.update(key.as_bytes());
            hasher.update(b"=");
            hasher.update(value.as_bytes());
            hasher.update(b"\n");
        }
        hasher.update(b"engine_pin=");
        hasher.update(ENGINE_PIN.as_bytes());
        let sorted_digest = format!("{:x}", hasher.finalize());
        assert_ne!(compute_config_sha256(), sorted_digest);
        // Documents the counterfactual: sorting yields this digest instead.
        assert_eq!(&sorted_digest[..12], "42c63ead830c");
    }

    /// Adding, removing or reordering a rule in `PERMITTED` or `REFUSED_FOREIGN`
    /// changes what this replica would admit, and so changes this value. That is
    /// the point — under deny-by-default the allow list *is* the admission
    /// surface — but it means the value is an identity with the same retirement
    /// cost as the configuration digest, and every published copy moves with it:
    /// this constant, `docs/adr/0002-replica-identity-surface.md`, `README.md`
    /// and `deploy/README.md`. Update it deliberately, never by accident.
    #[test]
    fn admission_sha256_is_pinned() {
        assert_eq!(
            compute_admission_sha256(),
            "45121fb83fef631f8464c32dada6100b23f0a0af80347031f812803ee9ec2a09",
        );
    }

    /// The digest is over rule *names*, and the reasons beside them are prose
    /// for an operator. A row's text may be rewritten without retiring a public
    /// identity; the name may not.
    #[test]
    fn the_admission_digest_covers_rule_names_and_not_their_prose() {
        let mut hasher = Sha256::new();
        for permit in PERMITTED {
            hasher.update(permit.rule.digest_tag().as_bytes());
            hasher.update(permit.rule.literal().as_bytes());
            hasher.update(b"\n");
        }
        for (name, _) in REFUSED_FOREIGN {
            hasher.update(b"refuse_foreign=");
            hasher.update(name.as_bytes());
            hasher.update(b"\n");
        }
        hasher.update(b"namespace=");
        hasher.update(NAMESPACE.as_bytes());
        assert_eq!(compute_admission_sha256(), format!("{:x}", hasher.finalize()));
        assert!(
            PERMITTED.iter().all(|permit| !permit.reason.is_empty())
                && REFUSED_FOREIGN.iter().all(|(_, reason)| !reason.is_empty()),
            "every row owes an operator a reason, even though the digest does not carry it"
        );
    }

    /// The two digests answer two different questions, so they must not be the
    /// same bytes and neither may be derivable from the other. Concretely: this
    /// is what fails if someone "simplifies" by folding the admission policy back
    /// into the configuration preimage, which would retire a published lane
    /// identity for a change that alters no applied value.
    #[test]
    fn the_two_digests_are_independent_claims() {
        assert_ne!(compute_config_sha256(), compute_admission_sha256());
        // The configuration digest is a pure function of CANONICAL and the pin.
        // If the admission policy leaked into it, this would not hold, because
        // the local recomputation below knows nothing about PERMITTED.
        let mut hasher = Sha256::new();
        for (key, value) in CANONICAL {
            hasher.update(key.as_bytes());
            hasher.update(b"=");
            hasher.update(value.as_bytes());
            hasher.update(b"\n");
        }
        hasher.update(b"engine_pin=");
        hasher.update(ENGINE_PIN.as_bytes());
        assert_eq!(compute_config_sha256(), format!("{:x}", hasher.finalize()));
    }

    /// The pin is written down in two places that nothing keeps in step. Bump the
    /// dependency without bumping `ENGINE_PIN` and the digest stays valid while
    /// the lane attributes its output to an engine it is not running.
    ///
    /// Whitespace is stripped before matching. TOML is free to write `rev="…"`,
    /// `rev = "…"`, or the key on its own line, and a formatter that picks a
    /// different one of those is not a pin mismatch — but a raw substring match
    /// would report it as one, sending a reader to look for a discrepancy that
    /// does not exist.
    #[test]
    fn engine_pin_matches_the_dependency_revision() {
        let manifest: String = include_str!("../Cargo.toml")
            .chars()
            .filter(|c| !c.is_whitespace())
            .collect();
        assert!(
            manifest.contains(&format!("rev=\"{ENGINE_PIN}\"")),
            "ENGINE_PIN is {ENGINE_PIN}, which is not the revision the camelid dependency is \
             pinned to in crates/server/Cargo.toml. The lane would attribute its output to an \
             engine build it is not running."
        );
    }

    /// The whole point of the allow list is that it contains nothing the engine
    /// answers to. A rule that overlaps an engine key hands an operator a knob on
    /// the forward pass while the replica keeps publishing the same digest.
    #[test]
    fn no_permit_rule_overlaps_an_engine_key() {
        let engine_owned = engine_owned();
        for permit in PERMITTED {
            let rule = permit.rule.literal();
            assert!(
                rule.starts_with(NAMESPACE),
                "permit rule {rule} is outside the {NAMESPACE} namespace the scan claims, so it \
                 can never match and is dead"
            );
            match permit.rule {
                Match::Exact(_) => assert!(
                    !engine_owned.iter().any(|key| name_eq(key, rule)),
                    "{rule} is on the allow list, but the engine reads or writes that key at the \
                     pin. Permitting an engine key makes this distribution responsible for \
                     re-deriving engine semantics at every pin bump, which is what \
                     deny-by-default exists to avoid. Give the flag a CAMELID_ENTERPRISE_ name \
                     instead."
                ),
                Match::Prefix(_) => {
                    let covered: Vec<_> = engine_owned
                        .iter()
                        .filter(|key| name_starts_with(key, rule))
                        .collect();
                    assert!(
                        covered.is_empty(),
                        "prefix rule {rule}* admits engine keys at the pin: {covered:?}. A prefix \
                         may only claim a namespace this distribution owns end to end."
                    );
                }
            }
        }
    }

    /// A canonical entry the engine does not read is dead weight inside the
    /// public digest — either a typo, or a key the pin dropped.
    #[test]
    fn every_canonical_key_is_read_by_the_engine() {
        let read: Vec<_> = engine_keys()
            .into_iter()
            .filter(|key| key.class == "READ")
            .map(|key| key.name)
            .collect();
        for (key, _) in CANONICAL {
            assert!(
                read.iter().any(|name| name_eq(name, key)),
                "{key} is in the frozen configuration vector, and so inside the published digest, \
                 but the engine does not read it at the pin. Either the name is wrong, or the pin \
                 moved and the vector needs revisiting."
            );
        }
    }

    /// Guards the guard: the two assertions above are vacuously true against an
    /// empty or malformed table, so pin the shape of the table itself.
    ///
    /// `DOC_ONLY` is checked because it is the one class `engine_owned` filters
    /// out. A key misfiled there is invisible to the overlap test — which is not
    /// hypothetical: the sweep that produced this table misfiled exactly one key
    /// that way, and it was the flag gating the route the lane's single foreign
    /// refusal is about.
    #[test]
    fn engine_key_fixture_is_intact() {
        let keys = engine_keys();
        assert!(
            keys.len() > 250,
            "engine key fixture parsed {} rows; the sweep at the pin recorded far more, so the \
             file or the parser is wrong and every assertion made against it is vacuous",
            keys.len()
        );
        assert!(
            keys.iter().all(|key| key.name.starts_with(NAMESPACE)),
            "every fixture row should be a {NAMESPACE} key"
        );
        assert!(
            keys.iter().all(|key| key.class != "DOC_ONLY"),
            "a DOC_ONLY row is excluded from engine_owned() and so from the overlap guard. If a \
             key genuinely only appears in engine prose, say why in the fixture header first."
        );
        for anchor in [
            "CAMELID_DETERMINISTIC",
            "CAMELID_ROPE_PAIRING",
            "CAMELID_THREADS",
            "CAMELID_APPLE_ACCELERATE",
        ] {
            assert!(
                keys.iter().any(|key| key.name == anchor),
                "{anchor} is missing from the engine key fixture"
            );
        }
    }

    #[test]
    fn a_clean_environment_is_admitted() {
        refuse_unpermitted(Vec::<(OsString, OsString)>::new()).expect("empty environment");
        refuse_unpermitted(env_of(&["PATH", "HOME", "RUST_LOG", "LANG"]))
            .expect("no CAMELID_ names present");
    }

    #[test]
    fn every_permitted_name_is_admitted() {
        for permit in PERMITTED {
            let name = match permit.rule {
                Match::Exact(rule) => rule.to_owned(),
                Match::Prefix(rule) => format!("{rule}EXAMPLE"),
            };
            refuse_unpermitted(env_of(&[&name]))
                .unwrap_or_else(|err| panic!("{name} is on PERMITTED but was refused:\n{err}"));
        }
    }

    /// A hypothetical prefix rule must actually admit its namespace, tag itself
    /// apart in the digest, and print with its `*` — otherwise the `Prefix` arms
    /// could all be broken and no test would notice, since nothing ships a
    /// prefix rule today.
    ///
    /// Every assertion here drives production code with a locally-built rule:
    /// `matches` for dispatch, `permitted_by` for the lookup the scan performs,
    /// `digest_tag`/`literal` for the digest preimage, `is_prefix` for the
    /// message. A test that re-implements any of those passes on its own
    /// re-implementation and says nothing about the code that ships — which is
    /// the specific way a prefix test can look green over a broken arm.
    #[test]
    fn a_prefix_rule_is_exercised_through_the_production_dispatch() {
        let rule = Match::Prefix("CAMELID_ENTERPRISE_LOCAL_");
        assert!(matches(&rule, "CAMELID_ENTERPRISE_LOCAL_ANYTHING"));
        assert!(matches(&rule, "CAMELID_ENTERPRISE_LOCAL_"));
        assert!(!matches(&rule, "CAMELID_ENTERPRISE_LOCALE"));
        assert!(!matches(&rule, "CAMELID_ROPE_PAIRING"));

        // The exact arm, through the same dispatch, so the two cannot be
        // silently swapped without this failing.
        let exact = Match::Exact("CAMELID_ENTERPRISE_LOCAL_");
        assert!(matches(&exact, "CAMELID_ENTERPRISE_LOCAL_"));
        assert!(!matches(&exact, "CAMELID_ENTERPRISE_LOCAL_ANYTHING"));

        // The lookup the scan actually performs, over a table that has a prefix
        // rule in it. `permitted_by` is the production function; `permitted` is
        // it applied to `PERMITTED`.
        let table = &[
            Permit {
                rule: Match::Prefix("CAMELID_ENTERPRISE_LOCAL_"),
                reason: "test-only table",
            },
            Permit {
                rule: Match::Exact("CAMELID_ENTERPRISE_MODEL"),
                reason: "test-only table",
            },
        ];
        assert!(permitted_by(table, "CAMELID_ENTERPRISE_LOCAL_ANYTHING"));
        assert!(permitted_by(table, "CAMELID_ENTERPRISE_MODEL"));
        assert!(!permitted_by(table, "CAMELID_ENTERPRISE_MODELS"));
        assert!(!permitted_by(table, "CAMELID_ROPE_PAIRING"));

        // The shipped table has no prefix rule, so the same name is refused
        // there. This is what makes the assertions above a statement about the
        // arm rather than about the name.
        assert!(!permitted("CAMELID_ENTERPRISE_LOCAL_ANYTHING"));

        // A prefix hashes under its own tag: a prefix rule admits strictly more
        // than the exact rule spelled the same way, so the two must not produce
        // the same admission digest.
        assert_eq!(rule.digest_tag(), "permit_prefix=");
        assert_eq!(exact.digest_tag(), "permit_exact=");
        assert_eq!(rule.literal(), exact.literal());
        assert!(rule.is_prefix() && !exact.is_prefix());
    }

    /// The scan reads names out of the process environment, where a name is a
    /// byte string and need not be UTF-8. It decodes lossily rather than
    /// panicking, and the decision is made on *presence*, so the value never
    /// enters into it — `env::var` would have returned `Err` for both of these
    /// and admitted a variable the engine can still match.
    #[test]
    #[cfg(unix)]
    fn a_non_utf8_variable_name_is_still_scanned() {
        use std::os::unix::ffi::OsStringExt;

        // CAMELID_ROPE_PAIRING with a lone continuation byte appended: not a
        // permitted name, so refused, and the refusal must not panic on its way
        // to saying so.
        let mut raw = b"CAMELID_ROPE_PAIRING".to_vec();
        raw.push(0xff);
        let err = refuse_unpermitted(vec![(
            OsString::from_vec(raw),
            OsString::from_vec(vec![0xff, 0xfe]),
        )])
        .expect_err("an undecodable name in the namespace is still refused");
        assert!(
            err.contains("CAMELID_ROPE_PAIRING"),
            "the refusal should name what it can of the variable:\n{err}"
        );

        // A permitted name is admitted no matter what its value holds. This is
        // the same code path that used to admit a *refused* name whose value was
        // not UTF-8, which is why presence is the trigger.
        refuse_unpermitted(vec![(
            OsString::from("CAMELID_ENTERPRISE_MODEL"),
            OsString::from_vec(vec![0xff, 0xfe]),
        )])
        .expect("admission is decided on the name, never the value");
    }

    /// Numeric overrides the engine reads on the CPU serving path. Each is
    /// reachable, none is gated by `CAMELID_DETERMINISTIC`, and none was
    /// classified by the ban list this allow list replaced.
    const REFUSED_NUMERIC_KEYS: &[&str] = &[
        "CAMELID_RMS_NORM_EPSILON",
        "CAMELID_ROPE_PAIRING",
        "CAMELID_ROPE_DIRECTION",
        "CAMELID_ROPE_POSITION_MODE",
        "CAMELID_ATTENTION_SCORE_SCALE",
        "CAMELID_GQA_HEAD_MAPPING",
        "CAMELID_LINEAR_ACCUMULATION",
        "CAMELID_FFN_GATE_UP_ORDER",
        "CAMELID_SQUARE_LINEAR_LAYOUT",
        "CAMELID_OUTPUT_PROJECTION_LAYOUT",
        "CAMELID_RECTANGULAR_LINEAR_LAYOUT",
        "CAMELID_DEBUG_DISABLE_QK_NORM",
        "CAMELID_PREFILL_THREADS",
        "CAMELID_DECODE_THREADS",
        "CAMELID_PARALLEL_LINEAR",
        "CAMELID_PROFILE",
        "CAMELID_ZERO_ATTENTION_DELTA",
        "CAMELID_ZERO_FFN_DELTA",
        "CAMELID_THREADS",
    ];

    #[test]
    fn numeric_overrides_are_refused_and_named() {
        for key in REFUSED_NUMERIC_KEYS {
            let err = refuse_unpermitted(env_of(&[key]))
                .expect_err(&format!("{key} moves the numerics and must be refused"));
            assert!(
                err.contains(key),
                "refusal for {key} does not name it:\n{err}"
            );
        }
    }

    /// Every one of those is a key the engine actually reads at the pin, so the
    /// list above is a claim about the engine and not a list of names someone
    /// found alarming.
    #[test]
    fn the_numeric_overrides_are_engine_keys_at_the_pin() {
        let engine_owned = engine_owned();
        for key in REFUSED_NUMERIC_KEYS {
            assert!(
                engine_owned.iter().any(|name| name_eq(name, key)),
                "{key} is asserted to move the engine's numerics but does not appear in the \
                 engine key fixture at the pin"
            );
        }
    }

    /// The engine assembles this family's names from a runtime role string, so
    /// no list of forbidden names could ever cover it. Under deny-by-default it
    /// is closed by construction: there is no rule for these names, which is
    /// exactly why they are refused.
    #[test]
    fn the_computed_key_family_is_refused_without_a_rule_for_it() {
        for key in [
            "CAMELID_RECTANGULAR_LINEAR_LAYOUT_FFN_DOWN",
            "CAMELID_RECTANGULAR_LINEAR_LAYOUT_FFN_GATE",
            "CAMELID_RECTANGULAR_LINEAR_LAYOUT_ATTENTION_Q",
            // A role that exists nowhere in the engine at the pin. Refused all
            // the same: nothing enumerated it, and nothing had to.
            "CAMELID_RECTANGULAR_LINEAR_LAYOUT_A_ROLE_INVENTED_BY_A_LATER_PIN",
        ] {
            let err = refuse_unpermitted(env_of(&[key]))
                .expect_err("the computed family is refused");
            assert!(err.contains(key), "refusal does not name {key}:\n{err}");
        }
    }

    /// The lane is the sole writer of its own vector. A pre-set canonical key is
    /// refused even at the canonical value: sole-writership can be checked at
    /// startup, whereas continued agreement with an operator cannot.
    #[test]
    fn canonical_keys_are_refused_even_at_their_canonical_value() {
        for (key, value) in CANONICAL {
            let err = refuse_unpermitted(vec![(OsString::from(*key), OsString::from(*value))])
                .expect_err("the lane sets its own vector; it does not accept one");
            assert!(err.contains(key), "refusal does not name {key}:\n{err}");
            assert!(
                err.contains("the lane applies itself"),
                "the refusal for {key} does not explain who writes it, which is the answer a \
                 careful operator who set it in good faith needs:\n{err}"
            );
        }
    }

    /// Presence, not value: several engine overrides fire on the variable merely
    /// existing, and an empty string is the case a read-and-compare scan is
    /// likeliest to wave through.
    #[test]
    fn an_empty_value_is_still_a_set_variable() {
        let err = refuse_unpermitted(vec![(
            OsString::from("CAMELID_ROPE_PAIRING"),
            OsString::from(""),
        )])
        .expect_err("an empty value still changes the numerics");
        assert!(err.contains("CAMELID_ROPE_PAIRING"));

        // And an empty value on a permitted name is still admitted: the rule is
        // about the name in both directions.
        refuse_unpermitted(vec![(
            OsString::from("CAMELID_ENTERPRISE_ADDR"),
            OsString::from(""),
        )])
        .expect("a permitted name is admitted whatever it holds");
    }

    /// One refusal, every offender: five stray variables must not cost five
    /// restarts to discover.
    #[test]
    fn the_refusal_names_every_offender_sorted() {
        let err = refuse_unpermitted(env_of(&[
            "CAMELID_ROPE_PAIRING",
            "CAMELID_DETERMINISTIC",
            "PATH",
            "CAMELID_PROFILE",
        ]))
        .expect_err("three offenders");
        for key in [
            "CAMELID_DETERMINISTIC",
            "CAMELID_PROFILE",
            "CAMELID_ROPE_PAIRING",
        ] {
            assert!(err.contains(key), "refusal does not name {key}:\n{err}");
        }
        let order = |key: &str| err.find(key).expect("named");
        assert!(
            order("CAMELID_DETERMINISTIC") < order("CAMELID_PROFILE")
                && order("CAMELID_PROFILE") < order("CAMELID_ROPE_PAIRING"),
            "offenders should be listed sorted so the diagnostic diffs cleanly:\n{err}"
        );
        assert!(
            !err.contains("PATH"),
            "the scan claims one namespace:\n{err}"
        );
    }

    /// Values may hold anything and refusal text lands in journals that ship
    /// off-box, so the message reports names only.
    #[test]
    fn the_refusal_reports_names_without_values() {
        let err = refuse_unpermitted(vec![(
            OsString::from("CAMELID_SPEC_DRAFT_MODEL"),
            OsString::from("/secrets/tenant-a-draft.gguf"),
        )])
        .expect_err("refused");
        assert!(err.contains("CAMELID_SPEC_DRAFT_MODEL"));
        assert!(
            !err.contains("/secrets/tenant-a-draft.gguf"),
            "the refusal echoed a variable's value:\n{err}"
        );
    }

    /// The refusal is the answer to "what may I set, then?", so it carries the
    /// allow list with it.
    #[test]
    fn the_refusal_echoes_the_allow_list_with_reasons() {
        let err = refuse_unpermitted(env_of(&["CAMELID_PROFILE"])).expect_err("refused");
        for permit in PERMITTED {
            let name = permit.rule.literal();
            assert!(err.contains(name), "refusal omits permitted {name}:\n{err}");
            assert!(
                err.contains(permit.reason),
                "refusal omits the reason for {name}:\n{err}"
            );
        }
    }

    /// Keys with an advisory row explain themselves; keys without one print
    /// bare, which is correct — the table annotates a few dozen of the engine's
    /// hundreds and must stay comfortable being incomplete.
    ///
    /// Asserted against the offenders block alone. The allow-list table beneath
    /// it names `CAMELID_ENTERPRISE_THREADS` on every refusal whatsoever, so a
    /// whole-message `contains` for that string passes even if the advisory row
    /// for `CAMELID_THREADS` is deleted — a test that cannot fail for the reason
    /// it names.
    #[test]
    fn advisory_rows_annotate_the_refusal() {
        let err = refuse_unpermitted(env_of(&["CAMELID_THREADS"])).expect_err("refused");
        let offenders = offenders_of(&err);
        assert!(
            offenders.contains("CAMELID_THREADS — "),
            "the offender line should carry its advisory reason:\n{err}"
        );
        assert!(
            offenders.contains("use CAMELID_ENTERPRISE_THREADS"),
            "refusing CAMELID_THREADS should name its replacement where the operator is \
             reading, not only in the table below:\n{err}"
        );

        // …and a key with no row prints as a bare name. Checked on the line
        // itself, because "no explanation" is invisible to a `contains`.
        let bare = refuse_unpermitted(env_of(&["CAMELID_SOMETHING_UNANNOTATED"]))
            .expect_err("refused");
        let line = offenders_of(&bare)
            .lines()
            .find(|line| line.contains("CAMELID_SOMETHING_UNANNOTATED"))
            .expect("the offender is named");
        assert_eq!(line, "  CAMELID_SOMETHING_UNANNOTATED", "unexpected annotation: {line}");
    }

    /// An advisory row for a permitted key would print a warning next to a name
    /// the replica is telling the operator to use.
    #[test]
    fn no_advisory_row_annotates_a_permitted_key() {
        for (key, _) in ADVISORY {
            assert!(!permitted(key), "{key} is both permitted and warned about");
        }
    }

    /// Canonical keys are explained from `CANONICAL` itself, so a row here would
    /// be a second, silently divergent account of the lane's own vector.
    #[test]
    fn no_advisory_row_restates_a_canonical_key() {
        for (key, _) in ADVISORY {
            assert!(
                !CANONICAL.iter().any(|(canonical, _)| canonical == key),
                "{key} is in the frozen vector; its explanation comes from CANONICAL"
            );
        }
    }

    /// A row that annotates a name the scan admits never prints, and a reader of
    /// this table would have no way to tell. Every row must describe something
    /// that actually gets refused.
    #[test]
    fn every_advisory_row_annotates_a_name_the_scan_refuses() {
        for (key, reason) in ADVISORY {
            assert!(
                refused(key),
                "{key} has an advisory row but the scan admits it, so the row is unreachable"
            );
            let err = refuse_unpermitted(env_of(&[key])).expect_err("refused");
            assert!(
                offenders_of(&err).contains(reason),
                "the refusal for {key} does not carry its advisory reason beside the offender:\n{err}"
            );
        }
    }

    /// The Service names this distribution deploys under share the scanned
    /// prefix, so an orchestrator's injected link variables land in the scan.
    /// They are refused — but the refusal has to say why, because the first
    /// rollout succeeds and every one after it does not.
    #[test]
    fn service_link_variables_are_refused_with_the_fix() {
        // Both Services this repository ships, so the count an operator sees is
        // sixteen and not eight.
        let injected: Vec<String> = ["DETERMINISTIC", "GATEWAY"]
            .iter()
            .flat_map(|service| {
                [
                    "SERVICE_HOST",
                    "SERVICE_PORT",
                    "SERVICE_PORT_HTTP",
                    "PORT",
                    "PORT_80_TCP",
                    "PORT_80_TCP_PROTO",
                    "PORT_80_TCP_PORT",
                    "PORT_80_TCP_ADDR",
                ]
                .iter()
                .map(move |suffix| format!("CAMELID_ENTERPRISE_{service}_{suffix}"))
            })
            .collect();
        assert_eq!(injected.len(), 16);
        for name in &injected {
            assert!(
                looks_like_a_service_link(name),
                "{name} should be recognized as a service-link variable"
            );
        }
        let refs: Vec<&str> = injected.iter().map(String::as_str).collect();
        let err = refuse_unpermitted(env_of(&refs)).expect_err("refused");
        assert!(
            err.contains("enableServiceLinks: false"),
            "the refusal should name the fix:\n{err}"
        );

        // The heuristic annotates; it must not reach past variables shaped like
        // link injection.
        for name in [
            "CAMELID_ENTERPRISE_MODEL",
            "CAMELID_ROPE_PAIRING",
            "CAMELID_PROFILE",
        ] {
            assert!(
                !looks_like_a_service_link(name),
                "{name} is not a service-link variable"
            );
        }
    }

    /// A false positive here is worse than a missing annotation: it would tell
    /// an operator to change a pod spec when what they actually set was an
    /// engine override. The engine's whole key set at the pin is the strongest
    /// available corpus for that, so check against all of it.
    #[test]
    fn no_engine_key_is_mistaken_for_a_service_link() {
        for key in engine_keys() {
            assert!(
                !looks_like_a_service_link(key.name),
                "{} is an engine key at the pin but would be annotated as an injected \
                 service-link variable, sending the operator to fix the wrong thing",
                key.name
            );
        }
    }

    /// Names outside the scanned namespace are not this scan's business, even
    /// when they are near misses. Case handling is platform-specific by design.
    #[test]
    fn the_scan_claims_exactly_one_namespace() {
        refuse_unpermitted(env_of(&[
            "CAMELIDROPE",
            "CAMEL_ID_ROPE_PAIRING",
            "XCAMELID_PROFILE",
        ]))
        .expect("outside the namespace");

        let mixed_case = refuse_unpermitted(env_of(&["Camelid_Rope_Pairing"]));
        if cfg!(windows) {
            assert!(
                mixed_case.is_err(),
                "Windows environment lookups are case-insensitive, so this is the same variable"
            );
        } else {
            assert!(
                mixed_case.is_ok(),
                "on unix this is a different variable, and not one the engine reads"
            );
        }
    }

    /// The namespace claim has documented exceptions, and each has to arrive
    /// with its reason attached — an operator refused over a variable that is
    /// not even `CAMELID_*` has nothing to reason from otherwise.
    #[test]
    fn a_foreign_lever_is_refused_and_explains_itself() {
        for (name, reason) in REFUSED_FOREIGN {
            let err = refuse_unpermitted(env_of(&[name]))
                .expect_err("a foreign lever moves the numerics and must be refused");
            let offenders = offenders_of(&err);
            assert!(offenders.contains(name), "refusal does not name {name}:\n{err}");
            assert!(
                offenders.contains(reason),
                "refusal for {name} does not carry its reason beside the offender:\n{err}"
            );
        }
    }

    /// The refusal's last section lists the foreign names, and it is the only
    /// place an operator learns that this replica claims anything outside its
    /// own namespace. Nothing else in this module reads that section, so without
    /// this it could be dropped, truncated or left listing a stale set and every
    /// other test would still pass.
    #[test]
    fn the_refusal_lists_every_foreign_refusal_after_the_allow_list() {
        // Triggered by an in-namespace offender, so the trailer is reached on a
        // refusal that has nothing to do with the foreign list — which is the
        // case an operator meets, and the case in which a missing trailer is
        // invisible.
        let err = refuse_unpermitted(env_of(&["CAMELID_ROPE_PAIRING"])).expect_err("refused");
        let tables = tables_of(&err);
        for (name, _) in REFUSED_FOREIGN {
            assert!(
                tables.contains(name),
                "the refusal's trailing section omits {name}, so an operator refused over it \
                 later has no way to learn the replica claims that name:\n{err}"
            );
        }
        assert!(
            tables.contains("Refused from outside that namespace as well"),
            "the trailer needs its heading, or the names read as more of the allow list:\n{err}"
        );
    }

    /// A foreign row that is inside the namespace is dead weight: the general
    /// rule already refuses it, and the row would sit in the admission digest's
    /// preimage implying a policy the scan does not have.
    #[test]
    fn no_foreign_row_is_inside_the_namespace() {
        for (name, _) in REFUSED_FOREIGN {
            assert!(
                !in_namespace(name),
                "{name} is inside {NAMESPACE}, so deny-by-default already refuses it"
            );
        }
    }

    /// Both spellings of the pool-width lever are refused, and both point at the
    /// one width control this binary documents.
    ///
    /// The second name is the whole test. `rayon-core` falls through to
    /// `RAYON_RS_NUM_CPUS` whenever `RAYON_NUM_THREADS` is unset or does not
    /// parse as a positive integer, so a scan that refused only the current
    /// spelling refused nothing: the deprecated name sizes the same pool, flips
    /// the same width-greater-than-one kernel gates, and never appears in
    /// anything the replica prints. A refusal a synonym walks around is a hint.
    #[test]
    fn both_spellings_of_the_pool_width_lever_are_refused() {
        for name in ["RAYON_NUM_THREADS", "RAYON_RS_NUM_CPUS"] {
            let err = refuse_unpermitted(env_of(&[name]))
                .expect_err("an undocumented second way to size the pool is refused");
            let offenders = offenders_of(&err);
            assert!(offenders.contains(name), "refusal does not name {name}:\n{err}");
            assert!(
                offenders.contains("or CAMELID_ENTERPRISE_THREADS"),
                "refusing {name} must name the width control this binary documents:\n{err}"
            );
        }

        // Together, because that is how a shell carries them: one message, both
        // named, one restart.
        let err = refuse_unpermitted(env_of(&["RAYON_NUM_THREADS", "RAYON_RS_NUM_CPUS"]))
            .expect_err("refused");
        assert!(err.contains("RAYON_NUM_THREADS") && err.contains("RAYON_RS_NUM_CPUS"));
    }

    // `apply_deterministic` itself is exercised end to end in
    // `crates/server/tests/lane_environment.rs`, and it is over there rather
    // than here on purpose. It writes the process environment, and this binary's
    // test process is full of readers: every engine `AppState` a `main.rs` test
    // constructs captures planner and queue variables, and every `tempdir()`
    // reads `TMPDIR`. libtest runs tests as threads of one process, and glibc's
    // `getenv` takes no lock while `setenv` reallocates and frees the `environ`
    // array — the race that made `std::env::set_var` unsafe in edition 2024. An
    // integration test is its own binary, so the writer runs alone. Everything
    // in this module drives `refuse_unpermitted` directly, which touches nothing
    // global; keep it that way.
}
