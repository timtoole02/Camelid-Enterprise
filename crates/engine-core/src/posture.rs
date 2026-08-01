//! The numeric contract a kernel holds to, declared by the crate that supplies
//! it and published on the replica's identity surface.
//!
//! Acceleration is not always bit-identical, and this vocabulary exists so the
//! replica can say which it is serving under rather than leave a client to
//! assume. Every Q8_0 dot supplied through [`crate::tensor::Q8DotRows`] today —
//! the portable reference, engine-macos's NEON, engine-windows's AVX2 — declares
//! [`NumericPosture::BitIdentical`], so the published value is the same
//! everywhere. A posture is a claim about the kernel that crosses a seam, not
//! about everything the crate supplying it exports: engine-macos also has a NEON
//! quantizer that documents an inert divergence from the portable one, and no
//! seam takes it, so it declares nothing here. A GPU lane admitted under a
//! published tolerance would be the first path that is not, and the field is
//! shipped now, at the value it truthfully has, so that arrival is a value
//! change on an existing field rather than a new surface bolted on at the
//! moment the claim weakens.
//!
//! This is a sibling of [`crate::host`] and not part of it, because the two
//! answer different questions. [`crate::host::HostCapabilities`] answers "which
//! kernels *could* have run here", which is a fact about hardware. A posture
//! answers "what numeric contract does the kernel that *did* run hold to",
//! which is a claim about code. A replica can move between hosts without
//! changing its posture, and can change its posture without moving host.
//!
//! It is also not part of [`crate::tensor`]. The Q8_0 dot is the only seam a
//! platform crate accelerates today, but the per-path obligation below refers to
//! paths that are not tensor concepts at all, and `engine_core::tensor::NumericPosture`
//! would read as a property of the tensor layer rather than of the replica.

/// The numeric contract a kernel holds to, relative to the portable reference.
///
/// # Posture is per-PATH, not per-replica
///
/// This is the property most likely to be weakened by accident, so it is stated
/// before anything else. A replica can legitimately be tolerance-conformant on
/// prefill and bit-identical on decode at the same time: those are two execution
/// paths, and each holds to its own contract.
///
/// Today one seam ([`crate::tensor::Q8DotRows`]) serves every projection in both
/// prefill and decode, so a single replica-wide value is a *true statement*
/// rather than a simplification — there is exactly one path and it has exactly
/// one posture.
///
/// The first execution path with a posture of its own **must** make the
/// published field per-path. It must not collapse the replica-wide claim to its
/// weakest member. Collapsing publishes `tolerance-conformant` for a decode that
/// is bit-identical, which destroys the only thing this field answers: a client
/// comparing two replicas would see the field move for a path it never used, and
/// a client auditing the decode path would be told a weaker thing than the truth
/// and have no way to recover the stronger one. The machinery for that is
/// deliberately not built here; this paragraph is the only control on it, which
/// is why it names the failure rather than merely instructing.
///
/// # The published strings are wire values
///
/// [`NumericPosture::published`] returns the exact token that appears in the
/// `x-camelid-posture` header, the `camelid_posture` completion-body field and
/// the serving receipt's `posture`. Renaming one retires a published identity
/// the same way changing a digest preimage does, and it is not covered by any
/// digest, so nothing else in the tree would catch it. It is pinned by a test
/// beside this type and by one beside every kernel that declares a posture.
///
/// # What is deliberately not derived
///
/// `Eq` and `Hash` are not derived. [`Tolerance`] carries `f32` values, for
/// which neither is well defined, and adding a derive later is additive while
/// removing one is the breaking change this shape exists to avoid.
///
/// `Copy` is derived because the server's per-request identity record is cloned
/// by the axum layer on every request, and this is a compile-time constant
/// chosen by target rather than anything a request owns.
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub enum NumericPosture {
    /// The kernel reproduces the portable reference bit for bit, on every input,
    /// with no exception. `to_bits()` equality, not `==`: `-0.0 == 0.0` is true
    /// and a NaN compares unequal to itself, so `==` is a weaker claim than this
    /// variant makes.
    ///
    /// This is what every kernel in this workspace declares. It is the strongest
    /// posture there is, and it is achievable for a CPU Q8_0 kernel because the
    /// integer block dot is exact and the f32 reduction can be reproduced in
    /// order — see `engine_core::tensor::q8_0_dot_rows`, which is the definition
    /// the claim is stated against rather than a second thing measured against a
    /// first.
    BitIdentical,
    /// The kernel does not reproduce the reference bit for bit, and is admitted
    /// under a published numeric tolerance instead.
    ///
    /// **No kernel in this tree declares this.** It exists now, ahead of the
    /// first one, for a reason that is not anticipation: with a single-variant
    /// enum no test in the workspace could construct a second posture, and the
    /// published field would ship with its power unmeasured — a field that
    /// distinguishes nothing distinguishes nothing whether or not it is on the
    /// wire.
    ///
    /// What a kernel must show to declare it, so the variant is not a place to
    /// hide: a measured maximum relative error against the CPU reference, and a
    /// measured top-1 agreement floor over a *fixed* prompt set, both published.
    /// Those numbers are not on the wire today — the header carries the bare
    /// token `tolerance-conformant` — so the first kernel to declare this owes
    /// the surface that publishes them as well. A token with no numbers behind it
    /// is a claim without a contract, which is worse than no claim.
    Conformant(Tolerance),
}

/// The numeric contract a non-bit-identical kernel is admitted under.
///
/// Constructed through [`Tolerance::new`] rather than by literal, because the
/// type is `#[non_exhaustive]`: a third term (a per-token divergence bound, say)
/// has to be addable without breaking every declaration site.
///
/// None of these fields reaches the wire yet; see [`NumericPosture::Conformant`].
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub struct Tolerance {
    /// Largest relative error observed against the portable CPU reference over
    /// the prompt set, as a fraction (`1e-3`, not `0.1%`).
    pub max_relative_error: f32,
    /// Fraction of positions at which the accelerated path's argmax agreed with
    /// the reference's, over the prompt set. A floor, not an average: it is the
    /// number a client is entitled to hold the replica to.
    pub top1_agreement: f32,
    /// The fixed prompt set the two numbers were measured over. Named rather
    /// than described, because a tolerance measured over a set nobody can
    /// reproduce is not a contract.
    pub prompt_set: &'static str,
}

impl Tolerance {
    pub const fn new(
        max_relative_error: f32,
        top1_agreement: f32,
        prompt_set: &'static str,
    ) -> Self {
        Self {
            max_relative_error,
            top1_agreement,
            prompt_set,
        }
    }
}

impl NumericPosture {
    /// The published token for this posture.
    ///
    /// `&'static str` rather than `String` so that stamping a response header is
    /// infallible: there is no conversion here that could fail and be silently
    /// swallowed on the one surface a streaming client can read.
    pub const fn published(self) -> &'static str {
        match self {
            Self::BitIdentical => "bit-identical",
            Self::Conformant(_) => "tolerance-conformant",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The tokens are wire values with the same retirement cost as a digest, and
    /// nothing else in this workspace would catch a rename — they are in no
    /// digest preimage, which is the whole point of the field.
    #[test]
    fn the_published_tokens_are_pinned() {
        assert_eq!(NumericPosture::BitIdentical.published(), "bit-identical");
        assert_eq!(
            NumericPosture::Conformant(Tolerance::new(1e-3, 0.999, "fixture")).published(),
            "tolerance-conformant"
        );
    }

    /// Two postures must not render identically, or the field distinguishes
    /// nothing on any of the three surfaces it is published on.
    #[test]
    fn distinct_postures_render_distinctly() {
        assert_ne!(
            NumericPosture::BitIdentical.published(),
            NumericPosture::Conformant(Tolerance::new(0.0, 1.0, "fixture")).published(),
        );
    }

    /// The tolerance terms survive construction. They are not on the wire yet,
    /// so this is the only thing standing between a declared contract and a
    /// constructor that quietly drops one of its arguments.
    #[test]
    fn a_tolerance_carries_the_numbers_it_was_declared_with() {
        let tolerance = Tolerance::new(1.5e-3, 0.997, "eval/greedy-512");
        assert_eq!(tolerance.max_relative_error, 1.5e-3);
        assert_eq!(tolerance.top1_agreement, 0.997);
        assert_eq!(tolerance.prompt_set, "eval/greedy-512");
        assert_eq!(
            NumericPosture::Conformant(tolerance),
            NumericPosture::Conformant(Tolerance::new(1.5e-3, 0.997, "eval/greedy-512")),
            "two declarations of the same contract are the same posture"
        );
    }
}
