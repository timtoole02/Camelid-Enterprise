//! Windows engine backend.
//!
//! Everything Windows-specific lives in this crate. The macOS and Linux ports
//! land first; this crate currently provides capability detection only.
//!
//! The reported feature list is part of the replica's published identity — it
//! answers "which kernels could have run here?" — so under-reporting lets two
//! hosts that route differently publish the identical `simd=` string. The
//! vocabularies below are therefore not maintained by comparing this file to
//! engine-linux's by eye. Both crates assert their lists against one checked-in
//! table of the engine's detection sites, and `vocabulary_matches_the_engine`
//! at the bottom of this file is this crate's half of that. A shared constant
//! would be tighter still, but it would have to live somewhere both platform
//! crates can see, and neither may depend on the other without inverting the
//! boundary that keeps one platform's code out of another's build.

use engine_core::host::HostCapabilities;

/// Declare a reported feature set once, as both a name list and its detection.
///
/// Duplicated from engine-linux rather than shared, for the reason in the
/// module docs: there is no home a platform crate may take it from today. What
/// is genuinely shared is the fixture the vocabularies are checked against, so
/// the duplication is of mechanism, not of the claim.
///
/// The name list must exist on every architecture — the vocabulary test checks
/// the x86-64 list while running on an aarch64 runner and vice versa — while
/// the detection can only compile where its intrinsics exist. Emitting both
/// from one list is what stops them drifting, and drift here is silent in one
/// direction: a feature added to the detector and forgotten in the list simply
/// under-reports.
///
/// Feature names arrive as `tt`, not `literal`. `std`'s detection macros
/// inspect the raw token to resolve a feature name and reject a `literal`
/// fragment with "unknown target feature", so the looser fragment specifier is
/// load-bearing rather than incidental.
macro_rules! reported_features {
    (
        $(#[$attr:meta])* $vocabulary:ident, $detect:ident,
        $arch:literal, $detected:ident, [$($feature:tt),+ $(,)?]
    ) => {
        $(#[$attr])*
        pub const $vocabulary: &[&str] = &[$($feature),+];

        #[cfg(target_arch = $arch)]
        fn $detect(simd: &mut Vec<&'static str>) {
            $(
                if std::arch::$detected!($feature) {
                    simd.push($feature);
                }
            )+
        }
    };
}

reported_features!(
    /// x86-64 features this crate reports.
    ///
    /// Identical to engine-linux's x86-64 list, and not by convention: every
    /// name the engine routes a kernel on by default is `required` in the
    /// shared fixture, no x86-64 name is optional, so the two vocabularies are
    /// pinned to the same set from both ends. The reason each name is here —
    /// in particular why `avx512bw` sits beside `avx512f` and `avx512vnni` —
    /// is recorded in the fixture rather than restated on either probe.
    X86_64_REPORTED_FEATURES,
    detect_x86_64,
    "x86_64",
    is_x86_feature_detected,
    ["avx", "avx2", "avx512bw", "avx512f", "avx512vnni", "f16c", "fma"]
);

reported_features!(
    /// aarch64 features this crate reports.
    ///
    /// Declared identically to engine-linux, but what this platform can
    /// actually detect is narrower, and the difference is runtime rather than
    /// declarative. Windows resolves aarch64 features through
    /// `IsProcessorFeaturePresent`, a narrower flag set than Linux's HWCAP:
    /// `neon` (architecturally mandatory) and `dotprod` are exposed by the
    /// stable detection backend, while `i8mm` is not in this toolchain — the
    /// mapping exists upstream but has not reached stable. So the `i8mm` probe
    /// is a no-op here today. It stays declared so the feature lights up on its
    /// own once the flag ships, rather than this crate silently under-reporting
    /// at that moment; the shared fixture permits it either way, which is what
    /// lets one table describe two platforms whose detection backends differ.
    AARCH64_REPORTED_FEATURES,
    detect_aarch64,
    "aarch64",
    is_aarch64_feature_detected,
    ["dotprod", "i8mm", "neon"]
);

/// Detect this host's capabilities. The result participates in the replica's
/// declared identity (startup banner, response headers, serving receipts):
/// kernel routing keys on these features, so they are part of what a
/// deterministic replica vouches for.
pub fn probe() -> HostCapabilities {
    let mut simd: Vec<&'static str> = Vec::new();
    #[cfg(target_arch = "x86_64")]
    detect_x86_64(&mut simd);
    #[cfg(target_arch = "aarch64")]
    detect_aarch64(&mut simd);
    simd.sort_unstable();
    HostCapabilities {
        os: "windows",
        arch: std::env::consts::ARCH,
        logical_cores: std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1),
        simd,
    }
}

#[cfg(all(test, target_os = "windows"))]
mod tests {
    #[test]
    fn probe_reports_this_host() {
        let caps = super::probe();
        assert_eq!(caps.os, "windows");
        assert!(caps.logical_cores >= 1);
        #[cfg(target_arch = "aarch64")]
        assert!(caps.simd.contains(&"neon"), "aarch64 always has NEON");
    }
}

/// This crate's half of the shared vocabulary check.
///
/// Deliberately not gated on `target_os`. Nothing in it runs Windows code or
/// executes an instruction — it compares two `const`s against a table — so it
/// holds on the Linux and macOS runners, which are the only ones this project
/// has. Gating it on Windows would be a test that exists and never runs, which
/// is how the parity between this crate and engine-linux went unenforced while
/// a comment asserted it.
///
/// What it proves: this crate's declared vocabulary satisfies the same bounds
/// engine-linux's does, against the same file, so the two cannot diverge on any
/// architecture where the fixture leaves no room. What it does not prove: that
/// this crate's *detection* fires on a real Windows host. Windows's backend
/// exposes a narrower aarch64 set than it declares, which is stated on the
/// vocabulary above and is not observable from here.
#[cfg(test)]
mod vocabulary_matches_the_engine {
    use super::{AARCH64_REPORTED_FEATURES, X86_64_REPORTED_FEATURES};

    /// The same table engine-linux's coverage test reads. Reached by path
    /// because a data file cannot be shared through a dependency this crate is
    /// allowed to take; a copy would be the drift the file exists to prevent.
    const FIXTURE: &str = include_str!("../../engine-linux/tests/fixtures/engine-kernel-features.tsv");

    /// `(arch, feature, published)` for every fixture row. The routing column
    /// is engine-linux's to police — it owns the invariant tying routing to
    /// published, and re-asserting it here would be a second copy of a rule
    /// that must have exactly one.
    fn rows() -> Vec<(&'static str, &'static str, &'static str)> {
        FIXTURE
            .lines()
            .map(str::trim_end)
            .filter(|line| !line.is_empty() && !line.starts_with('#'))
            .map(|line| {
                let fields: Vec<&str> = line.split('\t').collect();
                assert!(fields.len() >= 4, "fixture row has too few columns: {line}");
                (fields[0], fields[1], fields[3])
            })
            .collect()
    }

    fn vocabularies() -> [(&'static str, &'static [&'static str]); 2] {
        [("x86_64", X86_64_REPORTED_FEATURES), ("aarch64", AARCH64_REPORTED_FEATURES)]
    }

    #[test]
    fn every_default_reachable_kernel_gate_is_published() {
        let rows = rows();
        for (arch, reported) in vocabularies() {
            for (_, feature, _) in
                rows.iter().filter(|(row_arch, _, published)| *row_arch == arch && *published == "required")
            {
                assert!(
                    reported.contains(feature),
                    "the pinned engine selects an inference kernel on {arch}/{feature} by \
                     default, and this crate does not report it. A Windows and a Linux host with \
                     the same CPU would publish different simd= strings, and two Windows hosts \
                     differing only in that feature would publish the same one while routing \
                     through different kernels."
                );
            }
        }
    }

    #[test]
    fn nothing_is_published_that_the_fixture_does_not_declare() {
        let rows = rows();
        for (arch, reported) in vocabularies() {
            for feature in reported {
                let row = rows
                    .iter()
                    .find(|(row_arch, row_feature, _)| *row_arch == arch && row_feature == feature);
                let Some((_, _, published)) = row else {
                    panic!(
                        "{arch} reports {feature}, which is not in the kernel-feature fixture at \
                         all. Add the row there — it is the one place both platform crates read — \
                         rather than to this vocabulary alone."
                    );
                };
                assert_ne!(
                    *published, "no",
                    "{arch} reports {feature}, which the fixture marks as one no probe should \
                     publish: the engine does not select a kernel on it."
                );
            }
        }
    }

    /// Vacuity guard. A path typo cannot happen — `include_str!` fails to
    /// compile — but a fixture rewritten into a shape this parser skips would
    /// leave both assertions above trivially true.
    #[test]
    fn the_shared_fixture_parsed() {
        let rows = rows();
        assert!(
            rows.len() >= 18,
            "parsed {} fixture rows; the sweep at the pin recorded 18, so this crate's checks are \
             weaker than they read",
            rows.len()
        );
        for (arch, _) in vocabularies() {
            assert!(
                rows.iter().any(|(row_arch, _, published)| *row_arch == arch && *published == "required"),
                "no required {arch} rows, so that vocabulary is effectively unchecked here"
            );
        }
    }
}
