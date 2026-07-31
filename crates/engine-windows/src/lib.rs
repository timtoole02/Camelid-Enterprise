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
    /// at that moment.
    ///
    /// The fixture now *requires* `i8mm` on aarch64 rather than permitting it:
    /// the engine's one gate on that feature also names an environment key, but
    /// the key is one the engine's own execution planner writes at model load,
    /// so the branch is default-reachable and the name is part of the routing
    /// identity. That gate is compiled into the macOS aarch64 build only, so it
    /// is not a branch a Windows host takes — which is why a detection backend
    /// that cannot see `i8mm` does not make this platform's replicas ambiguous
    /// about a kernel they could have run. It does mean this crate declares a
    /// required name it cannot yet report, and that is a runtime shortfall
    /// against a declarative table, invisible from any runner this project has.
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

/// Exercises `probe` itself, as opposed to the vocabulary it draws from.
///
/// Not gated on `target_os`. It was, and for as long as it was it ran nowhere:
/// this project had no Windows runner, so the only test that touched the
/// Windows probe was compiled by no job — the anti-pattern the module below
/// names as the reason it is ungated. There is a Windows job now, and it runs
/// this. The gate stays off anyway, because nothing in `probe` is Windows-only
/// code — the `os` field is a constant and the detection arms are keyed by
/// `target_arch` — so the check costs nothing on the other runners, and a
/// crate's tests should not become unrun by the deletion of one CI job.
///
/// What running it off Windows proves: `probe` returns the shape the identity
/// is rendered from, and the architecture-keyed detection arm compiles and
/// resolves against whatever backend that host has. What it does not prove, and
/// what only the Windows job settles: that `IsProcessorFeaturePresent` reports
/// what the vocabulary above claims. The assertions are written so that they
/// are true statements on either host rather than true-by-vacuity on one.
#[cfg(test)]
mod tests {
    #[test]
    fn probe_reports_this_host() {
        let caps = super::probe();
        assert_eq!(
            caps.os, "windows",
            "this crate's probe speaks for Windows on whatever host it is built on"
        );
        assert!(caps.logical_cores >= 1);
        assert_eq!(caps.arch, std::env::consts::ARCH);
        #[cfg(target_arch = "aarch64")]
        assert!(caps.simd.contains(&"neon"), "aarch64 always has NEON");
    }

    /// Sorted and free of duplicates: `simd` is compared for string equality
    /// between replicas, so two hosts with the same features must render it
    /// identically. `probe` sorts but does not deduplicate, which is why the
    /// vocabularies themselves are checked for uniqueness in the shared
    /// vocabulary module rather than relying on this one — it can only see the
    /// running host. engine-linux has had this check since it landed; this
    /// crate did not, because its test module was gated onto a runner that does
    /// not exist.
    #[test]
    fn the_reported_set_is_sorted_and_unique() {
        let simd = super::probe().simd;
        let mut expected = simd.clone();
        expected.sort_unstable();
        expected.dedup();
        assert_eq!(simd, expected);
    }
}

/// This crate's half of the shared vocabulary check.
///
/// Deliberately not gated on `target_os`. Nothing in it runs Windows code or
/// executes an instruction — it compares two `const`s against a table — so it
/// holds on every runner rather than only the one that can run Windows code.
/// Gating it on Windows would narrow a check that costs nothing to widen, which
/// is how the parity between this crate and engine-linux went unenforced while
/// a comment asserted it.
///
/// What it proves: this crate's declared vocabulary satisfies the same bounds
/// engine-linux's does, against the same file, so the two cannot diverge on any
/// architecture where the fixture leaves no room. What it does not prove: that
/// this crate's *detection* fires on a real Windows host. That is a question for
/// the Windows job, and even there only for the architecture that job runs on —
/// the narrower aarch64 flag set stated on the vocabulary above is not settled
/// by an x86-64 Windows runner any more than by a Linux one.
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
