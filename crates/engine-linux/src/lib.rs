//! Linux engine backend.
//!
//! Everything Linux-specific lives in this crate: runtime CPU feature
//! detection now; the x86-64 AVX2/AVX-512/VNNI kernel ports and (later) CUDA
//! dispatch as the engine port proceeds. The macOS port is landing first;
//! this crate currently provides capability detection only.
//!
//! The feature list this crate reports is not a description of the CPU — it is
//! the part of the replica's published identity that answers "which kernels
//! could have run here?". A feature the engine routes on and this crate does
//! not report is two hosts publishing the identical `simd=` string while
//! executing different code, which is an identity that is wrong in exactly the
//! way an identity must never be wrong. So the vocabulary below is pinned
//! against the engine's own detection sites by `tests/kernel_feature_coverage`
//! and its fixture: the thing that fails at a pin bump instead of a client
//! discovering it.
//!
//! Note what the claim deliberately is not. It is a *routing* identity, not an
//! assertion that every distinction moves arithmetic. The engine documents
//! bitwise equality between some of these arms and says nothing about others,
//! and re-deriving which is which at every pin is the work this distribution
//! exists to avoid. Publishing the route is cheap and checkable; proving the
//! route is numerically inert is neither.

use engine_core::host::HostCapabilities;

/// Declare a reported feature set once, as both a name list and its detection.
///
/// The name list has to exist on every architecture — the coverage test checks
/// the x86-64 vocabulary while running on an aarch64 runner and vice versa —
/// while the detection can only compile on the architecture whose intrinsics it
/// names. Emitting both from one list is what makes them unable to drift, and
/// drift here has a silent direction: a feature added to the detector and
/// forgotten in the list under-reports, which is the whole defect this guards.
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
    /// Every name here is one the pinned engine calls runtime detection on and
    /// selects a different kernel by, with no environment key in the predicate.
    /// `avx512bw` earns its place beside `avx512f` and `avx512vnni` because the
    /// engine's VNNI decode requires all three: reporting two of the three
    /// describes a routing decision incompletely, which is the same defect as
    /// not reporting it at all, and it is the one this vocabulary was widened
    /// to fix. `avx` and `f16c` are here for the same reason on the KV-cache
    /// f16 conversion path.
    X86_64_REPORTED_FEATURES,
    detect_x86_64,
    "x86_64",
    is_x86_feature_detected,
    ["avx", "avx2", "avx512bw", "avx512f", "avx512vnni", "f16c", "fma"]
);

reported_features!(
    /// aarch64 features this crate reports.
    ///
    /// `dotprod` is a live kernel gate. `neon` and `i8mm` are not — the engine
    /// detects `neon` only to report it, and its one `i8mm` kernel gate also
    /// requires an environment key this distribution's startup scan refuses.
    /// Both stay because dropping them would retire a `simd=` string replicas
    /// publish today, for no gain; the fixture records why each is here so the
    /// distinction is written down rather than inferred from a list.
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
        os: "linux",
        arch: std::env::consts::ARCH,
        logical_cores: std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1),
        simd,
    }
}

#[cfg(all(test, target_os = "linux"))]
mod tests {
    #[test]
    fn probe_reports_this_host() {
        let caps = super::probe();
        assert_eq!(caps.os, "linux");
        assert!(caps.logical_cores >= 1);
        #[cfg(target_arch = "aarch64")]
        assert!(caps.simd.contains(&"neon"), "aarch64 always has NEON");
    }

    /// Sorted and free of duplicates: `simd` is compared for string equality
    /// between replicas, so two hosts with the same features must render it
    /// identically. `probe` sorts but does not deduplicate, which is why the
    /// vocabularies themselves are checked for uniqueness in the coverage test
    /// rather than relying on this one — it can only see the running host.
    #[test]
    fn the_reported_set_is_sorted_and_unique() {
        let simd = super::probe().simd;
        let mut expected = simd.clone();
        expected.sort_unstable();
        expected.dedup();
        assert_eq!(simd, expected);
    }
}
