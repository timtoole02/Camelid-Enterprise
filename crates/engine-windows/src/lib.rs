//! Windows engine backend.
//!
//! Everything Windows-specific lives in this crate. The macOS and Linux ports
//! land first; this crate currently provides capability detection only.

use engine_core::host::HostCapabilities;

/// Detect this host's capabilities. The result participates in the replica's
/// declared identity (startup banner, serving receipts): kernel routing keys
/// on these features, so they are part of what a deterministic replica vouches
/// for.
///
/// Detection is OS-specific: each backend can only report what its platform's
/// detection API exposes. On x86_64 the reported set matches engine-linux. On
/// aarch64 it does not, because Windows and Linux expose different feature sets
/// (see the aarch64 branch below).
pub fn probe() -> HostCapabilities {
    let mut simd: Vec<&'static str> = Vec::new();
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2") {
            simd.push("avx2");
        }
        if std::arch::is_x86_feature_detected!("avx512f") {
            simd.push("avx512f");
        }
        if std::arch::is_x86_feature_detected!("avx512vnni") {
            simd.push("avx512vnni");
        }
        if std::arch::is_x86_feature_detected!("fma") {
            simd.push("fma");
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        // Windows detects aarch64 features via `IsProcessorFeaturePresent`, a
        // narrower flag set than Linux's HWCAP, so this branch does NOT match
        // engine-linux one-for-one. `neon` (architecturally mandatory) and
        // `dotprod` are exposed by this stable `std_detect` backend and reported.
        // `i8mm` is not exposed by the stable Windows backend in this toolchain
        // (the mapping exists upstream in std_detect but has not shipped in
        // stable), so the check below is a no-op on current stable — unlike
        // engine-linux, Windows cannot report i8mm today. It is kept so the
        // feature lights up automatically once the flag reaches stable, rather
        // than silently under-reporting then.
        if std::arch::is_aarch64_feature_detected!("neon") {
            simd.push("neon");
        }
        if std::arch::is_aarch64_feature_detected!("dotprod") {
            simd.push("dotprod");
        }
        if std::arch::is_aarch64_feature_detected!("i8mm") {
            simd.push("i8mm");
        }
    }
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
