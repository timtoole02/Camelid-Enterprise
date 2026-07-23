//! macOS engine backend.
//!
//! Everything Apple-Silicon-specific lives in this crate: runtime CPU feature
//! detection now; the NEON/i8mm kernel ports and (later) Metal dispatch as the
//! engine port proceeds. Platform separation is enforced at the crate
//! boundary — `engine-core` never inspects the host, and no other platform's
//! code is compiled into a macOS build.

use engine_core::host::HostCapabilities;

/// Detect this host's capabilities. The result participates in the replica's
/// declared identity (startup banner, serving receipts): kernel routing keys
/// on these features, so they are part of what a deterministic replica vouches
/// for.
pub fn probe() -> HostCapabilities {
    let mut simd: Vec<&'static str> = Vec::new();
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            simd.push("neon");
        }
        if std::arch::is_aarch64_feature_detected!("i8mm") {
            simd.push("i8mm");
        }
        if std::arch::is_aarch64_feature_detected!("dotprod") {
            simd.push("dotprod");
        }
    }
    simd.sort_unstable();
    HostCapabilities {
        os: "macos",
        arch: std::env::consts::ARCH,
        logical_cores: std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1),
        simd,
    }
}

#[cfg(all(test, target_os = "macos"))]
mod tests {
    #[test]
    fn probe_reports_this_host() {
        let caps = super::probe();
        assert_eq!(caps.os, "macos");
        assert!(caps.logical_cores >= 1);
        #[cfg(target_arch = "aarch64")]
        assert!(caps.simd.contains(&"neon"), "Apple Silicon always has NEON");
    }
}
