//! Windows engine backend.
//!
//! Everything Windows-specific lives in this crate. The macOS and Linux ports
//! land first; this crate currently provides capability detection only.

use engine_core::host::HostCapabilities;

pub fn probe() -> HostCapabilities {
    let mut simd: Vec<&'static str> = Vec::new();
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2") {
            simd.push("avx2");
        }
        if std::arch::is_x86_feature_detected!("fma") {
            simd.push("fma");
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
