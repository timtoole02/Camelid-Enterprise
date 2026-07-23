//! Linux engine backend.
//!
//! Everything Linux-specific lives in this crate: runtime CPU feature
//! detection now; the x86-64 AVX2/AVX-512/VNNI kernel ports and (later) CUDA
//! dispatch as the engine port proceeds. The macOS port is landing first;
//! this crate currently provides capability detection only.

use engine_core::host::HostCapabilities;

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
        if std::arch::is_aarch64_feature_detected!("neon") {
            simd.push("neon");
        }
        if std::arch::is_aarch64_feature_detected!("i8mm") {
            simd.push("i8mm");
        }
    }
    simd.sort_unstable();
    HostCapabilities {
        os: "linux",
        arch: std::env::consts::ARCH,
        logical_cores: std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1),
        simd,
    }
}
