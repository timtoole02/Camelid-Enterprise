//! Host capability description, filled in by the per-platform crates.
//!
//! The deterministic lane's guarantee is scoped to a hardware class, so the
//! capabilities a replica detects are part of its declared identity and are
//! surfaced in the startup banner and serving receipts.

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HostCapabilities {
    /// Operating system this build serves on (e.g. "macos").
    pub os: &'static str,
    /// CPU architecture (e.g. "aarch64").
    pub arch: &'static str,
    /// Logical cores available to the process.
    pub logical_cores: usize,
    /// SIMD / matrix instruction-set extensions detected at runtime, sorted.
    /// These change kernel routing, which is why they belong to the replica's
    /// identity rather than being an invisible implementation detail.
    pub simd: Vec<&'static str>,
}

impl HostCapabilities {
    /// One-line form for banners and logs, stable field order.
    pub fn summary(&self) -> String {
        format!(
            "{}/{} cores={} simd={}",
            self.os,
            self.arch,
            self.logical_cores,
            if self.simd.is_empty() { "none".to_string() } else { self.simd.join("+") }
        )
    }
}
