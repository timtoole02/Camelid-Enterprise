//! Lane identity and the frozen configuration vector.
//!
//! A deterministic-lane replica's output guarantee is stated against the engine
//! at a pinned revision *under a frozen configuration vector*. That vector is
//! enforced here at startup: canonical values are applied to the process
//! environment, and any operator override of a pinned key is a startup error —
//! the lane fails closed rather than serving under a config it cannot vouch for.

use sha2::{Digest, Sha256};

pub const ENGINE_PIN: &str = "b4e3a9056567ed8145fc4fa29850d6f1f261ac2b";

/// Keys the deterministic lane sets to canonical values before the engine
/// reads any of them. The engine pins its forward pass to the order-stable
/// CPU path when `CAMELID_DETERMINISTIC` is on; the explicit `false` entries
/// mirror the engine CLI's own deterministic-mode behavior so the vector is
/// complete even if a key's default changes upstream.
const CANONICAL: &[(&str, &str)] = &[
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

/// Keys that must NOT be overridden on the deterministic lane. Each moves a
/// numeric route boundary, changes batch shape, or enables a feature the lane
/// excludes; the guarantee holds only at the engine defaults.
const MUST_BE_UNSET: &[&str] = &[
    "CAMELID_PREFILL_CHUNK_TOKENS",
    "CAMELID_PREFILL_LAYER_MAJOR_CHUNK_TOKENS",
    "CAMELID_MAC_Q8_PREFILL_I8MM",
    "CAMELID_MAC_Q8_I8MM_SMALL_M_MAX_ROWS",
    "CAMELID_X86_Q8_PARALLEL_INPUT_QUANTIZE",
    "CAMELID_ATTN_COALESCED",
    "CAMELID_SPEC_DECODE",
    "CAMELID_SPEC_DRAFT_MODEL",
    "CAMELID_SPEC_DRAFT_TOKENS",
    "CAMELID_QUEUE_DEPTH",
];

/// The applied, frozen configuration vector.
pub struct ConfigVector {
    /// SHA-256 over the canonical `KEY=VALUE` list **in declaration order**,
    /// plus `engine_pin=<rev>`, identifying this exact vector in attribution
    /// headers and serving receipts. Declaration order is load-bearing:
    /// reordering `CANONICAL` changes the digest (see `compute_config_sha256`).
    pub sha256: String,
}

impl ConfigVector {
    pub fn short(&self) -> &str {
        &self.sha256[..12]
    }
}

/// Apply the deterministic lane's configuration vector, failing closed on any
/// operator override of a pinned or excluded key.
pub fn apply_deterministic() -> Result<ConfigVector, String> {
    for key in MUST_BE_UNSET {
        if let Ok(value) = std::env::var(key) {
            return Err(format!(
                "deterministic lane refuses to start: {key}={value} overrides a pinned \
                 configuration key. Unset it — the lane's output guarantee holds only at \
                 the engine defaults."
            ));
        }
    }
    for (key, value) in CANONICAL {
        if let Ok(existing) = std::env::var(key) {
            if existing != *value {
                return Err(format!(
                    "deterministic lane refuses to start: {key}={existing} conflicts with \
                     the canonical value {value}. Unset it."
                ));
            }
        }
        std::env::set_var(key, value);
    }
    Ok(ConfigVector {
        sha256: compute_config_sha256(),
    })
}

/// Compute the configuration-vector digest: SHA-256 over each `KEY=VALUE\n` of
/// `CANONICAL` **in declaration order**, then `engine_pin=<ENGINE_PIN>`.
/// Declaration order is part of the contract — reordering `CANONICAL` silently
/// changes the published hash. Kept free of environment access so the digest can
/// be pinned by a test without the fail-closed startup side effects.
fn compute_config_sha256() -> String {
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

#[cfg(test)]
mod tests {
    use super::*;

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
}
