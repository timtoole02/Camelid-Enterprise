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
    /// SHA-256 over the canonical `KEY=VALUE` list (sorted), identifying this
    /// exact vector in attribution headers and serving receipts.
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
    let mut hasher = Sha256::new();
    for (key, value) in CANONICAL {
        hasher.update(key.as_bytes());
        hasher.update(b"=");
        hasher.update(value.as_bytes());
        hasher.update(b"\n");
    }
    hasher.update(b"engine_pin=");
    hasher.update(ENGINE_PIN.as_bytes());
    Ok(ConfigVector {
        sha256: format!("{:x}", hasher.finalize()),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `apply_deterministic` reads and mutates process-global environment, so
    /// these phases run inside one serialized test rather than as separate
    /// `#[test]`s that the harness could interleave on parallel threads.
    #[test]
    fn deterministic_lane_contract() {
        // Control the environment fully: clear every key this function reasons
        // about so the outcome does not depend on the ambient environment.
        for key in MUST_BE_UNSET {
            std::env::remove_var(key);
        }
        for (key, _) in CANONICAL {
            std::env::remove_var(key);
        }

        // Phase 1 — the frozen vector is exactly the documented value. This is
        // the contract: any change to CANONICAL or ENGINE_PIN must break it.
        // (Short form matches the README banner / attribution: `30d77c260803`.)
        let config = apply_deterministic().expect("a clean environment must apply");
        assert_eq!(
            config.sha256,
            "30d77c2608036f8475372ace9ec125ffc5fa16d8d63f0355a08c32c69f4449b7",
        );
        assert_eq!(config.short(), "30d77c260803");

        // The vector is a property of CANONICAL + ENGINE_PIN, not of the
        // environment: re-applying (now that the canonical keys are set) is
        // identical.
        let reapplied = apply_deterministic().expect("re-apply must succeed");
        assert_eq!(reapplied.sha256, config.sha256);

        // Phase 2 — a MUST_BE_UNSET key present fails closed, naming the key.
        std::env::set_var("CAMELID_QUEUE_DEPTH", "8");
        let err = match apply_deterministic() {
            Ok(_) => panic!("an excluded key must fail closed"),
            Err(e) => e,
        };
        assert!(
            err.contains("CAMELID_QUEUE_DEPTH"),
            "the error must name the offending key: {err}",
        );
        std::env::remove_var("CAMELID_QUEUE_DEPTH");

        // Phase 3 — a canonical key overridden to a conflicting value fails
        // closed; the same key at its canonical value is accepted.
        std::env::set_var("CAMELID_DETERMINISTIC", "0");
        let err = match apply_deterministic() {
            Ok(_) => panic!("a conflicting override must fail closed"),
            Err(e) => e,
        };
        assert!(
            err.contains("CAMELID_DETERMINISTIC"),
            "the error must name the offending key: {err}",
        );
        std::env::set_var("CAMELID_DETERMINISTIC", "1");
        apply_deterministic().expect("a canonical key at its canonical value must be accepted");
    }
}
