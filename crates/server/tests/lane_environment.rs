//! The lane's startup scan, driven against the real process environment.
//!
//! This is its own test binary for one reason, and the reason is soundness
//! rather than tidiness. `apply_deterministic` writes the process environment,
//! and libtest runs a binary's tests as threads of a single process. The server
//! binary's test process is dense with environment *readers* — each engine
//! `AppState` a `main.rs` test builds captures the planner's managed keys and the
//! queue depth, and each `tempdir()` reads `TMPDIR` — so a writer running there
//! is an unsynchronized `setenv` against concurrent `getenv`. glibc's `getenv`
//! takes no lock while `setenv` reallocates and frees the `environ` array, which
//! is exactly the race that made `std::env::set_var` unsafe in edition 2024. It
//! reproduces on the Linux runner and not on macOS, whose libc serializes
//! environment access — the worst shape a race can have.
//!
//! An integration test is a separate binary, so the writer here runs alone: the
//! module it pulls in has no other test that reads or writes the environment,
//! and nothing in this file constructs an engine or a temporary directory. Keep
//! it that way. A test added here that wants either belongs in the crate's own
//! `#[cfg(test)]` modules instead.
//!
//! The module is included by path rather than through a library target: the
//! server crate is a binary, and adding a `lib` to it purely so a test could
//! reach four private items would widen a public surface nothing consumes. The
//! unit tests inside `lane.rs` compile and run here a second time as a
//! consequence; they are pure functions of their arguments, so that costs
//! milliseconds and proves the same thing twice.

#[allow(dead_code)]
#[path = "../src/lane.rs"]
mod lane;

use lane::{apply_deterministic, compute_admission_sha256, compute_config_sha256, CANONICAL};
use std::ffi::OsString;

#[test]
fn apply_deterministic_end_to_end() {
    // The developer shell running `cargo test` may itself carry variables this
    // scan refuses. Take the environment to a known-clean state first, and put
    // it back afterwards, so the assertions describe the policy rather than
    // whoever ran them.
    let restore: Vec<(OsString, OsString)> = std::env::vars_os()
        .filter(|(name, _)| lane::refused(&name.to_string_lossy()))
        .collect();
    for (name, _) in &restore {
        std::env::remove_var(name);
    }

    let clean = apply_deterministic().expect("a clean environment starts");
    assert_eq!(clean.sha256, compute_config_sha256());
    assert_eq!(clean.short(), "30d77c260803");
    assert_eq!(clean.admission_sha256, compute_admission_sha256());
    assert_eq!(clean.admission_short(), "45121fb83fef");
    for (key, value) in CANONICAL {
        assert_eq!(
            std::env::var(key).ok().as_deref(),
            Some(*value),
            "{key} should have been applied to the process environment"
        );
    }

    // The scan is single-shot, and this is what that means concretely: the
    // canonical keys the lane just wrote are not keys it accepts, so calling it
    // a second time refuses on the lane's own writes. Nothing may re-run it —
    // not a timer, not a health check. This inverts a property earlier revisions
    // of this module asserted, and the inversion is the design.
    let second = apply_deterministic().expect_err("the scan cannot be re-run");
    assert!(second.contains("CAMELID_DETERMINISTIC"));

    for (key, _) in CANONICAL {
        std::env::remove_var(key);
    }

    // An operator override is refused before the lane writes anything, so a
    // refused start leaves the environment exactly as it found it.
    std::env::set_var("CAMELID_ROPE_PAIRING", "interleaved");
    let refused = apply_deterministic().expect_err("an engine override refuses the start");
    assert!(refused.contains("CAMELID_ROPE_PAIRING"));
    assert!(
        std::env::var_os("CAMELID_DETERMINISTIC").is_none(),
        "the refusal path must not have written the canonical vector"
    );
    std::env::remove_var("CAMELID_ROPE_PAIRING");

    // The same, for the foreign name whose only spelling in `rayon-core` is a
    // fallback: a scan that reads the process environment has to see it there,
    // not merely in a table.
    std::env::set_var("RAYON_RS_NUM_CPUS", "2");
    let aliased = apply_deterministic().expect_err("the deprecated pool-width spelling refuses");
    assert!(aliased.contains("RAYON_RS_NUM_CPUS"));
    assert!(
        std::env::var_os("CAMELID_DETERMINISTIC").is_none(),
        "the refusal path must not have written the canonical vector"
    );
    std::env::remove_var("RAYON_RS_NUM_CPUS");

    // A permitted variable does not block the start — and this is the row whose
    // admission the two digests structurally cannot describe. Start twice
    // against two different model paths: both starts succeed, and both digests
    // come out byte-identical, because neither preimage contains the model file.
    // The replicas nonetheless serve logits from different weights.
    //
    // That is the design, not a gap in it, and it is the reason the model digest
    // is a third published field rather than more bytes in the first two: what
    // tells these two replicas apart is `x-camelid-model-sha256`, taken from the
    // file that was actually loaded. Admission publishes a policy; it has never
    // promised that an admitted name is numerically inert. Any claim in this
    // tree that reads "nothing admitted can change the output" is false here,
    // one row before it reaches CAMELID_ENTERPRISE_THREADS.
    std::env::set_var("CAMELID_ENTERPRISE_MODEL", "/models/model.gguf");
    let permitted_start = apply_deterministic().expect("a permitted variable starts");
    assert_eq!(permitted_start.sha256, compute_config_sha256());
    for (key, _) in CANONICAL {
        std::env::remove_var(key);
    }

    std::env::set_var("CAMELID_ENTERPRISE_MODEL", "/models/other.gguf");
    let other_weights = apply_deterministic().expect("any model path is equally permitted");
    assert_eq!(
        other_weights.sha256, permitted_start.sha256,
        "the configuration digest does not distinguish two replicas started against different \
         weights, and must not be read as if it did"
    );
    assert_eq!(
        other_weights.admission_sha256, permitted_start.admission_sha256,
        "neither does the admission digest: the policy admitted the same name both times"
    );
    std::env::remove_var("CAMELID_ENTERPRISE_MODEL");

    for (key, _) in CANONICAL {
        std::env::remove_var(key);
    }
    for (name, value) in &restore {
        std::env::set_var(name, value);
    }
}
