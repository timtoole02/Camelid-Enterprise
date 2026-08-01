//! The Q8_0 dot this build serves through, paired with the posture it publishes.
//!
//! This lives in the library and not in the binary for the reason the crate doc
//! gives: what a replica *is* has to be code the tests drive, not code only
//! production runs. The selection used to sit in `serve`, where nothing could
//! reach it — so the one wire value this change adds, the numeric posture, was
//! published by a function with no test and asserted everywhere else against
//! hand-written literals. A harness that supplies the posture it then asserts is
//! comparing a copy of itself.
//!
//! The pair is returned together, never separately. Two independent cfg cascades
//! — one choosing the kernel, one choosing the posture — is exactly how a
//! published claim drifts from the code making it, which is the whole reason a
//! posture is declared beside its kernel rather than at the point of use.
//! Returning one tuple makes "take the kernel without its posture" impossible to
//! express at any call site.

use engine_core::posture::NumericPosture;
use engine_core::tensor::Q8DotRows;

/// macOS: engine-macos's NEON dot, with the posture written here as a literal.
///
/// The one arm that does not read a constant from the crate supplying the
/// kernel, and the exception is recorded rather than smoothed over: engine-macos
/// exports no `POSTURE` today. When it declares one beside its dot, this literal
/// is replaced by that constant and the arm stops being an exception — it is one
/// edit, not a second place to keep in sync, which is why the pair is bound here
/// rather than assembled by the caller.
///
/// What the literal is resting on, stated so it is not resting on nothing: the
/// NEON dot has to reduce from the reference's `-0.0` seed —
/// `empty_and_all_negative_zero_rows_keep_the_sign_of_zero` pins that for the
/// reference, and `an_empty_reduction_keeps_the_reference_sign_of_zero` pins it
/// for the Windows kernel. engine-macos's own bit-identity fuzz builds no
/// zero-length row and emits no negative scale, so it cannot reach the case at
/// all: this arm claims a property nothing on the macOS side asserts. That is a
/// gap in engine-macos's tests to close, not a reason to publish a weaker claim
/// from here.
#[cfg(target_os = "macos")]
const SELECTED: (Q8DotRows, NumericPosture) =
    (engine_macos::q8_0_dot_rows, NumericPosture::BitIdentical);

/// Windows: engine-windows's AVX2 dot, with the posture that crate declares
/// beside it. Nothing here restates the claim; `engine_windows::POSTURE` is the
/// only copy, so weakening the kernel and leaving the published value behind is
/// not expressible.
#[cfg(target_os = "windows")]
const SELECTED: (Q8DotRows, NumericPosture) =
    (engine_windows::q8_0_dot_rows, engine_windows::POSTURE);

/// Everywhere else: the portable reference itself, whose posture is
/// bit-identical by definition rather than by measurement — it is the definition
/// every other posture in this workspace is stated against.
///
/// Narrowed against every arm above rather than left as `not(macos)`: two live
/// arms would both compile, and the replica would load a multi-gigabyte model
/// twice and serve from whichever `let` won the shadowing.
#[cfg(not(any(target_os = "macos", target_os = "windows")))]
const SELECTED: (Q8DotRows, NumericPosture) = (
    engine_core::tensor::q8_0_dot_rows,
    NumericPosture::BitIdentical,
);

/// The kernel this build serves the Q8_0 matmul leaf through, and the numeric
/// contract it holds to.
///
/// Compile-time and target-chosen: not operator-settable by flag or by
/// environment, because a posture an operator can set is a replica declaring a
/// numeric contract it did not prove. It is an input to neither published
/// digest.
///
/// Every caller takes both or neither. `serve` passes the first into
/// `LoadedModel::load_with_q8_dot` and the second into [`crate::Attribution`],
/// and the model-backed conformance harness does the same, so the harness runs
/// the kernel a served replica runs and publishes the posture a served replica
/// publishes rather than a literal of its own.
pub const fn selected() -> (Q8DotRows, NumericPosture) {
    SELECTED
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::attribution::{attribute, Attribution, ModelIdentity, WorkerThreads};
    use axum::body::Body;
    use axum::http::Request;
    use axum::middleware::from_fn_with_state;
    use axum::routing::get;
    use axum::Router;
    use std::path::Path;
    use std::sync::Arc;
    use tower::ServiceExt;

    /// The posture this build selects is the token a response actually carries.
    ///
    /// This is the only test in the workspace whose published posture comes from
    /// the selection rather than from a literal the test wrote, which is what
    /// gives the assertion power: weaken `engine_windows::POSTURE` — or point an
    /// arm above at a kernel that declares something else — and this fails,
    /// where every other posture assertion in the tree would stay green because
    /// it supplies its own `BitIdentical`.
    ///
    /// It runs on every host with no model, so it holds the platform constant to
    /// the wire on the same job that compiles it.
    #[tokio::test]
    async fn the_selected_posture_is_the_token_a_response_publishes() {
        let identity = Attribution {
            lane: "deterministic",
            config_sha256: Arc::new("0".repeat(64)),
            admission_sha256: Arc::new("1".repeat(64)),
            model: ModelIdentity::of_file(Path::new("Cargo.toml"))
                .expect("this crate's own manifest is readable"),
            host: Arc::new("kernel-selection-test/host".to_string()),
            workers: WorkerThreads::resolved(1),
            posture: selected().1,
            receipts: None,
        };
        let app = Router::new()
            .route("/v1/health", get(|| async { "ok" }))
            .layer(from_fn_with_state(identity, attribute));

        let response = app
            .oneshot(Request::get("/v1/health").body(Body::empty()).unwrap())
            .await
            .unwrap();

        assert_eq!(response.headers()["x-camelid-posture"], "bit-identical");
    }

    /// The Windows arm publishes the constant engine-windows declares beside its
    /// kernel, not a copy of it made here. Target-gated because it is a statement
    /// about that arm and can only be executed where that arm is compiled.
    #[cfg(target_os = "windows")]
    #[test]
    fn the_windows_posture_is_the_constant_the_supplying_crate_declares() {
        assert_eq!(selected().1, engine_windows::POSTURE);
    }
}
