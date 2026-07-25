//! Camelid Enterprise deterministic replica composition.
//!
//! The pinned Camelid engine owns request and response schemas. This crate owns
//! the deployment contract around that engine: the exact revision, the declared
//! route surface, deterministic-lane configuration, and response attribution.
//!
//! Everything a replica *is* lives here rather than in the binary, so the
//! contract tests and the model-backed conformance harness are written against
//! the same code a served replica is built from rather than a restatement of it.
//! [`replica_router`] is the whole of that composition: the binary around it
//! supplies the CLI, the host probe, the worker pool, the listener and shutdown,
//! and composes nothing of its own. That is deliberate — a filter deleted from
//! the served stack has to be a change to code the tests drive, not a change to
//! a composition only production runs.

pub mod attribution;
pub mod contract;
mod lane;
pub mod surface;

use axum::middleware;
use axum::Router;
use std::path::Path;
use std::sync::Arc;

pub use attribution::{Attribution, ModelIdentity, WorkerThreads};
pub use lane::{apply_deterministic, ConfigVector, ENGINE_PIN};
pub use surface::ServedModel;

/// Load the weights this replica was started with, then compose the router it
/// serves from. Returns that router and the key the engine filed the weights
/// under.
///
/// This is the single composition point for the surface a client meets, and the
/// order of the three steps is the design rather than convenience.
///
/// The load goes through the **unfiltered** `engine` router, because the route
/// it uses is one [`surface::serve_only_the_lane`] refuses — if it were not, the
/// filter would have a hole exactly where the control plane is. The engine
/// exposes model loading over HTTP and nowhere else, so this is an HTTP request;
/// it is not a request over a socket. See [`load_startup_model`].
///
/// The digest is re-verified between the load and the composition, because the
/// engine loads by path string rather than by handle and this replica is about
/// to publish a digest for the bytes it hashed.
///
/// The served view is composed only after the load returns, because the body
/// filter is written against the key the load reports — a value this
/// distribution reads back from the engine rather than predicts.
pub async fn replica_router(
    engine: Router,
    canonical: &Path,
    requested: &Path,
    identity: Attribution,
) -> Result<(Router, String), Box<dyn std::error::Error>> {
    let model_id = load_startup_model(engine.clone(), canonical).await?;
    identity
        .model
        .verify_unchanged(canonical)
        .map_err(std::io::Error::other)?;
    let served_model = Arc::new(ServedModel::new(model_id.clone(), canonical, requested));
    Ok((served_router(engine, served_model, identity), model_id))
}

/// Compose the served view around an engine router whose model is already
/// loaded.
///
/// The layering is the whole content of this function, and each position is
/// load-bearing:
///
///   * the body filter is innermost, so it sees only requests the route filter
///     already admitted;
///   * the route filter sits above it;
///   * attribution is outermost, so a *refusal* from either filter is still a
///     response that says which replica emitted it, under what configuration and
///     which weights. A client that gets a `403` from a pool has to be able to
///     tell which replica refused it.
///
/// Split out from [`replica_router`] so a test can drive the exact production
/// stack against a `ServedModel` it constructs, without a multi-gigabyte load.
/// Nothing else in this workspace may restate this layering: a second copy is a
/// composition the tests pin and production does not use.
pub fn served_router(
    engine: Router,
    served_model: Arc<ServedModel>,
    identity: Attribution,
) -> Router {
    engine
        .layer(middleware::from_fn_with_state(
            served_model,
            surface::pin_generation_to_the_served_model,
        ))
        .layer(middleware::from_fn(surface::serve_only_the_lane))
        .layer(middleware::from_fn_with_state(
            identity,
            attribution::attribute,
        ))
}

/// The engine's router with attribution stamped on every response, and **no
/// served-surface filtering at all**.
///
/// This is not the surface a client meets — [`replica_router`] is — and the
/// distinction is worth keeping rather than collapsing. The contract tests use
/// this to pin what the *pinned engine* answers under attribution: its typed
/// `501` on the compatibility routes, its `malformed_json` on a body the
/// replica's own body filter would have refused first, its `Allow` headers. Put
/// the filters in front of those and they would be pinning this distribution's
/// answers instead, which is a different claim and one the served-stack tests
/// already make.
///
/// It carries no `ServedModel`, so a generation request reaching it is resolved
/// by the engine on the engine's own terms. Nothing serves this to a client.
pub fn attributed_router(state: camelid::api::AppState, identity: Attribution) -> Router {
    camelid::api::router_with_state(state).layer(middleware::from_fn_with_state(
        identity,
        attribution::attribute,
    ))
}

/// Load the model this replica was started with, without a route to load it,
/// and report the key the engine filed it under.
///
/// That key is a return value rather than something this crate derives, and the
/// distinction is load-bearing. The engine keys its loaded models by the GGUF's
/// `general.name` when the metadata carries one and by the file stem when it
/// does not, and that choice belongs to the engine at whatever revision is
/// pinned. The served-surface filter matches request bodies against it, so a key
/// guessed here and wrong there would refuse every request naming the model the
/// replica is serving — a rule about the engine's state, read back from the
/// engine, rather than a copy of its logic that goes stale at a pin bump.
///
/// The engine exposes model loading over HTTP and nowhere else — its loader is
/// module-private and its state exposes no in-process entry point — so this is
/// an HTTP request. It is not a request over a socket. The unfiltered router is
/// a `tower` service, so the load is dispatched into it in-process: same
/// handler, same middleware stack, same shared state, and no listener anywhere
/// that could accept a connection from anything else.
///
/// That is the point. `POST /api/models/load` is unauthenticated and changes
/// what a replica serves without changing anything it publishes about itself. A
/// private loopback port carrying the unfiltered router would shrink the window
/// in which that is reachable; dispatching in-process removes the window. It
/// also removes a bind that can fail, an ephemeral port, a retry loop for the
/// scheduling gap before the listener is polled, and a hand-rolled HTTP parse.
///
/// Source-restricting the route to loopback instead was considered and is
/// strictly weaker exactly where it matters: in a pod every sidecar shares the
/// network namespace and is "loopback", and on a host bound to `0.0.0.0` so is
/// every local process. It weakens "nothing can swap the weights" to "no remote
/// client can", and it leaves the control plane permanently mounted rather than
/// never mounted.
///
/// Blocking rather than spawning is the operator-visible consequence, and the
/// one to know about: the process prints that it is listening, then goes quiet
/// for the length of the load. Nothing is lost — the engine reports itself not
/// generation-ready until the load completes anyway, and every deployment
/// artifact here already treats listening as distinct from ready — and what is
/// gained is that no request can reach a served router whose model is not yet
/// the one this replica hashed. It also fixes a second problem: posting the load
/// to the replica's *own serving listener*, as this did before, made the load an
/// in-flight request there, so a stop signal three seconds into startup blocked
/// for the whole read while the documented drain probe reported an empty queue
/// the entire time.
///
/// No timeout. A timeout firing mid-load would abandon the engine's model
/// transition in a state this process cannot reason about; the bound is the
/// supervisor's own startup probe, which restarts the container when it gives
/// up. On failure this returns an error rather than exiting from a detached
/// task, so the caller decides — which also removes a race in which the loader
/// could kill the process while the listener was already accepting.
pub async fn load_startup_model(
    engine: Router,
    model: &Path,
) -> Result<String, Box<dyn std::error::Error>> {
    use axum::body::{to_bytes, Body};
    use axum::http::{header::CONTENT_TYPE, Request, StatusCode};
    use tower::ServiceExt;

    let body = serde_json::json!({ "path": model.to_string_lossy() }).to_string();
    let request = Request::builder()
        .method("POST")
        .uri("/api/models/load")
        .header(CONTENT_TYPE, "application/json")
        .body(Body::from(body))?;

    // `Error = Infallible`, so the only outcomes here are a status and a body.
    let response = engine.oneshot(request).await?;
    let status = response.status();
    // Bounded, and generously: the success reply is the engine's whole
    // loaded-model record, GGUF metadata and per-tensor descriptors included, so
    // it is megabytes for an ordinary model and a tight limit here reads as "the
    // engine reported no id". A bound is still worth having — this is a reply to
    // a request this process made, and an unbounded read would turn a strange
    // one into an allocation — but it has to sit above the record, not above a
    // guess at it.
    let bytes = to_bytes(response.into_body(), 256 * 1024 * 1024)
        .await
        .unwrap_or_default();
    if status == StatusCode::OK {
        return serde_json::from_slice::<serde_json::Value>(&bytes)
            .ok()
            .and_then(|value| value.get("id")?.as_str().map(ToOwned::to_owned))
            .filter(|id| !id.is_empty())
            .ok_or_else(|| {
                // The load succeeded and the replica still cannot start, because
                // the served-surface filter is written against this key. Serving
                // without it would mean serving a generation route whose model
                // field nothing checks.
                format!(
                    "the engine loaded {} but did not report the id it filed it under, so this \
                     replica cannot pin generation requests to the model it hashed",
                    model.display()
                )
                .into()
            });
    }
    let detail = String::from_utf8_lossy(&bytes).trim().to_string();
    Err(format!(
        "could not load the model at {}: the engine answered HTTP {}{}",
        model.display(),
        status.as_u16(),
        if detail.is_empty() {
            String::new()
        } else {
            format!(": {detail}")
        }
    )
    .into())
}
