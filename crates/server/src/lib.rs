//! Camelid Enterprise deterministic replica composition.
//!
//! The pinned Camelid engine owns request and response schemas. This crate owns
//! the deployment contract around that engine: the exact revision, the declared
//! route surface, deterministic-lane configuration, and response attribution.

mod attribution;
pub mod contract;
pub mod in_tree;
mod lane;

use attribution::Attribution;
use axum::middleware;
use axum::Router;
use std::path::PathBuf;
use std::sync::Arc;

pub use lane::{apply_deterministic, ConfigVector, ENGINE_PIN};

/// Build the exact attributed router served by a deterministic replica.
pub fn attributed_router(
    state: camelid::api::AppState,
    config_sha256: String,
    host: String,
    receipts: Option<PathBuf>,
) -> Router {
    let attribution = Attribution {
        lane: "deterministic",
        config_sha256: Arc::new(config_sha256),
        host: Arc::new(host),
        receipts: receipts.map(Arc::new),
    };
    camelid::api::router_with_state(state).layer(middleware::from_fn_with_state(
        attribution,
        attribution::attribute,
    ))
}
