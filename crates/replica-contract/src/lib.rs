//! Dependency-free public HTTP contract shared by replicas and gateways.
//!
//! This crate contains only the intentionally stable client-facing surface. It
//! does not expose the replica's private model, runtime, workspace, legacy, or
//! diagnostic routes.

pub const CONTRACT_ID: &str = "camelid-enterprise-replica-http-v1";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum HttpMethod {
    Get,
    Post,
    Delete,
}

impl HttpMethod {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Get => "GET",
            Self::Post => "POST",
            Self::Delete => "DELETE",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Stability {
    Contractual,
    PinnedImplementation,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Surface {
    PublicInference,
    PublicCompatibility,
    ReplicaOperations,
    WorkspaceApplication,
    LegacyCompatibility,
    Diagnostics,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RouteSpec {
    /// Route path in **axum path syntax**: path parameters are written `:name`
    /// (e.g. `/v1/models/:model`).
    ///
    /// This is a deliberate, load-bearing coupling worth naming, because this
    /// crate is otherwise framework-neutral. The path grammar stored here is the
    /// one both current consumers use — the pinned engine's replica router and
    /// the gateway — which are both on axum 0.7 and register `path` verbatim.
    /// axum 0.8 changed parameter syntax to `{name}`; under it a `:model`
    /// segment silently becomes a *literal* path segment: the router still
    /// builds, but the parameterized route stops matching real requests (404)
    /// with no compile error. A consumer on a framework with different path
    /// syntax must translate `path` rather than register it verbatim; and
    /// `probe_path` exists so conformance checks hit the real matcher and would
    /// catch exactly this kind of silent break.
    pub path: &'static str,
    /// Concrete path used by executable conformance checks for parameterized
    /// routes. Equal to `path` for non-parameterized routes.
    pub probe_path: &'static str,
    pub methods: &'static [HttpMethod],
    pub stability: Stability,
    pub surface: Surface,
}

const GET: &[HttpMethod] = &[HttpMethod::Get];
const POST: &[HttpMethod] = &[HttpMethod::Post];

const fn route(
    path: &'static str,
    probe_path: &'static str,
    methods: &'static [HttpMethod],
    surface: Surface,
) -> RouteSpec {
    RouteSpec {
        path,
        probe_path,
        methods,
        stability: Stability::Contractual,
        surface,
    }
}

/// Complete client-facing route set for [`CONTRACT_ID`].
///
/// Paths use axum path syntax (`:param`); see [`RouteSpec::path`] for the
/// framework-coupling caveat that entails.
pub const PUBLIC_ROUTES: &[RouteSpec] = &[
    route("/v1/health", "/v1/health", GET, Surface::PublicInference),
    route("/v1/models", "/v1/models", GET, Surface::PublicInference),
    route(
        "/v1/models/:model",
        "/v1/models/contract-probe",
        GET,
        Surface::PublicInference,
    ),
    route(
        "/v1/completions",
        "/v1/completions",
        POST,
        Surface::PublicInference,
    ),
    route(
        "/v1/chat/completions",
        "/v1/chat/completions",
        POST,
        Surface::PublicInference,
    ),
    route(
        "/v1/embeddings",
        "/v1/embeddings",
        POST,
        Surface::PublicCompatibility,
    ),
    route(
        "/v1/responses",
        "/v1/responses",
        POST,
        Surface::PublicCompatibility,
    ),
    route(
        "/v1/messages",
        "/v1/messages",
        POST,
        Surface::PublicCompatibility,
    ),
    route(
        "/v1/rerank",
        "/v1/rerank",
        POST,
        Surface::PublicCompatibility,
    ),
    route(
        "/v1/reranking",
        "/v1/reranking",
        POST,
        Surface::PublicCompatibility,
    ),
];

pub fn contractual_routes() -> impl Iterator<Item = &'static RouteSpec> {
    PUBLIC_ROUTES.iter()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn public_contract_is_unique_and_contains_only_v1_routes() {
        let mut paths = std::collections::HashSet::new();
        assert_eq!(PUBLIC_ROUTES.len(), 10);
        for route in PUBLIC_ROUTES {
            assert!(paths.insert(route.path), "duplicate route: {}", route.path);
            assert!(route.path.starts_with("/v1/"));
            assert!(!route.methods.is_empty());
            assert_eq!(route.stability, Stability::Contractual);
        }
    }
}
