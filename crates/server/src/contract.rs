//! Versioned route contract for the pinned replica API.
//!
//! `Contractual` routes are the client-facing surface Camelid Enterprise
//! intentionally carries forward. `PinnedImplementation` routes are inventoried
//! because they exist in the pinned engine, but are replica-private and receive
//! no compatibility promise from Camelid Enterprise.

use crate::ENGINE_PIN;
pub use replica_contract::{
    contractual_routes, HttpMethod, RouteSpec, Stability, Surface, CONTRACT_ID, PUBLIC_ROUTES,
};

pub const CONTRACT_ENGINE_PIN: &str = ENGINE_PIN;

#[cfg(test)]
const GET: &[HttpMethod] = &[HttpMethod::Get];
#[cfg(test)]
const POST: &[HttpMethod] = &[HttpMethod::Post];
#[cfg(test)]
const GET_POST: &[HttpMethod] = &[HttpMethod::Get, HttpMethod::Post];
#[cfg(test)]
const GET_DELETE: &[HttpMethod] = &[HttpMethod::Get, HttpMethod::Delete];
#[cfg(test)]
const POST_DELETE: &[HttpMethod] = &[HttpMethod::Post, HttpMethod::Delete];

#[cfg(test)]
macro_rules! route {
    ($path:literal, $methods:expr, $stability:ident, $surface:ident) => {
        RouteSpec {
            path: $path,
            probe_path: $path,
            methods: $methods,
            stability: Stability::$stability,
            surface: Surface::$surface,
        }
    };
    ($path:literal => $probe:literal, $methods:expr, $stability:ident, $surface:ident) => {
        RouteSpec {
            path: $path,
            probe_path: $probe,
            methods: $methods,
            stability: Stability::$stability,
            surface: Surface::$surface,
        }
    };
}

/// Source-reviewed inventory of every explicit route registered by
/// `camelid::api::router_with_state` at [`CONTRACT_ENGINE_PIN`]. Axum exposes no
/// inverse route-tree introspection: tests prove every declaration exists with
/// these methods, while the pinned count and immutable engine revision make a
/// complete source re-review mandatory when the revision moves. The embedded
/// WebUI fallback is intentionally not a route contract.
#[cfg(test)]
const PRIVATE_ROUTES: &[RouteSpec] = &[
    route!("/health", GET, PinnedImplementation, Diagnostics),
    route!("/api/capabilities", GET, PinnedImplementation, Diagnostics),
    route!(
        "/api/runtime/gpu",
        GET_POST,
        PinnedImplementation,
        ReplicaOperations
    ),
    route!(
        "/api/telemetry/stream",
        GET,
        PinnedImplementation,
        Diagnostics
    ),
    route!("/execution-plan", GET, PinnedImplementation, Diagnostics),
    route!(
        "/api/execution-plan",
        GET,
        PinnedImplementation,
        Diagnostics
    ),
    route!(
        "/api/models/load",
        POST,
        PinnedImplementation,
        ReplicaOperations
    ),
    route!(
        "/api/models/inspect",
        POST,
        PinnedImplementation,
        ReplicaOperations
    ),
    route!(
        "/api/models/unload",
        POST,
        PinnedImplementation,
        ReplicaOperations
    ),
    route!(
        "/api/models/current",
        GET,
        PinnedImplementation,
        ReplicaOperations
    ),
    route!(
        "/api/models/verify",
        GET_POST,
        PinnedImplementation,
        ReplicaOperations
    ),
    route!(
        "/api/models/metadata",
        GET,
        PinnedImplementation,
        ReplicaOperations
    ),
    route!(
        "/api/models/tokenizer",
        GET,
        PinnedImplementation,
        ReplicaOperations
    ),
    route!(
        "/api/models/tokenizer/encode",
        POST,
        PinnedImplementation,
        ReplicaOperations
    ),
    route!(
        "/api/models/tokenizer/decode",
        POST,
        PinnedImplementation,
        ReplicaOperations
    ),
    route!("/tokenize", POST, PinnedImplementation, LegacyCompatibility),
    route!(
        "/detokenize",
        POST,
        PinnedImplementation,
        LegacyCompatibility
    ),
    route!(
        "/apply-template",
        POST,
        PinnedImplementation,
        LegacyCompatibility
    ),
    route!("/models", GET, PinnedImplementation, LegacyCompatibility),
    route!(
        "/models/load",
        POST,
        PinnedImplementation,
        LegacyCompatibility
    ),
    route!(
        "/models/unload",
        POST,
        PinnedImplementation,
        LegacyCompatibility
    ),
    route!(
        "/props",
        GET_POST,
        PinnedImplementation,
        LegacyCompatibility
    ),
    route!(
        "/slots",
        GET_POST,
        PinnedImplementation,
        LegacyCompatibility
    ),
    route!("/metrics", GET, PinnedImplementation, LegacyCompatibility),
    route!(
        "/completion",
        POST,
        PinnedImplementation,
        LegacyCompatibility
    ),
    route!("/infill", POST, PinnedImplementation, LegacyCompatibility),
    route!(
        "/embedding",
        POST,
        PinnedImplementation,
        LegacyCompatibility
    ),
    route!(
        "/embeddings",
        POST,
        PinnedImplementation,
        LegacyCompatibility
    ),
    route!("/rerank", POST, PinnedImplementation, LegacyCompatibility),
    route!(
        "/reranking",
        POST,
        PinnedImplementation,
        LegacyCompatibility
    ),
    route!(
        "/api/generation/sessions",
        GET_POST,
        PinnedImplementation,
        ReplicaOperations
    ),
    route!(
        "/api/generation/preflight",
        POST,
        PinnedImplementation,
        ReplicaOperations
    ),
    route!(
        "/api/agent/workspace/models",
        GET,
        PinnedImplementation,
        WorkspaceApplication
    ),
    route!(
        "/api/agent/workspace/threads",
        GET,
        PinnedImplementation,
        WorkspaceApplication
    ),
    route!("/api/agent/workspace/threads/:id" => "/api/agent/workspace/threads/contract-probe", GET_DELETE, PinnedImplementation, WorkspaceApplication),
    route!("/api/agent/workspace/threads/:id/compact" => "/api/agent/workspace/threads/contract-probe/compact", POST_DELETE, PinnedImplementation, WorkspaceApplication),
    route!(
        "/api/agent/workspace/browse",
        GET,
        PinnedImplementation,
        WorkspaceApplication
    ),
    route!(
        "/api/agent/workspace/sessions",
        POST,
        PinnedImplementation,
        WorkspaceApplication
    ),
    route!("/api/agent/workspace/sessions/:id" => "/api/agent/workspace/sessions/contract-probe", GET_DELETE, PinnedImplementation, WorkspaceApplication),
    route!("/api/agent/workspace/sessions/:id/events" => "/api/agent/workspace/sessions/contract-probe/events", GET, PinnedImplementation, WorkspaceApplication),
    route!("/api/agent/workspace/sessions/:id/messages" => "/api/agent/workspace/sessions/contract-probe/messages", POST, PinnedImplementation, WorkspaceApplication),
    route!("/api/agent/workspace/sessions/:id/decisions" => "/api/agent/workspace/sessions/contract-probe/decisions", POST, PinnedImplementation, WorkspaceApplication),
    route!(
        "/api/models/local",
        GET,
        PinnedImplementation,
        ReplicaOperations
    ),
    route!(
        "/api/models/local/delete",
        POST,
        PinnedImplementation,
        ReplicaOperations
    ),
    route!(
        "/api/models/runnable-receipt",
        GET,
        PinnedImplementation,
        ReplicaOperations
    ),
    route!(
        "/api/models/runnable-smoke",
        POST,
        PinnedImplementation,
        ReplicaOperations
    ),
    route!(
        "/api/models/catalog",
        GET,
        PinnedImplementation,
        ReplicaOperations
    ),
    route!(
        "/api/models/catalog/install",
        POST,
        PinnedImplementation,
        ReplicaOperations
    ),
    route!(
        "/api/models/catalog/downloads",
        GET,
        PinnedImplementation,
        ReplicaOperations
    ),
    route!(
        "/api/models/catalog/cancel",
        POST,
        PinnedImplementation,
        ReplicaOperations
    ),
    route!(
        "/api/models/catalog/ack",
        POST,
        PinnedImplementation,
        ReplicaOperations
    ),
];

#[cfg(test)]
fn all_routes() -> impl Iterator<Item = &'static RouteSpec> {
    PRIVATE_ROUTES.iter().chain(PUBLIC_ROUTES)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::attribution::{Attribution, ModelIdentity, WorkerThreads};
    use axum::body::{to_bytes, Body};
    use axum::http::header::ALLOW;
    use axum::http::{Method, Request, StatusCode};
    use axum::Router;
    use engine_core::posture::NumericPosture;
    use std::path::Path;
    use std::sync::Arc;
    use tower::ServiceExt;

    const TEST_SHA: &str = "30d77c2608036f8475372ace9ec125ffc5fa16d8d63f0355a08c32c69f4449b7";
    /// Distinct from every other digest here, so no assertion can pass by
    /// reading a neighbouring field.
    const TEST_ADMISSION_SHA: &str =
        "45121fb83fef631f8464c32dada6100b23f0a0af80347031f812803ee9ec2a09";
    const TEST_HOST: &str = "linux/x86_64 cores=8 simd=avx2+fma";
    const TEST_THREADS: usize = 6;

    fn trace(path: &str) -> Request<Body> {
        Request::builder()
            .method(Method::TRACE)
            .uri(path)
            .body(Body::empty())
            .unwrap()
    }

    fn expected_allow(route: &RouteSpec) -> Vec<&'static str> {
        let mut methods: Vec<&str> = route.methods.iter().map(|method| method.as_str()).collect();
        if route.methods.contains(&HttpMethod::Get) {
            methods.push("HEAD");
        }
        methods.sort_unstable();
        methods
    }

    /// The identity these tests publish. The model digest is a real digest of a
    /// real file — this crate's own manifest — because [`ModelIdentity`] has no
    /// constructor that invents one, which is the point: a replica cannot
    /// publish a model digest it did not compute from bytes it read.
    fn test_identity() -> Attribution {
        Attribution {
            lane: "deterministic",
            config_sha256: Arc::new(TEST_SHA.to_string()),
            admission_sha256: Arc::new(TEST_ADMISSION_SHA.to_string()),
            model: ModelIdentity::of_file(Path::new("Cargo.toml"))
                .expect("this crate's own manifest is readable"),
            host: Arc::new(TEST_HOST.to_string()),
            workers: WorkerThreads::resolved(TEST_THREADS),
            posture: NumericPosture::BitIdentical,
            receipts: None,
        }
    }

    fn attributed_no_model_router() -> Router {
        crate::attributed_router(camelid::api::AppState::default(), test_identity())
    }

    async fn json(response: axum::response::Response) -> serde_json::Value {
        let bytes = to_bytes(response.into_body(), 1024 * 1024).await.unwrap();
        serde_json::from_slice(&bytes).unwrap()
    }

    /// Every identity field, on every response these tests drive.
    ///
    /// All seven and not the first three, and the omission this repairs was not
    /// cosmetic. `attributed_router` is the composition this file and the
    /// model-backed conformance harness both run, and it is the only place the
    /// admission digest, the model digest and the worker width are published
    /// without a served-surface filter in front of them. While those three went
    /// unasserted here, a replica could publish a fabricated admission digest, a
    /// digest of a file that is not its weights, and a hardcoded thread count,
    /// and the whole suite stayed green — which is precisely the guarantee this
    /// crate exists to make checkable.
    fn assert_attribution(response: &axum::response::Response) {
        let identity = test_identity();
        assert_eq!(response.headers()["x-camelid-lane"], "deterministic");
        assert_eq!(
            response.headers()["x-camelid-config-sha256"],
            &TEST_SHA[..12]
        );
        assert_eq!(
            response.headers()["x-camelid-admission-sha256"],
            &TEST_ADMISSION_SHA[..12]
        );
        assert_eq!(
            response.headers()["x-camelid-model-sha256"],
            identity.model.short()
        );
        assert_eq!(response.headers()["x-camelid-host"], TEST_HOST);
        assert_eq!(
            response.headers()["x-camelid-worker-threads"],
            TEST_THREADS.to_string()
        );
        assert_eq!(response.headers()["x-camelid-posture"], "bit-identical");
    }

    #[test]
    fn route_declarations_are_unique_and_well_formed() {
        assert_eq!(PUBLIC_ROUTES.len(), 10);
        assert_eq!(PRIVATE_ROUTES.len(), 51);
        assert_eq!(
            all_routes().count(),
            61,
            "review the full pinned implementation inventory when routes change"
        );
        let mut paths = std::collections::HashSet::new();
        for route in all_routes() {
            assert!(paths.insert(route.path), "duplicate route: {}", route.path);
            assert!(route.path.starts_with('/'));
            assert!(route.probe_path.starts_with('/'));
            assert!(!route.methods.is_empty());
        }
    }

    #[test]
    fn public_contract_is_exactly_the_declared_v1_surface() {
        let actual: Vec<(&str, &str)> = contractual_routes()
            .flat_map(|route| {
                route
                    .methods
                    .iter()
                    .map(move |method| (method.as_str(), route.path))
            })
            .collect();
        assert_eq!(
            actual,
            [
                ("GET", "/v1/health"),
                ("GET", "/v1/models"),
                ("GET", "/v1/models/:model"),
                ("POST", "/v1/completions"),
                ("POST", "/v1/chat/completions"),
                ("POST", "/v1/embeddings"),
                ("POST", "/v1/responses"),
                ("POST", "/v1/messages"),
                ("POST", "/v1/rerank"),
                ("POST", "/v1/reranking"),
            ]
        );
    }

    #[test]
    fn contract_document_lists_every_public_method_and_path() {
        let document = include_str!("../../../docs/contracts/replica-http-v1.md");
        assert!(document.contains(CONTRACT_ID));
        assert!(document.contains(CONTRACT_ENGINE_PIN));
        for route in contractual_routes() {
            for method in route.methods {
                let documented_path = route.path.replace(":model", "{model}");
                let row = format!("| `{}` | `{documented_path}` |", method.as_str());
                assert!(
                    document.contains(&row),
                    "contract document is missing {row}"
                );
            }
        }
    }

    #[tokio::test]
    async fn pinned_router_conforms_to_declared_paths_and_methods() {
        let app = camelid::api::router_with_state(camelid::api::AppState::default());
        for route in all_routes() {
            let response = app.clone().oneshot(trace(route.probe_path)).await.unwrap();
            assert_eq!(
                response.status(),
                StatusCode::METHOD_NOT_ALLOWED,
                "undeclared TRACE unexpectedly accepted for {}",
                route.path
            );
            let mut actual: Vec<&str> = response
                .headers()
                .get(ALLOW)
                .unwrap_or_else(|| panic!("405 response missing Allow for {}", route.path))
                .to_str()
                .unwrap()
                .split(',')
                .map(str::trim)
                .collect();
            actual.sort_unstable();
            assert_eq!(
                actual,
                expected_allow(route),
                "method contract drift for {}",
                route.path
            );
        }
    }

    #[tokio::test]
    async fn no_model_health_is_live_but_not_generation_ready() {
        let response = attributed_no_model_router()
            .oneshot(Request::get("/v1/health").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        assert_attribution(&response);
        let body = json(response).await;
        assert_eq!(body["ok"], true);
        assert_eq!(body["engine"], "camelid");
        assert_eq!(body["loaded_now"], false);
        assert_eq!(body["generation_ready"], false);
        assert!(body["active_model_id"].is_null());
        assert_eq!(body["backend"], "none");
        assert_eq!(body["engine_queue_depth"], 0);
    }

    #[tokio::test]
    async fn no_model_discovery_is_an_empty_openai_list() {
        let response = attributed_no_model_router()
            .oneshot(Request::get("/v1/models").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        assert_attribution(&response);
        let body = json(response).await;
        assert_eq!(body, serde_json::json!({ "object": "list", "data": [] }));
    }

    #[tokio::test]
    async fn missing_model_lookup_is_a_typed_attributed_404() {
        let response = attributed_no_model_router()
            .oneshot(
                Request::get("/v1/models/not-loaded")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::NOT_FOUND);
        assert_attribution(&response);
        let body = json(response).await;
        assert_eq!(body["error"]["type"], "invalid_request");
        assert_eq!(body["error"]["code"], "model_not_found");
        assert_eq!(body["error"]["param"], "model");
    }

    #[tokio::test]
    async fn compatibility_routes_fail_explicitly_instead_of_falling_back() {
        for (path, code) in [
            ("/v1/embeddings", "unsupported_embeddings"),
            ("/v1/responses", "unsupported_responses"),
            ("/v1/messages", "unsupported_messages"),
            ("/v1/rerank", "unsupported_reranking"),
            ("/v1/reranking", "unsupported_reranking"),
        ] {
            let response = attributed_no_model_router()
                .oneshot(
                    Request::post(path)
                        .header("content-type", "application/json")
                        .body(Body::from("{}"))
                        .unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(response.status(), StatusCode::NOT_IMPLEMENTED, "{path}");
            assert_attribution(&response);
            let body = json(response).await;
            assert_eq!(body["error"]["type"], "not_implemented", "{path}");
            assert_eq!(body["error"]["code"], code, "{path}");
            assert_eq!(body["error"]["param"], "input", "{path}");
        }
    }

    #[tokio::test]
    async fn malformed_generation_json_is_a_typed_attributed_400() {
        for path in ["/v1/completions", "/v1/chat/completions"] {
            let response = attributed_no_model_router()
                .oneshot(
                    Request::post(path)
                        .header("content-type", "application/json")
                        .body(Body::from("{"))
                        .unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(response.status(), StatusCode::BAD_REQUEST, "{path}");
            assert_attribution(&response);
            let body = json(response).await;
            assert_eq!(body["error"]["type"], "invalid_request", "{path}");
            assert_eq!(body["error"]["code"], "malformed_json", "{path}");
            assert!(body["error"]["param"].is_null(), "{path}");
            assert_eq!(body["camelid_lane"], "deterministic", "{path}");
            assert_eq!(body["camelid_config_sha256"], &TEST_SHA[..12], "{path}");
        }
    }

    #[tokio::test]
    async fn head_and_cors_preflight_are_provided_by_the_pinned_router() {
        let app = attributed_no_model_router();
        let head = app
            .clone()
            .oneshot(Request::head("/v1/health").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(head.status(), StatusCode::OK);
        assert_attribution(&head);
        assert!(to_bytes(head.into_body(), 1024).await.unwrap().is_empty());

        let preflight = app
            .oneshot(
                Request::builder()
                    .method(Method::OPTIONS)
                    .uri("/v1/chat/completions")
                    .header("origin", "https://client.example")
                    .header("access-control-request-method", "POST")
                    .header("access-control-request-headers", "content-type")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(preflight.status(), StatusCode::OK);
        assert_eq!(preflight.headers()["access-control-allow-origin"], "*");
        assert_eq!(preflight.headers()["access-control-allow-methods"], "*");
        assert_eq!(preflight.headers()["access-control-allow-headers"], "*");
    }
}
