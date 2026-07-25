//! The surface a replica actually serves.
//!
//! The engine this distribution wraps is a complete local inference
//! application, and its router is that application's router: a control plane
//! sits alongside the OpenAI-compatible API, all of it unauthenticated,
//! including routes that load and unload weights and flip process-global
//! execution flags. Bound as-is, a replica's published identity describes what
//! it *started* with rather than what it is serving — the configuration digest,
//! the model digest and the host summary all stay byte-identical across a model
//! swap performed over the very port they are published on.
//!
//! So the served router is filtered to the requests a client of the
//! deterministic lane has business making, and everything else is refused. The
//! filter is an **allow list** for the same reason admission is: the engine's
//! router registers 61 routes at this pin and a later pin adds more, so a list
//! of paths to block is a list that goes stale silently, while a list of paths
//! to serve stays correct across a pin bump by refusing what it has never heard
//! of. A route a later revision invents arrives refused rather than arriving
//! served and waiting to be noticed.
//!
//! Withholding routes is necessary and is not sufficient, because one of the
//! routes that must stay is itself a weights-loading control. A generation
//! request carries a `model` field, and the engine resolves it against the
//! filesystem before it resolves it against anything else: a string that names
//! an existing file is loaded on demand and becomes the process's active model,
//! for that request and for every later request that names no model at all. A
//! path-and-method filter cannot see that, because the request it arrives in is
//! one the replica has to serve. So there is a second filter here, over the body
//! of the two admitted generation routes — [`pin_generation_to_the_served_model`]
//! — and the two together are what turn the model digest on every response from a
//! startup observation into a claim about the process lifetime: the file this
//! replica hashed before it bound its port is the file it answers from until it
//! stops.
//!
//! It is not an access-control layer and must not be read as one. The engine
//! applies a permissive CORS policy inside this filter, so every route admitted
//! here is still readable and postable from any web origin by anyone who can
//! reach the port. The filter bounds *what* a caller may ask for, never *who*
//! may ask; keeping the port private remains the deployment's job.

use axum::{
    body::{to_bytes, Body},
    extract::{Request, State},
    http::{
        header::{CONTENT_LENGTH, CONTENT_TYPE},
        HeaderValue, Method, StatusCode,
    },
    middleware::Next,
    response::Response,
};
use std::collections::BTreeSet;
use std::path::Path;
use std::sync::Arc;

/// Ceiling on a generation request body this filter will buffer.
///
/// The filter has to read the body to see the field it exists to check, so a
/// body it cannot hold is a body it cannot vouch for and is refused rather than
/// forwarded unchecked.
///
/// The number is the limit the engine's own JSON extractor applies, and it is
/// that number for a reason worth stating: buffering is the only new memory this
/// filter introduces, and a filter that held *more* than the engine would accept
/// would hand an unauthenticated caller a larger allocation than the surface it
/// was added to protect. Reading a body the engine is about to reject is work
/// nobody asked for; reading one it would have accepted is the job.
///
/// It is deliberately not the attribution middleware's ceiling. That one bounds
/// a *response* the engine produced, which is a different party and a different
/// risk, and matching them for symmetry would be matching the wrong thing.
/// `the_filter_never_buffers_more_than_the_engine_would_accept` is what keeps
/// this honest if the engine's extractor default ever moves.
pub const REQUEST_BODY_LIMIT: usize = 2 * 1024 * 1024;

/// How an admitted rule recognizes a path.
enum Route {
    /// The whole path, exactly.
    Exact(&'static str),
    /// The prefix plus exactly one more non-empty path segment.
    ///
    /// Used only where the engine's own route takes a single path parameter.
    /// "One segment" is the whole point and not pedantry: the engine's route is
    /// `/v1/models/:model`, which matches one segment, and its router falls back
    /// to the embedded web UI. A looser prefix admits `/v1/models/a/b`, which
    /// misses the axum route, reaches the fallback, and is answered by the app
    /// shell with HTTP 200 — a served response from a surface this filter exists
    /// to withhold.
    OneSegmentUnder(&'static str),
}

/// One admitted request shape: the methods, then the path rule.
struct Served {
    methods: &'static [Method],
    route: Route,
}

/// Read methods, and the preflight that has to precede a cross-origin write.
///
/// `HEAD` is here because the engine's `get(...)` routes answer it, so a probe
/// or a gateway that issues one is making a request the replica can serve; a
/// filter that refused it would break a caller for no gain.
///
/// `OPTIONS` is here for a sharper reason. The engine applies a permissive CORS
/// layer *inside* this filter, so the filter sees a browser's preflight first. A
/// client posting `Content-Type: application/json` to a completion route
/// cross-origin sends one, and refusing it fails the preflight so the real
/// request is never issued at all. Admitting it concedes nothing: preflight for
/// a withheld path is still refused, and the write it precedes is refused on its
/// own rule regardless.
const READ: &[Method] = &[Method::GET, Method::HEAD, Method::OPTIONS];

/// Generation, and its preflight.
const WRITE: &[Method] = &[Method::POST, Method::OPTIONS];

/// Every request this replica serves.
///
/// Method and path together, because the engine overloads paths by method:
/// `/api/runtime/gpu` is `get(gpu_runtime).post(set_gpu_runtime)` on one line of
/// its router, so a path-only rule cannot tell reading the accelerator state
/// from mutating it.
///
/// The list is what the deployment artifacts and the documented client surface
/// need, and nothing else. Notable absences, each deliberate and each of which
/// answers unauthenticated on the engine's router at this pin:
///
///   * `/api/models/load`, `/api/models/unload`, `/api/models/inspect`, the
///     catalog and local-delete routes — the replica serves the model named on
///     its command line, and withholding these is what makes that true of the
///     whole process lifetime rather than of its first second;
///   * `/api/runtime/gpu` — flips a process-global accelerator flag under a lane
///     whose guarantee is stated over the CPU forward pass;
///   * the engine's legacy completion-server-compatible routes, which include a
///     second generation route the attribution middleware does not inject bodies
///     for: a way to get tokens out of this replica without the fields that say
///     what produced them;
///   * the agent-workspace family, telemetry streams, execution-plan,
///     capabilities, tokenizer and generation-session routes, and the embedded
///     web UI fallback — a serving replica is not an interactive application.
///
/// Also refused, and worth stating because it is a behavior change rather than
/// an omission: the engine's typed "unsupported" replies on `/v1/embeddings`,
/// `/v1/responses`, `/v1/messages` and the rerank spellings become refusals
/// here. A client SDK probing those for capability detection now reads 403
/// rather than the engine's own 501-shaped answer. That is the consistent
/// outcome — this replica does not serve them — but it is a decision, not a side
/// effect of the list being short.
const SERVED: &[Served] = &[
    // Readiness and the drain probe. Both spellings, because the engine answers
    // both and the deployment documentation reaches for `/v1/health`. Neither is
    // used by an artifact in this tree today; they are admitted anyway because
    // this is the only endpoint that reports whether the replica can actually
    // generate, the drain sequence polls its queue depth, and adding a route to
    // this list later means re-touching the filter under time pressure.
    Served { methods: READ, route: Route::Exact("/health") },
    Served { methods: READ, route: Route::Exact("/v1/health") },
    // The model listing the container HEALTHCHECK, the three Kubernetes probes
    // and the gateway's readiness probe all read.
    Served { methods: READ, route: Route::Exact("/v1/models") },
    Served { methods: READ, route: Route::OneSegmentUnder("/v1/models/") },
    // Generation.
    Served { methods: WRITE, route: Route::Exact("/v1/completions") },
    Served { methods: WRITE, route: Route::Exact("/v1/chat/completions") },
];

/// Whether `method path` is on the served surface.
///
/// The single dispatch point: the layer, and every test, decide through this and
/// not through a copy of its rules.
fn is_served(method: &Method, path: &str) -> bool {
    SERVED.iter().any(|served| {
        served.methods.contains(method)
            && match served.route {
                Route::Exact(route) => path == route,
                Route::OneSegmentUnder(route) => path
                    .strip_prefix(route)
                    .is_some_and(|rest| !rest.is_empty() && !rest.contains('/')),
            }
    })
}

/// The refusal, as a typed JSON error rather than a bare status.
///
/// A client that asked for a route this replica does not serve gets an
/// OpenAI-shaped error object, because that is what its error handling already
/// parses. 403 and not 404: the route exists and is withheld, and a 404 sends an
/// operator hunting a version mismatch that is not there. The message says which
/// request was refused and why the surface is restricted, so the answer to "is
/// this a bug or the design?" is in the response body rather than in a document
/// the reader has to go and find.
fn refused(method: &Method, path: &str) -> Response {
    let body = serde_json::json!({
        "error": {
            "message": format!(
                "{method} {path} is not served by this replica. The deterministic lane serves \
                 generation, model listing and health; the engine's model-management and \
                 runtime-control routes are withheld, because a replica that can be \
                 reconfigured over its serving port cannot vouch for what produced its output."
            ),
            "type": "invalid_request_error",
            "code": "route_not_served",
        }
    })
    .to_string();
    json_response(StatusCode::FORBIDDEN, body)
}

/// Refuse anything outside the served surface.
///
/// Layered *inside* the attribution middleware, so a refusal still carries the
/// replica's full identity: a client that gets a 403 from a pool of
/// lane-attributed replicas has to be able to tell which one refused it, and
/// under what configuration and weights. A refusal is a response this replica
/// emitted, so it is attributed like any other.
///
/// Note for whoever reads a browser's console rather than this file: a 403 from
/// here carries no CORS headers, because the engine's CORS layer sits behind the
/// filter and never runs. A cross-origin caller therefore sees an opaque network
/// error rather than the typed message above; the message is in the response and
/// is visible to anything that is not a browser enforcing the same-origin
/// policy.
pub async fn serve_only_the_lane(req: Request, next: Next) -> Response {
    if !is_served(req.method(), req.uri().path()) {
        return refused(req.method(), req.uri().path());
    }
    next.run(req).await
}

/// The one model this replica answers from, and every name it answers to.
///
/// The engine keys its loaded models by a string it derives at load time — the
/// GGUF's `general.name`, or the file stem when that metadata is absent — so
/// this distribution does not get to pick it and must not guess it. It is read
/// back from the load's own reply, which is the only account of that key that
/// cannot drift from the engine's.
///
/// [`aliases`](Self::aliases) is every spelling of *this one file*: the engine's
/// key, the path as the operator wrote it, the path this replica canonicalized
/// and hashed, and the file's name and stem. Being generous there costs nothing,
/// because an admitted spelling is rewritten to the engine's key before the
/// request goes on — so no value from a request body ever reaches the engine's
/// path resolution, whatever it says.
pub struct ServedModel {
    /// The engine's own key for the loaded weights.
    id: String,
    /// Every accepted spelling, including `id`.
    aliases: BTreeSet<String>,
}

impl ServedModel {
    /// `id` as the engine reported it; `canonical` as this replica hashed it;
    /// `requested` as it arrived on the command line, which differs whenever a
    /// symlink or a relative path is in play and is the spelling an operator
    /// will reach for first.
    pub fn new(id: String, canonical: &Path, requested: &Path) -> Self {
        let mut aliases = BTreeSet::new();
        aliases.insert(id.clone());
        for path in [canonical, requested] {
            aliases.insert(path.to_string_lossy().into_owned());
            for part in [path.file_name(), path.file_stem()] {
                if let Some(part) = part {
                    aliases.insert(part.to_string_lossy().into_owned());
                }
            }
        }
        // An empty name is not a name. `file_stem` of a dotfile and a stray
        // trailing separator can both produce one, and admitting it would let a
        // request carrying `"model": ""` through on an accident of the path
        // rather than on a rule.
        aliases.remove("");
        Self { id, aliases }
    }

    fn names_this_model(&self, requested: &str) -> bool {
        self.aliases.contains(requested)
    }
}

/// Reject a generation request that asks to be answered by other weights.
///
/// Withholding the control plane is necessary and is not sufficient, because one
/// of the routes that has to stay is itself a weights-loading control. The
/// engine resolves a generation request's `model` field against the filesystem
/// before it resolves it against anything else — a string naming an existing
/// file is loaded on demand, becomes the process's active model, and answers
/// every later request that names no model at all. Nothing the replica publishes
/// moves when that happens: the digest, the host summary and the receipts all go
/// on describing the file hashed at startup. It is also unbounded, so a caller
/// naming a handful of files off the model mount can walk a container into its
/// memory limit over routes the path filter admits.
///
/// So the value is checked here, and the check is an allow list of names for one
/// file rather than a search for dangerous ones. An admitted name is **rewritten
/// to the engine's own key** before the request continues, which is what makes
/// the guarantee structural instead of a race against the engine's resolution
/// order: after this middleware the field is either absent or an exact key in
/// the engine's loaded-model map, so the branch that reads the filesystem is
/// unreachable from a request body. A name that is not this model's is refused
/// identically whether or not a file of that name exists — the same status, the
/// same code, the same message — so the field stops being a way to ask the
/// replica what is on its disk.
///
/// Layered *inside* [`serve_only_the_lane`], so only requests already admitted
/// by method and path get here, and inside attribution, so a refusal carries the
/// replica's identity like any other response.
pub async fn pin_generation_to_the_served_model(
    State(served): State<Arc<ServedModel>>,
    req: Request,
    next: Next,
) -> Response {
    if !(req.method() == Method::POST && is_generation_path(req.uri().path())) {
        return next.run(req).await;
    }

    let (mut parts, body) = req.into_parts();
    let Ok(bytes) = to_bytes(body, REQUEST_BODY_LIMIT).await else {
        return oversized_request();
    };

    // Anything this cannot parse, the engine cannot parse either — it is the
    // same parser — so there is no field to check and no swap to prevent.
    // Forward the original bytes and let the engine give its own typed answer
    // rather than inventing a second dialect of "malformed request".
    let body = match serde_json::from_slice::<serde_json::Value>(&bytes) {
        Ok(serde_json::Value::Object(mut object)) => match object.get("model") {
            // A non-string value fails the engine's own deserialization before
            // any resolution runs, so it is passed through to be refused there,
            // in the engine's vocabulary.
            None | Some(serde_json::Value::Null) => Body::from(bytes),
            Some(serde_json::Value::String(requested)) => {
                if !served.names_this_model(requested) {
                    return not_this_replicas_model(&served);
                }
                if requested == &served.id {
                    Body::from(bytes)
                } else {
                    object.insert("model".into(), served.id.as_str().into());
                    match serde_json::to_vec(&serde_json::Value::Object(object)) {
                        Ok(rewritten) => {
                            // Only the length changed; the request is otherwise
                            // the one that arrived.
                            parts.headers.insert(
                                CONTENT_LENGTH,
                                HeaderValue::from_str(&rewritten.len().to_string())
                                    .expect("a decimal length is a valid header value"),
                            );
                            Body::from(rewritten)
                        }
                        // Unreachable for a value that just parsed, and handled
                        // rather than unwrapped. Forwarding the original bytes
                        // is *not* the safe fallback here: the name admitted
                        // above may be a path spelling, and unrewritten it would
                        // reach the resolution this filter exists to keep it
                        // away from.
                        Err(_) => return could_not_rewrite(),
                    }
                }
            }
            Some(_) => Body::from(bytes),
        },
        _ => Body::from(bytes),
    };

    next.run(Request::from_parts(parts, body)).await
}

/// The two routes that carry a `model` field into the engine's resolver.
fn is_generation_path(path: &str) -> bool {
    matches!(path, "/v1/completions" | "/v1/chat/completions")
}

/// Refusal for a generation request naming weights this replica does not serve.
///
/// `404` and not `403`, unlike the route filter: the route is served, and what
/// is missing is the model. That is the answer an OpenAI-compatible client's
/// error handling already understands, and `model_not_served` says the replica
/// declined rather than lost it. The message names the model that *is* served,
/// because the fix is to ask for that one or to route to a different replica.
fn not_this_replicas_model(served: &ServedModel) -> Response {
    let body = serde_json::json!({
        "error": {
            "message": format!(
                "this replica serves one model, '{}', and answers only to that name. Its digest \
                 is published on every response, so a request that could repoint it to other \
                 weights would make that digest untrue; the model field is therefore checked \
                 rather than resolved.",
                served.id
            ),
            "type": "invalid_request_error",
            "param": "model",
            "code": "model_not_served",
        }
    })
    .to_string();
    json_response(StatusCode::NOT_FOUND, body)
}

/// Refusal for a checked request that could not be re-serialized.
///
/// Unreachable in practice — the value being written back is one `serde_json`
/// just parsed — but the failure is refused rather than swallowed. The name
/// admitted upstream may be a path spelling of this replica's own model, and
/// forwarding the original bytes unrewritten would hand that spelling to the
/// engine's resolver, which is the one thing this middleware exists to prevent.
/// Failing the request is the conservative loss; forwarding it is not.
fn could_not_rewrite() -> Response {
    let body = serde_json::json!({
        "error": {
            "message": "the request was accepted but could not be rewritten for the engine, and \
                        is refused rather than forwarded unchecked. Retry; if this repeats, it \
                        is a defect in this replica rather than in the request.",
            "type": "server_error",
            "code": "request_rewrite_failed",
        }
    })
    .to_string();
    json_response(StatusCode::INTERNAL_SERVER_ERROR, body)
}

/// Refusal for a generation request too large to check.
///
/// Fails closed: the filter cannot see the `model` field of a body it could not
/// buffer, and forwarding it unchecked would hand the engine's resolver exactly
/// the request this middleware exists to inspect.
fn oversized_request() -> Response {
    let body = serde_json::json!({
        "error": {
            "message": format!(
                "request body exceeds the {} MiB this replica will read. The model field of a \
                 generation request is checked before the request is served, so a body that \
                 cannot be read cannot be served — and a body this size would not have been \
                 accepted past this point either.",
                REQUEST_BODY_LIMIT / (1024 * 1024)
            ),
            "type": "invalid_request_error",
            "code": "request_too_large",
        }
    })
    .to_string();
    json_response(StatusCode::PAYLOAD_TOO_LARGE, body)
}

fn json_response(status: StatusCode, body: String) -> Response {
    let mut response = Response::new(Body::from(body));
    *response.status_mut() = status;
    response
        .headers_mut()
        .insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
    response
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::to_bytes;
    use axum::routing::{get, post};
    use axum::Router;
    use tower::ServiceExt;

    /// Every route a deployment artifact or the documented client surface
    /// actually uses. This is the test that costs a production outage if it is
    /// wrong in the other direction: a route missing from `SERVED` is not a
    /// tightened surface, it is a probe that fails in production and nowhere
    /// else, and a rollout that never goes ready.
    ///
    /// Sources, all in this tree: the container HEALTHCHECK and the Kubernetes
    /// startup, readiness and liveness probes read `/v1/models`; the gateway's
    /// readiness probe forwards to the same; the README documents
    /// `/v1/chat/completions`, `/v1/completions` and `/v1/models`; the drain
    /// sequence polls health.
    #[test]
    fn every_route_the_deployment_artifacts_and_clients_use_is_served() {
        for (method, path) in [
            (Method::GET, "/health"),
            (Method::GET, "/v1/health"),
            (Method::GET, "/v1/models"),
            (Method::GET, "/v1/models/Llama%203.2%201B%20Instruct"),
            (Method::POST, "/v1/completions"),
            (Method::POST, "/v1/chat/completions"),
        ] {
            assert!(is_served(&method, path), "{method} {path} must be served");
        }
    }

    /// A probe or a gateway may issue `HEAD` where a client issues `GET`, and
    /// the engine's `get(...)` routes answer it. Refusing it would break a
    /// caller the replica can serve.
    #[test]
    fn head_is_served_wherever_get_is() {
        for path in ["/health", "/v1/health", "/v1/models", "/v1/models/x"] {
            assert!(is_served(&Method::HEAD, path), "HEAD {path} must be served");
        }
        // …but only where GET is. HEAD on a generation route is not a cheap
        // completion, it is a request the engine has no handler for.
        assert!(!is_served(&Method::HEAD, "/v1/chat/completions"));
    }

    /// The engine's CORS layer sits behind this filter, so a preflight that is
    /// refused here fails before the layer that would have answered it — and the
    /// real request is then never sent. Every admitted path takes `OPTIONS`;
    /// nothing else does.
    #[test]
    fn preflight_is_served_on_admitted_paths_and_nowhere_else() {
        for path in [
            "/health",
            "/v1/health",
            "/v1/models",
            "/v1/models/x",
            "/v1/completions",
            "/v1/chat/completions",
        ] {
            assert!(
                is_served(&Method::OPTIONS, path),
                "OPTIONS {path} must be served or a browser client cannot reach it"
            );
        }
        assert!(
            !is_served(&Method::OPTIONS, "/api/models/load"),
            "preflighting a withheld route must not be a way to learn it is there"
        );
    }

    /// The routes that make a replica's published identity untrue. Each answers
    /// on the engine's router at this pin, unauthenticated, and each was
    /// demonstrated to change what a live replica serves while every published
    /// header stayed identical across the change.
    #[test]
    fn the_control_plane_is_not_served() {
        for (method, path) in [
            (Method::POST, "/api/models/load"),
            (Method::POST, "/api/models/unload"),
            (Method::POST, "/api/models/inspect"),
            (Method::POST, "/api/models/catalog/install"),
            (Method::POST, "/api/models/local/delete"),
            (Method::POST, "/api/runtime/gpu"),
            (Method::GET, "/api/runtime/gpu"),
            (Method::POST, "/models/load"),
            (Method::POST, "/models/unload"),
            (Method::GET, "/api/telemetry/stream"),
            (Method::GET, "/api/execution-plan"),
            (Method::GET, "/api/agent/workspace/threads"),
            (Method::POST, "/api/agent/workspace/sessions"),
        ] {
            assert!(
                !is_served(&method, path),
                "{method} {path} lets a client change or inspect what this replica serves while \
                 it keeps publishing the same identity"
            );
        }
    }

    /// A second generation route that the attribution middleware does not inject
    /// bodies for is a way to get tokens out of this replica without the fields
    /// that say what produced them. The whole legacy completion-server surface
    /// stays off.
    #[test]
    fn no_unattributed_generation_route_is_served() {
        for path in ["/completion", "/infill", "/tokenize", "/detokenize", "/apply-template"] {
            assert!(!is_served(&Method::POST, path), "POST {path} must not be served");
        }
        assert!(!is_served(&Method::GET, "/models"), "the legacy model listing is not ours");
        assert!(!is_served(&Method::GET, "/props"));
        assert!(!is_served(&Method::GET, "/slots"));
    }

    /// Method and path are one rule, because the engine overloads paths by
    /// method. A path-only filter would admit the mutation along with the read.
    #[test]
    fn a_route_is_admitted_by_method_as_well_as_path() {
        assert!(is_served(&Method::GET, "/v1/models"));
        assert!(!is_served(&Method::POST, "/v1/models"));
        assert!(!is_served(&Method::DELETE, "/v1/models"));
        assert!(!is_served(&Method::PUT, "/v1/models"));
        assert!(!is_served(&Method::GET, "/v1/chat/completions"));
    }

    /// The one prefix rule claims exactly the namespace of the engine's
    /// single-path-parameter route and cannot reach past it.
    ///
    /// The multi-segment case is the one that matters and the one a looser rule
    /// gets wrong: `/v1/models/a/b` matches no engine route, so it falls through
    /// to the embedded web UI, which answers an extensionless unmatched path
    /// with the application shell and HTTP 200. Admitting it would serve a page
    /// from a surface this filter exists to withhold.
    #[test]
    fn the_prefix_rule_admits_exactly_one_segment() {
        assert!(is_served(&Method::GET, "/v1/models/anything"));
        assert!(
            !is_served(&Method::GET, "/v1/models/a/b"),
            "a second segment misses the engine's route and reaches the web-UI fallback"
        );
        assert!(
            !is_served(&Method::GET, "/v1/models/a/"),
            "a trailing slash is a second (empty) segment and misses the route too"
        );
        assert!(!is_served(&Method::GET, "/v1/models/"), "the bare prefix names no model");
        assert!(!is_served(&Method::GET, "/v1/modelsomething"));
        assert!(!is_served(&Method::GET, "/v1/models-secret"));
    }

    /// The property that survives a pin bump: a route nobody wrote a rule for is
    /// refused, so new engine routes arrive refused rather than arriving served.
    #[test]
    fn an_unknown_route_is_refused_by_default() {
        assert!(!is_served(&Method::POST, "/api/some/route/a/later/pin/adds"));
        assert!(!is_served(&Method::GET, "/"));
        assert!(!is_served(&Method::GET, "/index.html"), "the web UI is not served");
        assert!(!is_served(&Method::POST, "/v1/embeddings"));
    }

    /// The refusal has to be legible to a client's existing error handling, and
    /// has to say the route is withheld rather than absent.
    #[tokio::test]
    async fn the_refusal_is_a_typed_json_error_naming_the_request() {
        let resp = refused(&Method::POST, "/api/models/load");
        assert_eq!(resp.status(), StatusCode::FORBIDDEN);
        assert_eq!(resp.headers()[CONTENT_TYPE], "application/json");
        let bytes = to_bytes(resp.into_body(), 64 * 1024).await.unwrap();
        let body: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(body["error"]["code"], "route_not_served");
        assert_eq!(body["error"]["type"], "invalid_request_error");
        let message = body["error"]["message"].as_str().unwrap();
        assert!(message.contains("POST /api/models/load"), "unexpected message: {message}");
    }

    /// End to end through the real layer: the refusal happens before the handler
    /// runs, and a served route passes through untouched.
    ///
    /// The withheld handlers panic rather than return, so "refused" here means
    /// the request never reached them — not that it reached them and their reply
    /// was discarded.
    #[tokio::test]
    async fn the_layer_refuses_before_the_handler_runs() {
        async fn never_reached() -> &'static str {
            panic!("a withheld handler must never be reached")
        }

        let router = Router::new()
            .route("/v1/models", get(|| async { "served" }))
            .route("/v1/chat/completions", post(|| async { "served" }))
            .route("/api/models/load", post(never_reached))
            .route("/api/models/unload", post(never_reached))
            .route("/api/runtime/gpu", get(never_reached).post(never_reached))
            .route("/completion", post(never_reached))
            .layer(axum::middleware::from_fn(serve_only_the_lane));

        for (method, path) in [
            ("POST", "/api/models/load"),
            ("POST", "/api/models/unload"),
            ("POST", "/api/runtime/gpu"),
            ("GET", "/api/runtime/gpu"),
            ("POST", "/completion"),
        ] {
            let resp = router
                .clone()
                .oneshot(
                    Request::builder().method(method).uri(path).body(Body::empty()).unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(
                resp.status(),
                StatusCode::FORBIDDEN,
                "{method} {path} must be refused by the layer"
            );
            let bytes = to_bytes(resp.into_body(), 64 * 1024).await.unwrap();
            let body: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
            assert_eq!(body["error"]["code"], "route_not_served");
        }

        for (method, path) in [("GET", "/v1/models"), ("POST", "/v1/chat/completions")] {
            let resp = router
                .clone()
                .oneshot(
                    Request::builder().method(method).uri(path).body(Body::empty()).unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(resp.status(), StatusCode::OK, "{method} {path} must pass through");
        }
    }

    // ---- the body filter ----

    const MODEL_ID: &str = "Llama 3.2 1B Instruct";
    const MODEL_PATH: &str = "/models/Llama-3.2-1B-Instruct-Q8_0.gguf";

    fn served_model() -> Arc<ServedModel> {
        Arc::new(ServedModel::new(
            MODEL_ID.to_string(),
            Path::new(MODEL_PATH),
            Path::new("models/../models/Llama-3.2-1B-Instruct-Q8_0.gguf"),
        ))
    }

    /// A router that answers with whatever body reached it, so a test can read
    /// the request the engine would have seen rather than infer it.
    fn echoing_router() -> Router {
        Router::new()
            .route("/v1/chat/completions", post(echo))
            .route("/v1/completions", post(echo))
            .route("/v1/models", get(|| async { "listing" }))
            .layer(axum::middleware::from_fn_with_state(
                served_model(),
                pin_generation_to_the_served_model,
            ))
    }

    async fn echo(body: axum::body::Bytes) -> Vec<u8> {
        body.to_vec()
    }

    async fn post_body(path: &str, body: &str) -> (StatusCode, String) {
        let resp = echoing_router()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri(path)
                    .header(CONTENT_TYPE, "application/json")
                    .header(CONTENT_LENGTH, body.len().to_string())
                    .body(Body::from(body.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();
        let status = resp.status();
        let bytes = to_bytes(resp.into_body(), REQUEST_BODY_LIMIT).await.unwrap();
        (status, String::from_utf8(bytes.to_vec()).unwrap())
    }

    /// Every spelling of the one file is a name this replica answers to, and
    /// nothing else is. The set is generous on purpose — a filter that refuses
    /// the path an operator started the replica with is an outage — and generous
    /// costs nothing because an admitted name is rewritten before it travels.
    #[test]
    fn the_alias_set_is_every_spelling_of_one_file_and_no_other() {
        let served = served_model();
        for name in [
            MODEL_ID,
            MODEL_PATH,
            "Llama-3.2-1B-Instruct-Q8_0.gguf",
            "Llama-3.2-1B-Instruct-Q8_0",
            "models/../models/Llama-3.2-1B-Instruct-Q8_0.gguf",
        ] {
            assert!(served.names_this_model(name), "{name} names the served model");
        }
        for name in [
            "",
            "/models/other.gguf",
            "other.gguf",
            "gpt-4",
            "/models",
            "Llama-3.2-1B-Instruct-Q8_0.gguf ",
        ] {
            assert!(!served.names_this_model(name), "{name} must not name the served model");
        }
    }

    /// An empty `file_stem` — a dotfile, a trailing separator — must not smuggle
    /// an empty name onto the alias set, where `"model": ""` would match it on
    /// an accident of the path rather than on a rule.
    #[test]
    fn a_path_that_yields_an_empty_component_does_not_admit_the_empty_name() {
        let served =
            ServedModel::new("id".to_string(), Path::new("/models/"), Path::new(".gguf"));
        assert!(!served.names_this_model(""));
        assert!(served.names_this_model("id"));
    }

    /// The blocker: a `model` naming another file is refused before the engine
    /// sees the request, on both generation routes.
    #[tokio::test]
    async fn a_model_field_naming_other_weights_is_refused() {
        for path in ["/v1/chat/completions", "/v1/completions"] {
            let (status, body) =
                post_body(path, r#"{"model":"/models/other.gguf","prompt":"hi"}"#).await;
            assert_eq!(status, StatusCode::NOT_FOUND, "{path} admitted other weights");
            let json: serde_json::Value = serde_json::from_str(&body).unwrap();
            assert_eq!(json["error"]["code"], "model_not_served");
            assert_eq!(json["error"]["param"], "model");
            assert!(
                json["error"]["message"].as_str().unwrap().contains(MODEL_ID),
                "the refusal must name the model that is served: {body}"
            );
        }
    }

    /// Refused identically whether or not the named file exists, so the field
    /// cannot be used to probe the replica's filesystem. The two calls differ
    /// only in that one path is real.
    #[tokio::test]
    async fn an_existing_and_a_missing_path_are_refused_with_the_same_bytes() {
        let real = std::env::current_dir().unwrap().join("Cargo.toml");
        assert!(real.exists(), "the test needs a path that really is there");
        let existing = post_body(
            "/v1/completions",
            &serde_json::json!({ "model": real.to_string_lossy(), "prompt": "hi" }).to_string(),
        )
        .await;
        let missing = post_body(
            "/v1/completions",
            r#"{"model":"/no/such/path/absent.gguf","prompt":"hi"}"#,
        )
        .await;
        assert_eq!(existing, missing);
    }

    /// An admitted alias is rewritten to the engine's own key, which is what
    /// makes the guarantee structural: after this filter the field is either
    /// absent or an exact key in the engine's loaded-model map, so the branch
    /// that resolves a request string against the filesystem is unreachable.
    #[tokio::test]
    async fn an_admitted_alias_is_rewritten_to_the_engines_key() {
        let (status, body) = post_body(
            "/v1/chat/completions",
            r#"{"model":"Llama-3.2-1B-Instruct-Q8_0.gguf","prompt":"hi","max_tokens":4}"#,
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        let json: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert_eq!(json["model"], MODEL_ID, "the alias should have been rewritten: {body}");
        // …and nothing else about the request moved.
        assert_eq!(json["prompt"], "hi");
        assert_eq!(json["max_tokens"], 4);
    }

    /// The engine's own key is already exact, so the body travels byte for byte.
    /// Re-serializing a request nobody needed to change is a JSON round-trip
    /// this filter has no reason to impose.
    #[tokio::test]
    async fn a_request_naming_the_engines_key_is_forwarded_unchanged() {
        let sent = format!(r#"{{"model":"{MODEL_ID}","prompt":"hi","temperature":1e-7}}"#);
        let (status, body) = post_body("/v1/chat/completions", &sent).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body, sent, "an exact match must not be rewritten");
    }

    /// The common case: no `model` field at all. Forwarded byte for byte, and
    /// answered by the engine from the model it has active — which, with the
    /// control plane withheld and this filter in front, is the one the replica
    /// hashed.
    #[tokio::test]
    async fn a_request_naming_no_model_is_forwarded_unchanged() {
        for sent in [
            r#"{"prompt":"hi","max_tokens":4}"#,
            r#"{"model":null,"prompt":"hi"}"#,
        ] {
            let (status, body) = post_body("/v1/completions", sent).await;
            assert_eq!(status, StatusCode::OK);
            assert_eq!(body, sent);
        }
    }

    /// Bodies this filter cannot make sense of are the engine's to reject, in
    /// the engine's vocabulary. None of them can carry a model name past it: a
    /// body that does not parse here does not parse there either — it is the
    /// same parser — and a non-string `model` fails the engine's own
    /// deserialization before any resolution runs.
    #[tokio::test]
    async fn bodies_this_filter_cannot_check_are_forwarded_for_the_engine_to_reject() {
        for sent in [
            "not json at all",
            "",
            r#"["an","array"]"#,
            r#""a bare string""#,
            r#"{"model":42,"prompt":"hi"}"#,
            r#"{"model":["/models/other.gguf"],"prompt":"hi"}"#,
        ] {
            let (status, body) = post_body("/v1/chat/completions", sent).await;
            assert_eq!(status, StatusCode::OK, "unexpected refusal for {sent}");
            assert_eq!(body, sent, "the body must reach the engine unchanged");
        }
    }

    /// The filter claims the two generation routes and nothing else. A `model`
    /// field on any other admitted route is not a model selector, and rewriting
    /// it would be this filter inventing semantics the engine does not have.
    #[tokio::test]
    async fn only_the_generation_routes_are_inspected() {
        assert!(is_generation_path("/v1/completions"));
        assert!(is_generation_path("/v1/chat/completions"));
        for path in ["/v1/models", "/v1/health", "/health", "/v1/models/x"] {
            assert!(!is_generation_path(path), "{path} is not a generation route");
        }

        let resp = echoing_router()
            .oneshot(Request::builder().uri("/v1/models").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
    }

    /// A body too large to buffer is a body whose `model` field cannot be seen,
    /// so it is refused rather than forwarded unchecked — the fail-closed
    /// direction, and the same one the attribution middleware takes on an
    /// oversized response.
    #[tokio::test]
    async fn a_body_too_large_to_check_is_refused_rather_than_forwarded() {
        let resp = echoing_router()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/completions")
                    .header(CONTENT_TYPE, "application/json")
                    .body(Body::from(vec![b'x'; REQUEST_BODY_LIMIT + 1]))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::PAYLOAD_TOO_LARGE);
        let bytes = to_bytes(resp.into_body(), 64 * 1024).await.unwrap();
        let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(json["error"]["code"], "request_too_large");
    }

    /// A rewrite changes the body's length, so the header that declares it has
    /// to move with it. A stale `Content-Length` is the kind of defect that
    /// surfaces as a truncated prompt rather than as an error.
    #[tokio::test]
    async fn a_rewrite_updates_the_declared_content_length() {
        async fn declared_length(sent: &str) -> usize {
            // Read the length the inner service sees by echoing the body back:
            // the echo is exactly the bytes the engine would have parsed.
            let (_, body) = post_body("/v1/completions", sent).await;
            body.len()
        }

        // The alias is shorter than the engine's key here, so a body forwarded
        // with its original length would be truncated.
        let sent = r#"{"model":"pinned","prompt":"hi"}"#;
        let served = Arc::new(ServedModel::new(
            "a considerably longer model identity".to_string(),
            Path::new("/models/pinned.gguf"),
            Path::new("/models/pinned.gguf"),
        ));
        let resp = Router::new()
            .route(
                "/v1/completions",
                post(|req: Request| async move {
                    let declared: usize = req
                        .headers()
                        .get(CONTENT_LENGTH)
                        .expect("the filter must declare a length it rewrote")
                        .to_str()
                        .unwrap()
                        .parse()
                        .unwrap();
                    let bytes = to_bytes(req.into_body(), REQUEST_BODY_LIMIT).await.unwrap();
                    assert_eq!(declared, bytes.len(), "declared length must match the body");
                    String::from_utf8(bytes.to_vec()).unwrap()
                }),
            )
            .layer(axum::middleware::from_fn_with_state(
                served,
                pin_generation_to_the_served_model,
            ))
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/completions")
                    .header(CONTENT_TYPE, "application/json")
                    .header(CONTENT_LENGTH, sent.len().to_string())
                    .body(Body::from(sent))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let bytes = to_bytes(resp.into_body(), REQUEST_BODY_LIMIT).await.unwrap();
        let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(json["model"], "a considerably longer model identity");

        // The unrewritten path keeps whatever length it arrived with, which is
        // the one it still has.
        assert_eq!(declared_length(r#"{"prompt":"hi"}"#).await, 15);
    }
}
