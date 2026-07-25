//! The surface a replica actually serves, and the one request it inspects.
//!
//! Two filters live here, and they now answer to different owners.
//!
//! # Which requests are served: the contract decides, not this module
//!
//! The served set is [`replica_contract::PUBLIC_ROUTES`] — the dependency-free
//! registry that `docs/contracts/replica-http-v1.md` publishes and that a gateway
//! can link without linking the engine. This module used to carry its own list of
//! admitted paths. That made two inventories of one public surface, and two
//! inventories are two things that can disagree: silently, at a distance, and in
//! the direction that costs an outage rather than a breach — a route the
//! published contract promises, refused by the replica that exists to serve it.
//! There is one list now, and this file reads it. Adding a route to the contract
//! is what adds it here; nothing else does.
//!
//! A filter is still needed, because the engine this distribution wraps is a
//! complete local inference application and its router is that application's
//! router: a control plane sits alongside the OpenAI-compatible API, all of it
//! unauthenticated, including routes that load and unload weights and flip
//! process-global execution flags. The contract names ten routes; the engine
//! registers sixty-one at this pin. Bound unfiltered, a replica's published
//! identity would describe what it *started* with rather than what it is serving:
//! the configuration digest, the model digest and the host summary all stay
//! byte-identical across a model swap performed over the very port they are
//! published on.
//!
//! Deriving the served set from the registry keeps the property that made an
//! allow list the right shape in the first place. A route a later engine revision
//! invents is not in the registry, so it arrives refused rather than arriving
//! served and waiting to be noticed — the pin bump is a contract review, not a
//! silent surface expansion.
//!
//! One consequence is worth stating because it reverses this module's earlier
//! behavior: `/v1/embeddings`, `/v1/responses`, `/v1/messages` and the two rerank
//! spellings are contractual. The promise on them is the engine's own typed
//! `501 not_implemented` answer, and a promise of an explicit refusal can only be
//! kept by letting the request reach the code that refuses it. They were answered
//! `403` here; they are served now.
//!
//! # This is not the trust boundary
//!
//! The boundary is the deployment's: traffic enters at the gateway, and
//! `deploy/k8s/replica-network-policy.yaml` is what keeps other workloads off the
//! replica's port. Everything in this module is defence in depth behind that. The
//! engine applies a permissive CORS policy *inside* this filter, so every route
//! admitted here is readable and postable from any web origin by anyone who can
//! reach the port at all. These filters bound *what* a caller may ask for, never
//! *who* may ask.
//!
//! # What this module still decides on its own: the generation body
//!
//! Withholding routes is necessary and is not sufficient, and the contract does
//! not close the gap. `/v1/chat/completions` is contractual and the gateway
//! forwards it, so a route registry and a network policy both leave it open — as
//! they should. But that request carries a `model` field, and the engine resolves
//! it against the filesystem before it resolves it against anything else: a
//! string that names an existing file is loaded on demand and becomes the
//! process's active model, for that request and for every later request that
//! names no model at all. Nothing a path-and-method filter can see, because the
//! request it arrives in is one the replica has to serve.
//!
//! So there is a second filter here, over the body of the two generation routes —
//! [`pin_generation_to_the_served_model`] — and it is what turns the model digest
//! on every response from a startup observation into a claim about the process
//! lifetime: the file this replica hashed before it bound its port is the file it
//! answers from until it stops.
//!
//! That second filter fails closed, and the reason is the one defect this design
//! is most likely to grow back. It has to *read* the body to check the field, so
//! it has a parser, and the engine behind it has its own; a filter that forwards
//! what its parser could not make sense of is only as strong as the agreement
//! between the two. There is no such agreement to rely on — the engine's JSON
//! extractor deserializes the first value in the buffer and never asks what
//! follows it, so a single trailing byte is a parse error here and an ordinary
//! request there. A generation body this filter cannot fully parse is therefore
//! refused rather than handed on.

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
use replica_contract::{HttpMethod, RouteSpec};
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

/// The contract's method vocabulary, as this stack spells methods.
///
/// Exhaustive on purpose. A method the registry grows later stops the build
/// here, which is the only place in this module where a contract change should
/// be able to require a decision — the alternative, a catch-all arm returning
/// something plausible, would decide it silently and in whichever direction the
/// arm happened to pick.
fn declared_as_http(method: HttpMethod) -> Method {
    match method {
        HttpMethod::Get => Method::GET,
        HttpMethod::Post => Method::POST,
        HttpMethod::Delete => Method::DELETE,
    }
}

/// Whether a contractual route admits `method`.
///
/// Two methods are admitted beyond the ones the registry declares, and neither
/// is a route the registry forgot — the contract says so in as many words:
/// "Axum also answers `HEAD` for registered `GET` routes. The pinned router
/// provides permissive CORS preflight […]. Neither behavior adds another
/// application route."
///
/// `HEAD` follows `GET`, because the engine's `get(...)` routes answer it, so a
/// probe or a gateway that issues one is making a request the replica can serve
/// and a filter that refused it would break a caller for no gain. It follows
/// `GET` and nothing else: `HEAD` on a generation route is not a cheap
/// completion, it is a request the engine has no handler for.
///
/// `OPTIONS` is admitted on every contractual path, for a sharper reason. The
/// engine applies its permissive CORS layer *inside* this filter, so the filter
/// sees a browser's preflight first. A client posting `Content-Type:
/// application/json` to a completion route cross-origin sends one, and refusing
/// it fails the preflight so the real request is never issued at all. Admitting
/// it concedes nothing: preflight for a path outside the contract is still
/// refused, and the write it precedes is refused on its own rule regardless.
fn method_is_admitted(route: &RouteSpec, method: &Method) -> bool {
    method == Method::OPTIONS
        || route.methods.iter().any(|declared| {
            let declared = declared_as_http(*declared);
            method == declared || (declared == Method::GET && method == Method::HEAD)
        })
}

/// Whether a request path matches a contractual path pattern.
///
/// Segment by segment, with `:name` matching exactly one non-empty segment —
/// the axum semantics the registry's patterns are written in, applied to the
/// same raw percent-encoded path axum's own router matches on, so the filter and
/// the router cannot disagree about where a segment ends.
///
/// "Exactly one segment" is the whole point and not pedantry. The engine's route
/// is `/v1/models/:model` and its router falls back to the embedded web UI, so a
/// looser prefix rule admits `/v1/models/a/b`, which misses the axum route,
/// reaches the fallback, and is answered by the application shell with HTTP 200 —
/// a served response from a surface this filter exists to withhold.
///
/// A pattern this matcher does not understand — an axum catch-all, say — matches
/// nothing and is therefore refused rather than over-admitted. Fail-closed is the
/// correct direction for a rule nobody has reviewed yet, and it is loud: the
/// contractual route would 403 and the conformance test in this module fails.
fn path_matches(pattern: &str, path: &str) -> bool {
    let mut pattern_segments = pattern.split('/');
    let mut path_segments = path.split('/');
    loop {
        match (pattern_segments.next(), path_segments.next()) {
            (None, None) => return true,
            (Some(expected), Some(actual)) => {
                let matched = match expected.strip_prefix(':') {
                    Some(_) => !actual.is_empty(),
                    None => expected == actual,
                };
                if !matched {
                    return false;
                }
            }
            // Different segment counts: `/v1/models/a/b` against
            // `/v1/models/:model`, and `/v1/models` against `/v1/models/:model`.
            _ => return false,
        }
    }
}

/// Whether `method path` is on the served surface.
///
/// The single dispatch point: the layer, and every test, decide through this and
/// not through a copy of its rules — and this is the only place the served set
/// is read, so there is nowhere else for a second inventory to accumulate.
///
/// Everything outside the contract is refused, including routes the engine
/// answers unauthenticated at this pin and that a reader might expect to find
/// admitted here:
///
///   * `/api/models/load`, `/api/models/unload`, `/api/models/inspect`, the
///     catalog and local-delete routes — the replica serves the model named on
///     its command line, and withholding these is what makes that true of the
///     whole process lifetime rather than of its first second;
///   * `/api/runtime/gpu` — flips a process-global accelerator flag under a lane
///     whose guarantee is stated over the CPU forward pass. It is also why method
///     and path are one rule rather than two: the engine registers that path as
///     `get(gpu_runtime).post(set_gpu_runtime)`, so a path-only filter cannot
///     tell reading the accelerator state from mutating it;
///   * the engine's legacy completion-server-compatible routes, which include a
///     second generation route the attribution middleware does not inject bodies
///     for: a way to get tokens out of this replica without the fields that say
///     what produced them;
///   * the agent-workspace family, telemetry streams, execution-plan,
///     capabilities, tokenizer and generation-session routes, and the embedded
///     web UI fallback — a serving replica is not an interactive application;
///   * bare `/health`. The contract classifies it as replica-private
///     diagnostics and `/v1/health` is the readiness route every deployment
///     artifact in this tree already polls, so serving it would be an eleventh
///     route outside the contract — exactly the second inventory this rewrite
///     removes, reintroduced for a probe nobody issues.
fn is_served(method: &Method, path: &str) -> bool {
    replica_contract::PUBLIC_ROUTES
        .iter()
        .any(|route| path_matches(route.path, path) && method_is_admitted(route, method))
}

/// The refusal, as a typed JSON error rather than a bare status.
///
/// A client that asked for a route this replica does not serve gets an
/// OpenAI-shaped error object, because that is what its error handling already
/// parses. 403 and not 404: the route exists on the engine's router and is
/// withheld, and a 404 sends an operator hunting a version mismatch that is not
/// there. The message names the contract, so the answer to "is this a bug or the
/// design?" is a document the reader can go and check rather than a judgement
/// they have to make.
fn refused(method: &Method, path: &str) -> Response {
    let body = serde_json::json!({
        "error": {
            "message": format!(
                "{method} {path} is not part of {}. This replica serves the published public \
                 route contract — health, model discovery and generation — and withholds the \
                 engine's model-management, runtime-control, workspace and legacy routes, \
                 because a replica that can be reconfigured over its serving port cannot vouch \
                 for what produced its output.",
                replica_contract::CONTRACT_ID
            ),
            "type": "invalid_request_error",
            "code": "route_not_served",
        }
    })
    .to_string();
    json_response(StatusCode::FORBIDDEN, body)
}

/// Refuse anything outside the published route contract.
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
            for part in [path.file_name(), path.file_stem()].into_iter().flatten() {
                aliases.insert(part.to_string_lossy().into_owned());
            }
        }
        // An empty name is not a name. A path with no components at all — what
        // an empty `--model` argument parses to — stringifies to `""`, and
        // admitting it would let a request carrying `"model": ""` through on an
        // accident of the path rather than on a rule. (It is the *path* that can
        // be empty, not its components: `file_name` and `file_stem` of a dotfile
        // or of a path with a trailing separator are non-empty, which is why the
        // test for this guard has to use an empty path to see it.)
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
/// order: after this middleware the field is either absent, `null`, or an exact
/// key in the engine's loaded-model map, so the branch that reads the filesystem
/// is unreachable from a request body. A name that is not this model's is
/// refused identically whether or not a file of that name exists — the same
/// status, the same code, the same message — so the field stops being a way to
/// ask the replica what is on its disk.
///
/// The parse is strict and a body that fails it is **refused**, which is the
/// part of this that is easy to get wrong and was wrong here. Both sides use the
/// same JSON library, and that is not the same as using the same parser: the
/// engine's extractor deserializes the first value in the buffer and never asks
/// whether anything follows it, while a whole-input parse rejects the leftovers.
/// So `{"model":"/some/other.gguf","prompt":"hi"}x` is a parse error here and an
/// ordinary, fully-honoured request there, and forwarding what this filter could
/// not parse handed that path straight to the resolution the filter exists to
/// keep it away from — one byte, and the model digest on every subsequent
/// response describes a file the replica is no longer answering from. Refusing
/// is also what makes the guarantee survive a pin bump: it holds however the
/// engine's extractor is spelled, rather than for as long as two parsers agree
/// about the end of the input.
///
/// It is a behavior change and not only a hardening, so it is stated plainly:
/// **a body carrying trailing content after the object is one the engine would
/// have accepted, and this replica now answers `400`.** That is the point rather
/// than a regrettable side effect — the trailing content is exactly what made
/// the `model` field uncheckable — and nothing that builds a request with a JSON
/// encoder produces one. The other bodies refused here, the ones that do not
/// parse at all and the valid JSON that is not an object, the engine would have
/// refused too; for those the difference is only which component says so, and
/// only this one can say it before the resolver runs.
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

    // Whole input, or nothing. `from_slice` consumes one JSON value and then
    // requires end-of-input, which is strictly more than the engine's extractor
    // asks for — and the gap between the two is the bypass this refusal closes:
    // a body the filter could not read used to be forwarded on the theory that
    // the engine could not read it either, and the engine reads it fine.
    let Ok(serde_json::Value::Object(mut object)) =
        serde_json::from_slice::<serde_json::Value>(&bytes)
    else {
        return uncheckable_request();
    };

    let rewrite_needed = match object.get("model") {
        // The common case, and the one the engine answers from the model it has
        // active — which, with the control plane withheld and this filter in
        // front, is the one the replica hashed.
        None | Some(serde_json::Value::Null) => false,
        Some(serde_json::Value::String(requested)) => {
            if !served.names_this_model(requested) {
                return not_this_replicas_model(&served);
            }
            // The engine's own key is already exact, so the body travels byte
            // for byte; anything else admitted is an alias and is rewritten.
            requested != &served.id
        }
        // A `model` that is not a string is not a name this replica answers to.
        // It cannot reach the filesystem resolution — that branch takes a
        // string — but it is refused rather than forwarded so that the invariant
        // after this middleware is one sentence with no exceptions in it: the
        // field is absent, `null`, or the engine's key.
        Some(_) => return not_this_replicas_model(&served),
    };

    let body = if rewrite_needed {
        object.insert("model".into(), served.id.as_str().into());
        match serde_json::to_vec(&serde_json::Value::Object(object)) {
            Ok(rewritten) => {
                // Only the length changed; the request is otherwise the one that
                // arrived.
                parts.headers.insert(
                    CONTENT_LENGTH,
                    HeaderValue::from_str(&rewritten.len().to_string())
                        .expect("a decimal length is a valid header value"),
                );
                Body::from(rewritten)
            }
            // Unreachable for a value that just parsed, and handled rather than
            // unwrapped. Forwarding the original bytes is *not* the safe
            // fallback here: the name admitted above may be a path spelling, and
            // unrewritten it would reach the resolution this filter exists to
            // keep it away from.
            Err(_) => return could_not_rewrite(),
        }
    } else {
        Body::from(bytes)
    };

    next.run(Request::from_parts(parts, body)).await
}

/// The two routes that carry a `model` field into the engine's resolver.
///
/// Deliberately *not* derived from the route contract, and the distinction is
/// the reason this module still exists. The contract classifies routes by what
/// this product promises a client; this asks what the pinned engine's handler
/// does with the request, which is a different question with a different answer
/// and a different reason to change. Both generation routes converge on one
/// engine function that resolves the request's `model` against the filesystem;
/// the other eight contractual routes do not reach it. `/v1/health`,
/// `/v1/models` and `/v1/models/:model` only read already-loaded state, and the
/// five compatibility routes are handled by functions that take no state and no
/// body at all — they answer their typed `501` and touch nothing.
///
/// A contract revision must therefore not silently extend this set, and a pin
/// bump must not silently leave it short. Both directions are re-reviewed by
/// hand against the engine's source, which is what
/// `only_the_generation_routes_are_inspected` pins from this side.
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

/// Refusal for a generation body this filter could not fully parse.
///
/// The fail-closed direction, and the one the size ceiling already takes: a body
/// this filter cannot read is a body whose `model` field it cannot check, and a
/// field it cannot check is a field it cannot vouch for. Forwarding it would put
/// the decision back in the hands of whichever parser runs next, which is
/// precisely the arrangement that let one trailing byte carry an arbitrary path
/// into the engine's resolver.
///
/// `400` and a request-shaped error code, because that is what it is: the body
/// is not a JSON object this replica can serve. Most bodies that land here the
/// engine would have called malformed as well; the one that would not is an
/// object with something stuck on the end of it, which is the case this refusal
/// exists for and is the one the message calls out by name.
fn uncheckable_request() -> Response {
    let body = serde_json::json!({
        "error": {
            "message": "a generation request body must be a single JSON object and nothing else. \
                        This replica checks the model field of every generation request before \
                        serving it, so a body it cannot parse in full is refused rather than \
                        forwarded unchecked — including one that carries trailing bytes after an \
                        otherwise valid object.",
            "type": "invalid_request_error",
            "code": "unparseable_request_body",
        }
    })
    .to_string();
    json_response(StatusCode::BAD_REQUEST, body)
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

    /// Every route the published contract promises is served, read from the
    /// registry rather than written out again.
    ///
    /// A copy of the list here would be a third inventory of the public surface,
    /// and it would hide precisely the failure that matters: a route the contract
    /// promises and this filter quietly withholds still passes a test that only
    /// ever asks about the routes its author already knew about.
    ///
    /// This is also the direction that costs a production outage rather than a
    /// breach. A contractual route missing from the served set is not a tightened
    /// surface, it is a probe that fails in production and nowhere else, and a
    /// rollout that never goes ready. The concrete callers, all in this tree: the
    /// container HEALTHCHECK and the Kubernetes startup, readiness and liveness
    /// probes read `/v1/health`; the gateway's readiness probe forwards to the
    /// replica; the drain sequence polls health; the README documents
    /// `/v1/chat/completions`, `/v1/completions` and `/v1/models`.
    #[test]
    fn every_contractual_route_is_served() {
        for route in replica_contract::PUBLIC_ROUTES {
            for declared in route.methods {
                let method = declared_as_http(*declared);
                assert!(
                    is_served(&method, route.probe_path),
                    "{method} {} is contractual and must be served",
                    route.path
                );
            }
        }
    }

    /// A path parameter is matched as a value, not as a prefix: a percent-encoded
    /// model id with spaces in it is one segment and is served.
    #[test]
    fn a_percent_encoded_model_id_is_one_segment() {
        assert!(is_served(&Method::GET, "/v1/models/Llama%203.2%201B%20Instruct"));
    }

    /// The five explicit-unsupported compatibility routes are contractual, so
    /// they are served.
    ///
    /// This assertion is the inverse of the one it replaces. These paths were
    /// refused here with `403`, which made a client SDK's capability probe read
    /// as "this replica is locked down" rather than as the engine's own typed
    /// "not implemented". The contract declares the `501` *itself* the promise —
    /// see the errors table in `docs/contracts/replica-http-v1.md` — and a
    /// promise of an explicit refusal can only be kept by letting the request
    /// reach the code that refuses it.
    #[test]
    fn the_compatibility_routes_are_served_rather_than_refused_here() {
        for path in [
            "/v1/embeddings",
            "/v1/responses",
            "/v1/messages",
            "/v1/rerank",
            "/v1/reranking",
        ] {
            assert!(
                is_served(&Method::POST, path),
                "POST {path} is contractual as an explicit unsupported answer, so the request \
                 has to reach the engine that gives it"
            );
        }
    }

    /// …and that the answer they reach really is the engine's own typed `501`,
    /// driven through the real layer against the real engine router.
    ///
    /// `is_served` returning true is not the property a client depends on. The
    /// property is that the request arrives at a handler and comes back with the
    /// documented code, which is a fact about this filter *and* the engine
    /// together and can only be asserted by running both.
    #[tokio::test]
    async fn the_compatibility_routes_answer_with_the_engines_own_not_implemented() {
        let served = camelid::api::router_with_state(camelid::api::AppState::default())
            .layer(axum::middleware::from_fn(serve_only_the_lane));
        for (path, code) in [
            ("/v1/embeddings", "unsupported_embeddings"),
            ("/v1/responses", "unsupported_responses"),
            ("/v1/messages", "unsupported_messages"),
            ("/v1/rerank", "unsupported_reranking"),
            ("/v1/reranking", "unsupported_reranking"),
        ] {
            let resp = served
                .clone()
                .oneshot(
                    Request::builder()
                        .method("POST")
                        .uri(path)
                        .header(CONTENT_TYPE, "application/json")
                        .body(Body::from("{}"))
                        .unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(resp.status(), StatusCode::NOT_IMPLEMENTED, "{path}");
            let bytes = to_bytes(resp.into_body(), 64 * 1024).await.unwrap();
            let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
            assert_eq!(json["error"]["type"], "not_implemented", "{path}");
            assert_eq!(json["error"]["code"], code, "{path}");
        }
    }

    /// Bare `/health` is not served, and it is named here because it used to be.
    ///
    /// The contract classifies it as replica-private diagnostics, nothing in this
    /// tree polls it — the container HEALTHCHECK, all three Kubernetes probes,
    /// the drain loop and the macOS installer all ask `/v1/health` — and keeping
    /// it would mean an eleventh served route living outside the registry, which
    /// is the second inventory this module was rewritten to remove. If a probe
    /// ever needs it, the fix is to add it to the contract, not to this file.
    #[test]
    fn the_engines_private_liveness_route_is_not_served() {
        for method in [Method::GET, Method::HEAD, Method::OPTIONS] {
            assert!(!is_served(&method, "/health"), "{method} /health must not be served");
        }
        assert!(is_served(&Method::GET, "/v1/health"), "the contractual spelling is served");
    }

    /// A probe or a gateway may issue `HEAD` where a client issues `GET`, and
    /// the engine's `get(...)` routes answer it. Refusing it would break a
    /// caller the replica can serve.
    #[test]
    fn head_is_served_wherever_get_is() {
        for path in ["/v1/health", "/v1/models", "/v1/models/x"] {
            assert!(is_served(&Method::HEAD, path), "HEAD {path} must be served");
        }
        // …but only where GET is. HEAD on a generation route is not a cheap
        // completion, it is a request the engine has no handler for.
        assert!(!is_served(&Method::HEAD, "/v1/chat/completions"));
        assert!(!is_served(&Method::HEAD, "/v1/embeddings"));
    }

    /// The engine's CORS layer sits behind this filter, so a preflight that is
    /// refused here fails before the layer that would have answered it — and the
    /// real request is then never sent. Every contractual path takes `OPTIONS`;
    /// nothing else does.
    #[test]
    fn preflight_is_served_on_contractual_paths_and_nowhere_else() {
        for route in replica_contract::PUBLIC_ROUTES {
            assert!(
                is_served(&Method::OPTIONS, route.probe_path),
                "OPTIONS {} must be served or a browser client cannot reach it",
                route.path
            );
        }
        assert!(
            !is_served(&Method::OPTIONS, "/api/models/load"),
            "preflighting a withheld route must not be a way to learn it is there"
        );
        assert!(!is_served(&Method::OPTIONS, "/health"));
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

    /// A `:param` in a contractual path claims exactly one segment and cannot
    /// reach past it.
    ///
    /// The multi-segment case is the one that matters and the one a looser rule
    /// gets wrong: `/v1/models/a/b` matches no engine route, so it falls through
    /// to the embedded web UI, which answers an extensionless unmatched path
    /// with the application shell and HTTP 200. Admitting it would serve a page
    /// from a surface this filter exists to withhold.
    #[test]
    fn a_path_parameter_matches_exactly_one_segment() {
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

    /// The property that survives a pin bump: a route the contract does not name
    /// is refused, so a route a later engine revision invents arrives refused
    /// rather than arriving served.
    ///
    /// This is the property that made an allow list the right shape before the
    /// registry existed, and it is unchanged by deriving the list from the
    /// registry — the registry moves when the contract is revised, not when the
    /// engine grows a route.
    #[test]
    fn an_unknown_route_is_refused_by_default() {
        assert!(!is_served(&Method::POST, "/api/some/route/a/later/pin/adds"));
        assert!(!is_served(&Method::GET, "/"));
        assert!(!is_served(&Method::GET, "/index.html"), "the web UI is not served");
        assert!(!is_served(&Method::POST, "/v2/chat/completions"));
        // A near-miss on a contractual path, which is where an over-eager
        // matcher would leak: same prefix, different route.
        assert!(!is_served(&Method::POST, "/v1/chat/completions/stream"));
    }

    /// The refusal has to be legible to a client's existing error handling, and
    /// has to say the route is withheld rather than absent — and now, which
    /// document decides that, so an operator can check the claim instead of
    /// taking the replica's word for it.
    #[tokio::test]
    async fn the_refusal_is_a_typed_json_error_naming_the_request_and_the_contract() {
        let resp = refused(&Method::POST, "/api/models/load");
        assert_eq!(resp.status(), StatusCode::FORBIDDEN);
        assert_eq!(resp.headers()[CONTENT_TYPE], "application/json");
        let bytes = to_bytes(resp.into_body(), 64 * 1024).await.unwrap();
        let body: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(body["error"]["code"], "route_not_served");
        assert_eq!(body["error"]["type"], "invalid_request_error");
        let message = body["error"]["message"].as_str().unwrap();
        assert!(message.contains("POST /api/models/load"), "unexpected message: {message}");
        assert!(
            message.contains(replica_contract::CONTRACT_ID),
            "the refusal must name the contract it is refusing against: {message}"
        );
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

    /// A path with no components at all must not smuggle an empty name onto the
    /// alias set, where `"model": ""` would match it on an accident of the path
    /// rather than on a rule.
    ///
    /// The input is the whole test. `Path::new("/models/")` and
    /// `Path::new(".gguf")` look like the cases that produce an empty component
    /// and do not — `file_name` and `file_stem` are `Some("models")` for the
    /// first and `Some(".gguf")` for the second — so a version of this test
    /// built on those passes with the guard it covers deleted. The empty path is
    /// the only input that puts `""` on the set, via `to_string_lossy`, and it
    /// is reachable: `--model ""` produces exactly it.
    #[test]
    fn a_path_with_no_components_does_not_admit_the_empty_name() {
        let empty = Path::new("");
        assert_eq!(empty.to_string_lossy(), "", "the input has to be able to produce an alias");
        let served = ServedModel::new("id".to_string(), empty, empty);
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
    ///
    /// The literal is chosen so that the round trip is *visible*: interior
    /// whitespace, keys in an order `serde_json` does not re-emit them in, and a
    /// float with a trailing zero. That is the whole difficulty of testing a
    /// pass-through — most JSON survives a round trip unchanged, and a literal
    /// that does cannot tell a body that was forwarded from one that was parsed
    /// and written back out. The final assertion pins that property of the
    /// literal itself, so this test cannot quietly stop being able to fail.
    #[tokio::test]
    async fn a_request_naming_the_engines_key_is_forwarded_unchanged() {
        let sent = format!(r#"{{"prompt":"hi", "model":"{MODEL_ID}", "temperature":0.10}}"#);
        let (status, body) = post_body("/v1/chat/completions", &sent).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body, sent, "an exact match must not be rewritten");

        let round_tripped =
            serde_json::to_string(&serde_json::from_str::<serde_json::Value>(&sent).unwrap())
                .unwrap();
        assert_ne!(
            round_tripped, sent,
            "the literal must be one a re-serialization would change, or the assertion above \
             holds whether or not the body was forwarded unchanged"
        );
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

    /// A body this filter cannot fully parse is refused, not forwarded.
    ///
    /// This test used to assert the opposite, on a premise that reads as
    /// obviously true and is false: that "a body that does not parse here does
    /// not parse there either — it is the same parser". Both sides are
    /// `serde_json`, and that is not the same as being the same parse. The
    /// engine's extractor deserializes the first value in the buffer and never
    /// asks what follows it, while the whole-input parse here rejects the
    /// leftovers — so the first entry below was a parse error here, an ordinary
    /// request there, and, forwarded, a `model` field going straight into the
    /// engine's filesystem resolution with the replica's published model digest
    /// still describing the file it had stopped answering from.
    ///
    /// The trailing bytes are varied because the ways to add one are: a stray
    /// character, a NUL, a second JSON value, a comment. None of them changes
    /// what the engine reads; all of them changed whether this filter could
    /// read it.
    #[tokio::test]
    async fn a_body_this_filter_cannot_fully_parse_is_refused_rather_than_forwarded() {
        for sent in [
            // Complete, valid, and then one byte more. The bypass.
            "{\"model\":\"/models/other.gguf\",\"prompt\":\"hi\"}x",
            "{\"model\":\"/models/other.gguf\",\"prompt\":\"hi\"}\0",
            "{\"model\":\"/models/other.gguf\",\"prompt\":\"hi\"}{}",
            "{\"model\":\"/models/other.gguf\",\"prompt\":\"hi\"}//x",
            "{\"model\":\"/models/other.gguf\",\"prompt\":\"hi\"}]",
            // Bodies that never parsed at all.
            "not json at all",
            "",
            "{\"model\":\"/models/other.gguf\",\"prompt\":\"hi\"",
            // Valid JSON that is not a generation request. It cannot carry the
            // field this filter checks, and "cannot carry it" is exactly the
            // reasoning that produced the bypass above — so it is refused on the
            // same rule rather than exempted by a second judgement call.
            r#"["an","array"]"#,
            r#""a bare string""#,
        ] {
            let (status, body) = post_body("/v1/chat/completions", sent).await;
            assert_eq!(
                status,
                StatusCode::BAD_REQUEST,
                "a body this filter cannot check must not be forwarded: {sent:?}"
            );
            assert_ne!(body, sent, "the echo means the body reached the engine: {sent:?}");
            let json: serde_json::Value = serde_json::from_str(&body).unwrap();
            assert_eq!(json["error"]["code"], "unparseable_request_body");
            assert_eq!(json["error"]["type"], "invalid_request_error");
        }
    }

    /// Trailing whitespace is not trailing content: a client that pretty-prints
    /// its request body, or a shell that adds a newline, is sending the same
    /// request and must be served.
    #[tokio::test]
    async fn trailing_whitespace_is_still_a_body_this_filter_can_check() {
        let sent = format!("{{\n  \"model\": \"{MODEL_ID}\",\n  \"prompt\": \"hi\"\n}}\n");
        let (status, body) = post_body("/v1/completions", &sent).await;
        assert_eq!(status, StatusCode::OK, "unexpected refusal: {body}");
        assert_eq!(body, sent);
    }

    /// A `model` that is not a string names nothing this replica serves, and is
    /// refused like any other name it does not answer to.
    ///
    /// It could not have reached the engine's path resolution — that branch
    /// takes a string — but "this value cannot reach the dangerous branch" is
    /// the reasoning that produced the trailing-byte bypass, and it is not the
    /// reasoning this filter runs on. The invariant it does run on has no
    /// exceptions in it: what reaches the engine names this replica's weights or
    /// names nothing.
    #[tokio::test]
    async fn a_model_field_that_is_not_a_string_is_refused() {
        for sent in [
            r#"{"model":42,"prompt":"hi"}"#,
            r#"{"model":["/models/other.gguf"],"prompt":"hi"}"#,
            r#"{"model":{"path":"/models/other.gguf"},"prompt":"hi"}"#,
            r#"{"model":true,"prompt":"hi"}"#,
        ] {
            let (status, body) = post_body("/v1/chat/completions", sent).await;
            assert_eq!(status, StatusCode::NOT_FOUND, "unexpected status for {sent}");
            let json: serde_json::Value = serde_json::from_str(&body).unwrap();
            assert_eq!(json["error"]["code"], "model_not_served");
        }
    }

    /// The property all of the arms above add up to, asserted once over the
    /// corpus rather than case by case: whatever reaches the engine from a
    /// generation route names this replica's own weights or names nothing at
    /// all. Anything else is refused before it gets there.
    ///
    /// Stated this way it survives the filter being rewritten. A test per arm
    /// pins the arms; this pins the guarantee, and a new arm that forwards
    /// something it should not fails here without anyone having to remember to
    /// add a case.
    #[tokio::test]
    async fn nothing_reaching_the_engine_names_weights_this_replica_did_not_hash() {
        let hostile = [
            r#"{"model":"/models/other.gguf","prompt":"hi"}"#,
            "{\"model\":\"/models/other.gguf\",\"prompt\":\"hi\"}x",
            "{\"model\":\"/models/other.gguf\",\"prompt\":\"hi\"}\0",
            "{\"model\":\"/models/other.gguf\",\"prompt\":\"hi\"}{}",
            r#"{"model":"","prompt":"hi"}"#,
            r#"{"model":42,"prompt":"hi"}"#,
            r#"{"model":"/models/other.gguf","model":"/models/third.gguf","prompt":"hi"}"#,
            "not json at all",
        ];
        let admitted = [
            r#"{"prompt":"hi"}"#,
            r#"{"model":null,"prompt":"hi"}"#,
            r#"{"model":"Llama-3.2-1B-Instruct-Q8_0","prompt":"hi"}"#,
            r#"{"model":"/models/Llama-3.2-1B-Instruct-Q8_0.gguf","prompt":"hi"}"#,
        ];

        let mut reached_the_engine = 0;
        for sent in hostile.iter().chain(admitted.iter()) {
            let (status, body) = post_body("/v1/completions", sent).await;
            if status != StatusCode::OK {
                continue;
            }
            reached_the_engine += 1;
            let json: serde_json::Value = serde_json::from_str(&body)
                .unwrap_or_else(|_| panic!("the engine was handed unparsed bytes: {body:?}"));
            match json.get("model") {
                None | Some(serde_json::Value::Null) => {}
                Some(serde_json::Value::String(model)) => assert_eq!(
                    model, MODEL_ID,
                    "the engine was handed a model field this replica does not serve: {body}"
                ),
                other => panic!("the engine was handed a non-string model field: {other:?}"),
            }
        }
        assert_eq!(
            reached_the_engine,
            admitted.len(),
            "every admitted request must still be served, or this passes by refusing everything"
        );
    }

    /// The body filter claims the two generation routes and nothing else. A
    /// `model` field on any other contractual route is not a model selector, and
    /// rewriting it would be this filter inventing semantics the engine does not
    /// have.
    ///
    /// The compatibility routes are the interesting entries, because they are the
    /// ones the contract added to the served set and they do accept a `model`
    /// field in the OpenAI shapes they imitate. They are still not inspected, and
    /// the reason is a fact about the pinned engine rather than a judgement call:
    /// their handlers take no state and no body, so there is no resolver behind
    /// them to keep a path away from. Inspecting them would buffer every such
    /// request to check a field that reaches nothing.
    #[tokio::test]
    async fn only_the_generation_routes_are_inspected() {
        assert!(is_generation_path("/v1/completions"));
        assert!(is_generation_path("/v1/chat/completions"));
        for path in [
            "/v1/models",
            "/v1/health",
            "/v1/models/x",
            "/v1/embeddings",
            "/v1/responses",
            "/v1/messages",
            "/v1/rerank",
            "/v1/reranking",
        ] {
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
