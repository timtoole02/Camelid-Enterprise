//! camelid-enterprise — multi-tenant serving distribution of the Camelid engine.
//!
//! This release ships the **deterministic lane**: whole-generation serialized
//! execution on the engine at a pinned revision under a frozen configuration
//! vector, with deterministic greedy output run to run. Requests the
//! replica cannot serve fail closed (typed 503 from the bounded engine queue);
//! there is no silent demotion to any other execution mode.

// What a replica *is* — the parity baseline, the lane's configuration vector, the
// attribution stamped on every response, the startup load and the served route
// surface — lives in this crate's library target, so the contract tests and the
// conformance harness are written against the same code a served replica is
// built from rather than a restatement of it. This binary is the process around
// that: the CLI, the host probe, the worker pool, the listener and shutdown. It
// composes no part of the served stack itself; `replica_router` is the one place
// that happens, which is what keeps the surface a client meets and the surface
// the tests drive from being two different things.
use camelid_enterprise::{
    apply_deterministic, replica_router, Attribution, ModelIdentity, WorkerThreads, ENGINE_PIN,
};
use clap::{Parser, Subcommand};
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;

#[derive(Parser)]
#[command(name = "camelid-enterprise", version, about)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Start a lane-attributed serving replica.
    Serve {
        /// GGUF model to load at startup.
        #[arg(long, env = "CAMELID_ENTERPRISE_MODEL")]
        model: PathBuf,
        /// Bind address.
        #[arg(long, default_value = "127.0.0.1:8181", env = "CAMELID_ENTERPRISE_ADDR")]
        addr: SocketAddr,
        /// Serving lane for this replica. Lane selection is per-deployment.
        #[arg(long, default_value = "deterministic")]
        lane: String,
        /// Worker threads. Sizes this replica's global data-parallel pool, and
        /// the width it actually resolves to is published on every response.
        ///
        /// Bound to `CAMELID_ENTERPRISE_THREADS` and not `CAMELID_THREADS`. The
        /// pinned engine reads that second name for its own purposes — and reads
        /// it for *presence*, not value, so setting it changes the engine's
        /// phase-threading shape rather than any thread count. A name two
        /// products answer to differently is exactly the confusion the startup
        /// environment scan exists to refuse, and it refuses this one: the
        /// binary's own documented flag must not be a variable its own scan
        /// rejects. Every other variable this binary consumes is already
        /// `CAMELID_ENTERPRISE_*`; this was the one outlier.
        #[arg(long, env = "CAMELID_ENTERPRISE_THREADS")]
        threads: Option<usize>,
        /// Append one JSONL serving receipt per request to this file.
        #[arg(long)]
        serving_receipts: Option<PathBuf>,
    },
}

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();
    let cli = Cli::parse();
    let result = match cli.command {
        Command::Serve { model, addr, lane, threads, serving_receipts } => {
            serve(model, addr, &lane, threads, serving_receipts).await
        }
    };
    // Printed with `Display`, and handled here rather than returned from `main`.
    // Returning a `Box<dyn Error>` renders it with `Debug`, which turns the
    // startup refusals — the environment scan's, in particular — into one line
    // of escaped `\n` literals wrapped in `Custom { kind: Other, … }`. Those
    // messages are the operator's entire diagnostic: every offending variable,
    // what each one does, and the fix. They have to arrive readable.
    if let Err(err) = result {
        eprintln!("{err}");
        std::process::exit(1);
    }
}

async fn serve(
    model: PathBuf,
    addr: SocketAddr,
    lane_name: &str,
    threads: Option<usize>,
    serving_receipts: Option<PathBuf>,
) -> Result<(), Box<dyn std::error::Error>> {
    if lane_name != "deterministic" {
        return Err(format!(
            "lane '{lane_name}' is not available in this release; this replica serves \
             the deterministic lane only"
        )
        .into());
    }
    let config = apply_deterministic().map_err(std::io::Error::other)?;
    #[cfg(target_os = "macos")]
    let host = engine_macos::probe();
    #[cfg(target_os = "linux")]
    let host = engine_linux::probe();
    #[cfg(target_os = "windows")]
    let host = engine_windows::probe();
    let host_summary = host.summary();

    // Before the engine state, and before the listener: everything this replica
    // publishes about itself has to be settled before it can answer anything.
    //
    // The pool is sized here, immediately after the environment freeze and
    // before anything else can touch rayon, and the order is load-bearing in
    // both directions. After the freeze, because the pool reads its own
    // environment as it is built and must not race the writer. Before the engine
    // state, because constructing that state spawns the engine's handle, and the
    // first thing to read the pool's width fixes it — a sizing call after that
    // point fails rather than resizes.
    let workers = WorkerThreads::resolved(resolve_worker_pool(threads)?);

    // Identifying the weights is a full read of a multi-gigabyte file, and it
    // happens here — before the port is bound — on purpose. A replica that
    // cannot say which weights produced a token has no attribution worth the
    // name, so there must be no window in which it is answering requests and
    // does not yet know. Failing here also fails before a port is claimed,
    // rather than leaving a half-started replica holding one. The read warms the
    // page cache the load is about to fault through, so the cost is smaller than
    // the wall clock suggests.
    let requested_model = model;
    let model = requested_model.canonicalize()?;
    let model_identity = ModelIdentity::of_file(&model).map_err(std::io::Error::other)?;

    eprintln!(
        "[lane] deterministic | in-tree engine | parity oracle pin {} | config vector sha256 {} | admission sha256 {} | \
         model sha256 {} | host {} | worker threads {}",
        ENGINE_PIN,
        config.short(),
        config.admission_short(),
        model_identity.short(),
        host_summary,
        workers.count()
    );
    eprintln!("[lane] model {}", model.display());

    // Everything this replica publishes about itself, settled before anything
    // can answer a request. It is handed to `replica_router` rather than applied
    // here: the served view needs two further layers, one of which cannot be
    // written until the model is loaded, and both have to sit *inside*
    // attribution so that a request this replica refuses is still a response
    // that says which replica refused it.
    let ctx = Attribution {
        lane: "deterministic",
        config_sha256: Arc::new(config.sha256),
        admission_sha256: Arc::new(config.admission_sha256),
        model: model_identity,
        host: Arc::new(host_summary),
        workers,
        receipts: serving_receipts.map(Arc::new),
    };

    // Bound before the load so an address collision fails in milliseconds rather
    // than after a multi-gigabyte read. Nothing is served from it until the load
    // returns: connections sit in the listen backlog, so a probe that connects
    // hangs to its own timeout and fails, which is the correct reading of a
    // replica that is not ready.
    let listener = tokio::net::TcpListener::bind(addr).await?;
    eprintln!("[lane] listening on http://{addr}");
    eprintln!("[lane] loading model; nothing is served until the load completes");

    // Platform crates may supply an accelerated kernel only through a seam that
    // is proven bit-identical to the portable implementation. Model ownership
    // remains in-tree and in-process; no load/unload control-plane router exists.
    #[cfg(target_os = "macos")]
    let loaded_model = engine_core::runtime::LoadedModel::load_with_q8_dot(
        &model,
        engine_macos::q8_0_dot_rows,
    )?;
    #[cfg(not(target_os = "macos"))]
    let loaded_model = engine_core::runtime::LoadedModel::load(&model)?;

    // Re-verify the digest and compose in the library, so the stack this process
    // serves is exactly the stack the model-backed contract drives.
    let (served, model_id) = replica_router(loaded_model, &requested_model, ctx)?;
    eprintln!("[lane] model loaded as '{model_id}'; replica ready");

    axum::serve(listener, served)
        .with_graceful_shutdown(shutdown_signal())
        .await?;
    Ok(())
}

/// Size the data-parallel worker pool, and report the width it came up with.
///
/// This is not redundant with passing `--threads` to the engine's state. The
/// engine's own pool configuration lives in *its* CLI, not in the library this
/// binary links, so without this call `--threads` would reach only the reported
/// execution plan: the pool would come up at the library's default width and two
/// replicas started with different `--threads` would be indistinguishable
/// everywhere this product publishes an identity. A flag that names a thread
/// count and sizes nothing is worse than no flag.
///
/// The read is what makes the published number honest — the width the process
/// has, not the width it was asked for — and it must come *after* any sizing,
/// because reading the width initializes the pool as a side effect and a sizing
/// call after that point fails rather than resizes.
///
/// Known divergence, written down rather than left to be discovered: the
/// engine's own CLI attaches a macOS QoS start handler to its pool to keep
/// workers on performance cores. This does not, so worker placement here is the
/// scheduler's default. That is a latency difference and not a numeric one —
/// placement does not change the arithmetic — but it does mean a performance
/// comparison against the engine's own serve command is not comparing like with
/// like.
fn resolve_worker_pool(threads: Option<usize>) -> Result<usize, Box<dyn std::error::Error>> {
    if let Some(threads) = threads {
        if threads == 0 {
            return Err("--threads must be greater than zero".into());
        }
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build_global()
            .map_err(|err| {
                format!(
                    "could not size the worker pool to {threads} threads: {err}. This replica \
                     publishes the width it is actually running, so it fails closed here rather \
                     than serve while claiming a width it does not have."
                )
            })?;
    }
    Ok(rayon::current_num_threads())
}

async fn shutdown_signal() {
    let ctrl_c = async {
        tokio::signal::ctrl_c()
            .await
            .expect("failed to install Ctrl+C handler");
    };

    #[cfg(unix)]
    let terminate = async {
        tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            .expect("failed to install SIGTERM handler")
            .recv()
            .await;
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {}
        _ = terminate => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use camelid_enterprise::{load_startup_model, served_router, surface, ServedModel};
    use std::path::Path;

    /// The engine's router, exactly as the startup loader receives it.
    fn engine_router() -> axum::Router {
        camelid::api::router_with_state(
            camelid::api::AppState::with_configured_threads(None)
                .with_default_enable_thinking(false)
                .with_models_dir(None),
        )
    }

    /// The id the tests' served stack pins generation to. Arbitrary, and not any
    /// spelling of a path, so an assertion cannot pass by accident.
    const TEST_MODEL_ID: &str = "Test Model 1B";

    /// The stack a client actually meets.
    ///
    /// It is composed by [`served_router`] — the same call `serve` makes through
    /// `replica_router`, and the only place in the workspace that layering is
    /// written down. Restating the layers here instead would have made these
    /// tests pass over a stack of this file's own construction while the shipped
    /// one drifted; that is exactly the gap that let both filters be deleted from
    /// production with the suite still green.
    ///
    /// The only thing supplied by hand is what a real start reads off the disk:
    /// an identity and the key the engine reported for the loaded weights.
    fn served_stack() -> axum::Router {
        let ctx = Attribution {
            lane: "deterministic",
            config_sha256: Arc::new("0".repeat(64)),
            admission_sha256: Arc::new("1".repeat(64)),
            model: ModelIdentity::of_file(Path::new("Cargo.toml"))
                .expect("this crate's own manifest is readable"),
            host: Arc::new("test/host cores=1 simd=none".to_string()),
            workers: WorkerThreads::resolved(1),
            receipts: None,
        };
        let served_model = Arc::new(ServedModel::new(
            TEST_MODEL_ID.to_string(),
            Path::new("/models/pinned.gguf"),
            Path::new("/models/pinned.gguf"),
        ));
        served_router(engine_router(), served_model, ctx)
    }

    async fn through_the_served_stack(method: &str, path: &str) -> axum::response::Response {
        through_the_served_stack_with(method, path, "{}").await
    }

    async fn through_the_served_stack_with(
        method: &str,
        path: &str,
        body: &str,
    ) -> axum::response::Response {
        through(&served_stack(), method, path, body).await
    }

    /// The same, against a stack the caller holds on to.
    ///
    /// `served_stack()` builds fresh engine state every call, so two requests
    /// through it are two replicas. A test about what one replica's state is
    /// after a request has to send both through the same one.
    async fn through(
        stack: &axum::Router,
        method: &str,
        path: &str,
        body: &str,
    ) -> axum::response::Response {
        use axum::body::Body;
        use axum::http::{header::CONTENT_TYPE, Request};
        use tower::ServiceExt;

        stack
            .clone()
            .oneshot(
                Request::builder()
                    .method(method)
                    .uri(path)
                    .header(CONTENT_TYPE, "application/json")
                    .body(Body::from(body.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap()
    }

    /// Every allow-listed route reaches the engine, and every response the
    /// replica emits is attributed — asserted against the engine's real router
    /// rather than a stand-in.
    ///
    /// This is the direction of failure that costs an outage. A route the
    /// contract promises but the filter refuses is a 403 that nothing in this
    /// workspace would otherwise catch, because the filter's own unit tests
    /// answer only "does the rule match", never "is there anything behind it".
    /// Passing here means the container HEALTHCHECK, the three Kubernetes probes
    /// and the gateway's readiness probe all reach a handler.
    ///
    /// The contractual half of the list is read from the shared route registry
    /// rather than written out again. A copy here would be a second inventory of
    /// the public surface, and the failure it would hide is the one that matters:
    /// a route the published contract promises and this replica quietly withholds
    /// would still pass a test that only ever asks about the routes it already
    /// knew about. That includes the compatibility routes, which are contractual
    /// as *explicit* refusals — the engine's own typed "not implemented" answer
    /// is the promise, and it can only be kept if the request reaches it.
    #[tokio::test]
    async fn every_allow_listed_route_reaches_the_engine_and_is_attributed() {
        use axum::http::StatusCode;

        let contractual = replica_contract::PUBLIC_ROUTES.iter().flat_map(|route| {
            route
                .methods
                .iter()
                .map(move |method| (method.as_str(), route.probe_path))
        });
        // Served, and not enumerated by the registry because they are not
        // separate promises: the `HEAD` and CORS-preflight forms of contractual
        // routes, which the contract document calls out in prose as behavior axum
        // and the pinned router provide for routes already listed.
        //
        // The engine's bare `/health` is deliberately absent. It is
        // replica-private diagnostics under the contract, `/v1/health` is what
        // every probe in this tree polls, and serving it would be an eleventh
        // route living outside the registry.
        let operational = [("HEAD", "/v1/models"), ("OPTIONS", "/v1/chat/completions")];

        for (method, path) in contractual.chain(operational) {
            let response = through_the_served_stack(method, path).await;
            assert_ne!(
                response.status(),
                StatusCode::FORBIDDEN,
                "{method} {path} is on the served surface but the filter refused it"
            );
            assert_ne!(
                response.status(),
                StatusCode::METHOD_NOT_ALLOWED,
                "{method} {path} is allowed by method but the engine has no handler for it"
            );
            // Attribution is outermost, so it stamps whatever the stack
            // produced — including the engine's own error answers.
            let headers = response.headers();
            for name in [
                "x-camelid-lane",
                "x-camelid-config-sha256",
                "x-camelid-admission-sha256",
                "x-camelid-model-sha256",
                "x-camelid-host",
                "x-camelid-worker-threads",
            ] {
                assert!(headers.contains_key(name), "{method} {path} lost {name}");
            }
        }
    }

    /// The readiness routes answer 200 with no model loaded, which is what makes
    /// them usable as probes: a probe that only passes once the replica is
    /// serving cannot report that the replica is starting.
    ///
    /// Both are contractual, which is the point — the container HEALTHCHECK, the
    /// three Kubernetes probes, the drain loop and the macOS installer all poll
    /// `/v1/health` or `/v1/models`, so the routes the deployment depends on are
    /// the routes the published contract promises. The engine's bare `/health` is
    /// replica-private and is not served; nothing in this tree asks for it.
    #[tokio::test]
    async fn the_probe_routes_answer_before_a_model_is_loaded() {
        use axum::http::StatusCode;

        for path in ["/v1/health", "/v1/models"] {
            let response = through_the_served_stack("GET", path).await;
            assert_eq!(response.status(), StatusCode::OK, "GET {path} must answer a probe");
        }
        assert_eq!(
            through_the_served_stack("GET", "/health").await.status(),
            StatusCode::FORBIDDEN,
            "the engine's private liveness route is outside the contract and is withheld"
        );
    }

    /// A refusal is a response this replica emitted, so it is attributed like
    /// any other. A client that gets a 403 from a pool has to be able to tell
    /// which replica refused it, under what configuration and which weights.
    #[tokio::test]
    async fn a_refusal_still_carries_the_replicas_identity() {
        use axum::http::StatusCode;

        let response = through_the_served_stack("POST", "/api/models/load").await;
        assert_eq!(response.status(), StatusCode::FORBIDDEN);
        for name in [
            "x-camelid-lane",
            "x-camelid-config-sha256",
            "x-camelid-admission-sha256",
            "x-camelid-model-sha256",
            "x-camelid-host",
            "x-camelid-worker-threads",
        ] {
            assert!(response.headers().contains_key(name), "a refusal lost {name}");
        }
    }

    /// The engine's router falls back to an embedded web application, which
    /// answers an unmatched extensionless path with the app shell and HTTP 200.
    /// Through the filter it is refused instead — including the multi-segment
    /// path under `/v1/models/` that a looser prefix rule would have admitted
    /// straight into that fallback.
    #[tokio::test]
    async fn the_embedded_web_application_is_not_reachable_through_the_filter() {
        use axum::http::StatusCode;

        for path in ["/", "/index.html", "/v1/models/a/b"] {
            let response = through_the_served_stack("GET", path).await;
            assert_eq!(
                response.status(),
                StatusCode::FORBIDDEN,
                "GET {path} reached the embedded application"
            );
        }
    }

    /// The load reaches the engine's own handler in-process.
    ///
    /// This is the assertion the in-process dispatch rests on, and it is not
    /// self-evident: a handler that extracted a connection-info extension would
    /// panic when dispatched without a listener, and the failure would appear
    /// only at startup on a real deployment. Asking for a model that is not
    /// there proves the request was routed, extracted and answered by the
    /// engine — a refusal from the handler, not a 404 from the router and not a
    /// panic.
    #[tokio::test]
    async fn the_startup_load_reaches_the_engine_handler_without_a_socket() {
        let dir = tempfile::tempdir().unwrap();
        let missing = dir.path().join("not-a-model.gguf");

        let err = load_startup_model(engine_router(), &missing)
            .await
            .expect_err("loading a model that does not exist must fail")
            .to_string();

        // The loader's own sentence, not merely the model's name: the engine's
        // error body also names the file, so a bare `contains` for the file name
        // passes even if this binary's message is deleted entirely.
        assert!(
            err.starts_with(&format!("could not load the model at {}", missing.display())),
            "the operator's line must be the loader's own, naming the model it was given: {err}"
        );
        assert!(
            err.contains("the engine answered HTTP"),
            "the failure must be the engine's answer, not a routing or transport error: {err}"
        );
        assert!(
            !err.contains("HTTP 404"),
            "a 404 means the request never reached the loader: {err}"
        );
        // The engine's own diagnosis has to survive into the operator's one
        // line, or a failed startup says only that a number was not 200.
        assert!(
            err.contains("model_io_error"),
            "the engine's error body must be carried through: {err}"
        );
    }

    /// The loader reads the id out of a reply the size the engine actually
    /// sends, which is megabytes rather than kilobytes.
    ///
    /// The engine answers a successful load with its whole loaded-model record —
    /// GGUF metadata and a descriptor per tensor — so a buffer sized for a typed
    /// error truncates it, the id parse fails, and the replica refuses to start
    /// with a message blaming the engine for reporting no id. That failure is
    /// invisible to every other test here, because they all drive the *error*
    /// path, where the body really is small. The filler below is far larger than
    /// any plausible error body and smaller than a real model record.
    #[tokio::test]
    async fn the_loader_reads_an_id_out_of_a_reply_the_size_the_engine_sends() {
        use axum::routing::post;
        use axum::Json;

        let filler: Vec<u32> = (0..2_000_000).collect();
        let router = axum::Router::new().route(
            "/api/models/load",
            post(move || async move {
                Json(serde_json::json!({ "id": "Llama 3.2 1B Instruct", "tensors": filler }))
            }),
        );
        let id = load_startup_model(router, Path::new("/models/model.gguf"))
            .await
            .expect("a successful load reports its id");
        assert_eq!(id, "Llama 3.2 1B Instruct");
    }

    /// A load that answers `200` without an id is a startup failure, not a
    /// replica served with an unenforced model field.
    #[tokio::test]
    async fn a_load_that_reports_no_id_refuses_to_start() {
        use axum::routing::post;
        use axum::Json;

        for reply in [serde_json::json!({ "path": "/models/model.gguf" }), serde_json::json!({ "id": "" })] {
            let router = axum::Router::new()
                .route("/api/models/load", post(move || {
                    let reply = reply.clone();
                    async move { Json(reply) }
                }));
            let err = load_startup_model(router, Path::new("/models/model.gguf"))
                .await
                .expect_err("without the engine's key nothing can pin the model field")
                .to_string();
            assert!(err.contains("did not report the id"), "unexpected message: {err}");
        }
    }

    /// The loader must be handed the *unfiltered* router, and the two facts that
    /// make that necessary are asserted together here rather than left to the
    /// reader to connect.
    ///
    /// The route the loader uses is one the served surface refuses — if it were
    /// not, the filter would have a hole exactly where the control plane is —
    /// and consequently the loader fails outright if it is ever handed the
    /// served stack instead. `serve` passes `engine` and layers `served`
    /// separately from it; this is what would catch that being collapsed into
    /// one router.
    #[tokio::test]
    async fn the_loader_fails_if_it_is_handed_the_filtered_router() {
        let dir = tempfile::tempdir().unwrap();
        let model = dir.path().join("model.gguf");
        std::fs::write(&model, b"not really a model").unwrap();

        let filtered =
            engine_router().layer(axum::middleware::from_fn(surface::serve_only_the_lane));
        let err = load_startup_model(filtered, &model)
            .await
            .expect_err("the served surface must not carry the loader")
            .to_string();
        assert!(
            err.contains("HTTP 403"),
            "the filter, not the engine, must be what refuses the loader here: {err}"
        );

        // …and through the unfiltered router the same request reaches the
        // engine, which refuses it on its own terms rather than on the filter's.
        let err = load_startup_model(engine_router(), &model)
            .await
            .expect_err("a file that is not a GGUF cannot load")
            .to_string();
        assert!(!err.contains("HTTP 403"), "the unfiltered router must not refuse it: {err}");
    }

    /// The blocker this filter exists for, driven end to end against the
    /// engine's real router: a generation request naming another file on disk
    /// must not reach the engine's model resolution.
    ///
    /// The path is real and readable — this crate's own manifest — because the
    /// engine's resolver takes its filesystem branch on `exists()`, so a
    /// nonexistent path would pass this test for the wrong reason. The refusal
    /// has to happen because the name is not this replica's, never because the
    /// file was missing.
    #[tokio::test]
    async fn a_generation_request_cannot_name_another_file_on_disk() {
        use axum::body::to_bytes;
        use axum::http::StatusCode;

        let elsewhere = std::fs::canonicalize("Cargo.toml").unwrap();
        for path in ["/v1/chat/completions", "/v1/completions"] {
            let body = serde_json::json!({
                "model": elsewhere.to_string_lossy(),
                "prompt": "hi",
                "messages": [{ "role": "user", "content": "hi" }],
                "max_tokens": 1,
            })
            .to_string();
            let response = through_the_served_stack_with("POST", path, &body).await;
            assert_eq!(
                response.status(),
                StatusCode::NOT_FOUND,
                "POST {path} admitted a request naming other weights"
            );
            let bytes = to_bytes(response.into_body(), 64 * 1024).await.unwrap();
            let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
            assert_eq!(json["error"]["code"], "model_not_served");
            assert!(
                json["error"]["message"].as_str().unwrap().contains(TEST_MODEL_ID),
                "the refusal should name the model this replica does serve: {json}"
            );
        }
    }

    /// The same request, plus one byte, against the engine's real router: the
    /// trailing-byte bypass, which is the shape the blocker actually took.
    ///
    /// The engine's JSON extractor deserializes the first value in the buffer
    /// and never asks what follows it, so `{…}x` is exactly as good a request to
    /// the engine as `{…}` — same `model` field, same on-demand load — while a
    /// whole-input parse in the filter in front of it fails. A filter that
    /// forwarded what it could not parse therefore refused the clean body and
    /// served the one with a byte stuck on the end.
    ///
    /// The second half is the assertion that makes this more than a status-code
    /// check: the *same stack* is then asked to generate with no `model` field
    /// at all, and still answers from no model. That is what says nothing was
    /// loaded on the way through — an on-demand load would have left the named
    /// file active for every later request, which is what made the swap
    /// permanent for the process lifetime rather than scoped to one request.
    #[tokio::test]
    async fn a_trailing_byte_cannot_carry_a_model_past_the_body_filter() {
        use axum::body::to_bytes;
        use axum::http::StatusCode;

        let stack = served_stack();
        let elsewhere = std::fs::canonicalize("Cargo.toml").unwrap();
        // One well-formed request per route — the engine refuses a body carrying
        // both spellings before it looks at anything else, and a refusal for the
        // wrong reason would pass this test without proving the request was
        // stopped.
        let clean = |path: &str| {
            let mut body = serde_json::json!({
                "model": elsewhere.to_string_lossy(),
                "max_tokens": 1,
            });
            if path == "/v1/completions" {
                body["prompt"] = "hi".into();
            } else {
                body["messages"] = serde_json::json!([{ "role": "user", "content": "hi" }]);
            }
            body.to_string()
        };

        for suffix in ["x", "\0", "{}", "//x", "]", " \t\nx"] {
            for path in ["/v1/completions", "/v1/chat/completions"] {
                let response =
                    through(&stack, "POST", path, &format!("{}{suffix}", clean(path))).await;
                assert_eq!(
                    response.status(),
                    StatusCode::BAD_REQUEST,
                    "POST {path} with {suffix:?} appended was not refused"
                );
                let bytes = to_bytes(response.into_body(), 64 * 1024).await.unwrap();
                let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
                assert_eq!(json["error"]["code"], "unparseable_request_body");
            }
        }

        // …and without the byte, the same request is refused by name — which is
        // the pair that says the byte was the whole difference.
        for path in ["/v1/completions", "/v1/chat/completions"] {
            let response = through(&stack, "POST", path, &clean(path)).await;
            assert_eq!(response.status(), StatusCode::NOT_FOUND);
            let bytes = to_bytes(response.into_body(), 64 * 1024).await.unwrap();
            let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
            assert_eq!(json["error"]["code"], "model_not_served");
        }

        let after = through(
            &stack,
            "POST",
            "/v1/chat/completions",
            r#"{"messages":[{"role":"user","content":"hi"}],"max_tokens":1}"#,
        )
        .await;
        let bytes = to_bytes(after.into_body(), 64 * 1024).await.unwrap();
        let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(
            json["error"]["code"], "model_not_loaded",
            "a request naming no model is being answered by weights something loaded on demand: \
             {json}"
        );
    }

    /// The same refusal for a path that does not exist, byte for byte. The
    /// `model` field of an admitted request must not become a way to ask the
    /// replica what is on its disk, and the only thing that makes the two
    /// indistinguishable is that neither reaches the engine.
    #[tokio::test]
    async fn a_missing_and_a_present_file_are_refused_identically() {
        use axum::body::to_bytes;

        async fn refusal_for(model: &str) -> (u16, String) {
            let body = serde_json::json!({
                "model": model,
                "messages": [{ "role": "user", "content": "hi" }],
            })
            .to_string();
            let response =
                through_the_served_stack_with("POST", "/v1/chat/completions", &body).await;
            let status = response.status().as_u16();
            let bytes = to_bytes(response.into_body(), 64 * 1024).await.unwrap();
            (status, String::from_utf8(bytes.to_vec()).unwrap())
        }

        let present = std::fs::canonicalize("Cargo.toml").unwrap();
        let (present_status, present_body) =
            refusal_for(&present.to_string_lossy()).await;
        let (missing_status, missing_body) =
            refusal_for("/no/such/directory/definitely-not-here.gguf").await;
        assert_eq!(present_status, missing_status);
        assert_eq!(
            present_body, missing_body,
            "an existing file and a missing one must be refused with the same bytes, or the \
             model field discloses what is on this replica's disk"
        );
    }

    /// The names this replica does answer to reach the engine. This is the
    /// direction that costs an outage: a filter that refused the model's own id,
    /// or the path the operator started it with, would refuse every request a
    /// documented client makes.
    ///
    /// "Reaches the engine" is pinned to the engine's *own* code rather than to
    /// the absence of this filter's, because "not refused here" is also true of
    /// a response that never had a body. No model is loaded in this stack, so
    /// the engine answers `model_not_loaded` — a code only the engine emits, and
    /// therefore proof the request got all the way through.
    #[tokio::test]
    async fn every_name_for_the_served_model_reaches_the_engine() {
        use axum::body::to_bytes;

        async fn error_code_for(model: Option<&str>) -> String {
            let mut body = serde_json::json!({
                "messages": [{ "role": "user", "content": "hi" }],
                "max_tokens": 1,
            });
            if let Some(model) = model {
                body["model"] = model.into();
            }
            let response =
                through_the_served_stack_with("POST", "/v1/chat/completions", &body.to_string())
                    .await;
            let bytes = to_bytes(response.into_body(), 64 * 1024).await.unwrap();
            let json: serde_json::Value = serde_json::from_slice(&bytes)
                .unwrap_or_else(|_| panic!("a JSON answer: {:?}", String::from_utf8_lossy(&bytes)));
            json["error"]["code"].as_str().unwrap_or_default().to_string()
        }

        // Every spelling of the one file: the engine's key, the path the replica
        // was started with, the file's name, and its stem.
        for name in [TEST_MODEL_ID, "/models/pinned.gguf", "pinned.gguf", "pinned"] {
            assert_eq!(
                error_code_for(Some(name)).await,
                "model_not_loaded",
                "'{name}' names the served model and must reach the engine"
            );
        }

        // …and a request naming no model at all is forwarded untouched, which is
        // the common case and the one the engine answers from its active model.
        assert_eq!(error_code_for(None).await, "model_not_loaded");
    }

    /// The body filter must never be the larger buffer.
    ///
    /// It reads a generation body to check the field it exists to check, so it
    /// holds bytes the engine has not seen yet. If its ceiling sat above the
    /// engine's own, this filter would have handed an unauthenticated caller a
    /// bigger allocation than the surface it was added to protect — a
    /// self-inflicted version of the unbounded loading it closes. The ceiling is
    /// therefore the engine extractor's, and this asserts that against the real
    /// engine router rather than against a number copied into a comment: one
    /// byte over, unfiltered, and the engine already refuses.
    #[tokio::test]
    async fn the_filter_never_buffers_more_than_the_engine_would_accept() {
        use axum::body::Body;
        use axum::http::{header::CONTENT_TYPE, Request, StatusCode};
        use tower::ServiceExt;

        // Valid JSON at a chosen size, so the engine's answer is about the size
        // and not about the syntax.
        async fn engine_answer_to(total: usize) -> String {
            let envelope = br#"{"max_tokens":1,"prompt":""}"#.len();
            let body = format!(
                r#"{{"max_tokens":1,"prompt":"{}"}}"#,
                "x".repeat(total - envelope)
            );
            assert_eq!(body.len(), total);
            let response = engine_router()
                .oneshot(
                    Request::builder()
                        .method("POST")
                        .uri("/v1/completions")
                        .header(CONTENT_TYPE, "application/json")
                        .body(Body::from(body))
                        .unwrap(),
                )
                .await
                .unwrap();
            let status = response.status();
            let bytes =
                axum::body::to_bytes(response.into_body(), 64 * 1024).await.unwrap_or_default();
            let json: serde_json::Value =
                serde_json::from_slice(&bytes).unwrap_or(serde_json::Value::Null);
            format!("{status} {}", json["error"]["code"].as_str().unwrap_or("-"))
        }

        // One byte under the ceiling the filter enforces, the engine parses the
        // request and fails it on its own terms — no model is loaded here.
        assert_eq!(
            engine_answer_to(surface::REQUEST_BODY_LIMIT).await,
            format!("{} model_not_loaded", StatusCode::NOT_FOUND),
            "at the filter's ceiling the engine still accepts the body, so the ceiling is not \
             below the engine's and no legitimate request is refused early"
        );

        // One byte over, and the engine refuses to read it at all. If this ever
        // starts matching the line above, the filter has become the larger
        // buffer and is holding bytes for a request the engine would not have.
        assert_ne!(
            engine_answer_to(surface::REQUEST_BODY_LIMIT + 1).await,
            format!("{} model_not_loaded", StatusCode::NOT_FOUND),
            "the filter's ceiling is above the engine's own, so it buffers bytes the engine \
             would have refused"
        );
    }

    /// A pool of zero threads is not a smaller pool, it is a request rayon reads
    /// as "choose for me" — the opposite of what an operator typing `0` means.
    /// Refuse it rather than publish a width nobody asked for.
    #[test]
    fn a_zero_width_pool_is_refused_before_anything_is_built() {
        let err = resolve_worker_pool(Some(0)).unwrap_err().to_string();
        assert!(err.contains("greater than zero"), "unexpected message: {err}");
    }

    /// With no `--threads`, the published width is the pool's own, not a
    /// placeholder — the number is read back from rayon in both cases, so there
    /// is no arm in which the replica publishes a width it did not resolve.
    #[test]
    fn an_unsized_pool_still_publishes_a_resolved_width() {
        let width = resolve_worker_pool(None).expect("reading the default width cannot fail");
        assert_eq!(width, rayon::current_num_threads());
        assert!(width > 0, "a resolved width of zero would be a published falsehood");
    }
}
