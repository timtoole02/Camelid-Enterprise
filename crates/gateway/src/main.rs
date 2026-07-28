use camelid_enterprise_gateway::{
    router_with_model_catalog, router_with_options, GatewayAuth, GatewayLog, LogFlush,
    ModelCatalog, ModelSelectionLimits, OrgQuota, UpstreamOrigin, VerifiedModelCatalog,
    DEFAULT_LOG_FLUSH_DEADLINE, DEFAULT_MAX_CONNECTION_DURATION, DEFAULT_MAX_IN_FLIGHT,
    DEFAULT_MAX_MODEL_SELECTION_BODY_BYTES, DEFAULT_MODEL_SELECTION_MEMORY_BUDGET_BYTES,
};
use clap::{Parser, Subcommand};
use identity::{OrganizationId, PrincipalId, RotationLifetime, SqliteIdentityStore, TokenLifetime};
use std::net::SocketAddr;
use std::num::{NonZeroU32, NonZeroU64, NonZeroUsize};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

#[derive(Parser)]
#[command(name = "camelid-enterprise-gateway", version, about)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Start a transparent single-upstream gateway or a static multi-model gateway.
    Serve {
        /// Replica origin for legacy transparent mode, including the http://
        /// scheme and optional port. Mutually exclusive with --model-route.
        #[arg(
            long,
            env = "CAMELID_GATEWAY_UPSTREAM",
            required_unless_present = "model_route",
            conflicts_with = "model_route"
        )]
        upstream: Option<String>,
        /// Static model-id-to-replica-pool mapping, written as
        /// `<model-id>=<http://origin>`. Repeat for every model. Enables
        /// catalog mode and is mutually exclusive with --upstream.
        #[arg(long, value_name = "MODEL_ID=ORIGIN", conflicts_with = "upstream")]
        model_route: Vec<String>,
        /// Largest JSON generation request catalog mode may materialize to
        /// select a model. Ignored by transparent --upstream mode.
        #[arg(
            long,
            default_value_t = DEFAULT_MAX_MODEL_SELECTION_BODY_BYTES,
            env = "CAMELID_GATEWAY_MAX_MODEL_SELECTION_BODY_BYTES"
        )]
        max_model_selection_body_bytes: NonZeroUsize,
        /// Total memory reserved for concurrently materialized catalog selector
        /// bodies. Must be at least --max-model-selection-body-bytes. The
        /// gateway permits `budget / (2 * max-body)` selector bodies at once,
        /// reserving space for a decoded escaped model id.
        #[arg(
            long,
            default_value_t = DEFAULT_MODEL_SELECTION_MEMORY_BUDGET_BYTES,
            env = "CAMELID_GATEWAY_MODEL_SELECTION_MEMORY_BUDGET_BYTES"
        )]
        model_selection_memory_budget_bytes: NonZeroUsize,
        /// Bind address for client traffic.
        #[arg(long, default_value = "127.0.0.1:8080", env = "CAMELID_GATEWAY_ADDR")]
        addr: SocketAddr,
        /// Maximum request streams forwarded concurrently by this gateway.
        #[arg(
            long,
            default_value_t = DEFAULT_MAX_IN_FLIGHT,
            env = "CAMELID_GATEWAY_MAX_IN_FLIGHT"
        )]
        max_in_flight: NonZeroUsize,
        /// Maximum seconds a single client connection may stay open. This is
        /// a hard cap, not an idle timeout: it bounds how long a stalled or
        /// malicious client (a slow request-body drip, or a client that
        /// never reads its response) can pin an admission permit. Legitimate
        /// long-running generations must finish within this bound.
        #[arg(
            long,
            default_value_t = DEFAULT_MAX_CONNECTION_DURATION.as_secs(),
            env = "CAMELID_GATEWAY_MAX_CONNECTION_SECONDS"
        )]
        max_connection_seconds: u64,
        /// Path to a local identity database. When set, every request must
        /// carry `Authorization: Bearer <token>` resolving to a known
        /// principal, or the gateway rejects it before it ever reaches a
        /// replica. When omitted, the gateway forwards every request
        /// unauthenticated, unchanged from prior releases.
        #[arg(long, env = "CAMELID_GATEWAY_IDENTITY_DB")]
        identity_db: Option<PathBuf>,
        /// Path to an append-only JSONL audit log. When set, the gateway
        /// records one line per request it handles — including requests it
        /// rejects for authentication or admission — carrying the request's
        /// correlation id, the resolved principal and organization (or null),
        /// the refusal reason (null for anything it forwarded), method, path,
        /// and status. Join it to a replica's serving receipts on `request_id`
        /// to see which deterministic configuration served each caller's
        /// request.
        #[arg(long, env = "CAMELID_GATEWAY_AUDIT_LOG")]
        audit_log: Option<PathBuf>,
        /// Path to an append-only JSONL transport-usage log. Each handled
        /// request records its terminal stream outcome and raw payload bytes
        /// observed by the gateway. This is not tokenizer usage or billing.
        #[arg(long, env = "CAMELID_GATEWAY_USAGE_LOG", requires = "identity_db")]
        usage_log: Option<PathBuf>,
        /// Maximum requests one organization may send per fixed quota window.
        /// Requires `--identity-db`, since a request needs a resolved
        /// organization to be charged against a quota. Omitted by default:
        /// no organization is rate-limited unless this is set.
        #[arg(
            long,
            env = "CAMELID_GATEWAY_ORG_REQUEST_QUOTA",
            requires = "identity_db"
        )]
        org_request_quota: Option<NonZeroU32>,
        /// Length, in seconds, of the fixed window `--org-request-quota`
        /// counts requests over. Ignored unless `--org-request-quota` is set.
        #[arg(
            long,
            default_value_t = NonZeroU64::new(60).unwrap(),
            env = "CAMELID_GATEWAY_ORG_REQUEST_QUOTA_WINDOW_SECONDS"
        )]
        org_request_quota_window_seconds: NonZeroU64,
    },
    /// Create a user and its personal organization, then print the principal id.
    CreateUser {
        #[arg(long, env = "CAMELID_GATEWAY_IDENTITY_DB")]
        identity_db: PathBuf,
        /// Display label for the user. Never used as a lookup key.
        name: String,
    },
    /// List every user as `<principal-id>\t<name>`, one per line.
    ///
    /// The way back to a principal id that was not written down when
    /// `create-user` printed it. Without it a lapsed token is a dead end:
    /// rotation refuses an expired credential, re-issuing needs the id, and
    /// creating the user again mints a different one.
    ListUsers {
        #[arg(long, env = "CAMELID_GATEWAY_IDENTITY_DB")]
        identity_db: PathBuf,
    },
    /// Create an organization and print its opaque organization id.
    CreateOrganization {
        #[arg(long, env = "CAMELID_GATEWAY_IDENTITY_DB")]
        identity_db: PathBuf,
        /// Display label for the organization. Never used as a lookup key.
        name: String,
    },
    /// List the opaque organization ids a principal belongs to, one per line.
    ListOrganizations {
        #[arg(long, env = "CAMELID_GATEWAY_IDENTITY_DB")]
        identity_db: PathBuf,
        /// A principal id printed by `create-user`.
        principal: String,
    },
    /// Add an existing principal to an existing organization.
    AddPrincipalToOrganization {
        #[arg(long, env = "CAMELID_GATEWAY_IDENTITY_DB")]
        identity_db: PathBuf,
        principal: String,
        organization: String,
    },
    /// Remove a principal from an organization and revoke that membership's tokens.
    RemovePrincipalFromOrganization {
        #[arg(long, env = "CAMELID_GATEWAY_IDENTITY_DB")]
        identity_db: PathBuf,
        principal: String,
        organization: String,
    },
    /// Issue a new bearer token for a principal and print it once.
    ///
    /// The plaintext token is never stored or shown again; only its hash is
    /// persisted.
    IssueToken {
        #[arg(long, env = "CAMELID_GATEWAY_IDENTITY_DB")]
        identity_db: PathBuf,
        /// A principal id printed by `create-user`.
        principal: String,
        /// Organization id to bind into the token. Required when the principal
        /// belongs to more than one organization.
        #[arg(long)]
        organization: Option<String>,
        /// How long the token stays valid, in seconds. Omitted by default,
        /// which issues a token that never expires -- the behavior of every
        /// token issued before this option existed. A token that has expired
        /// is refused by the gateway exactly as an unknown one is.
        #[arg(long)]
        expires_in_seconds: Option<NonZeroU64>,
    },
    /// Exchange a valid token for a fresh one and print the replacement once.
    ///
    /// The presented token stops working the moment this succeeds, and the
    /// replacement is issued in the same transaction: there is no window in
    /// which both work, and none in which neither does. Requires the plaintext
    /// of the token being replaced, so this refreshes a credential for whoever
    /// holds it rather than resetting someone else's.
    ///
    /// By default the replacement carries the same lifetime the presented
    /// token was issued with, measured from now -- rotating a credential that
    /// is nearing expiry gives another full lifetime, not a permanent
    /// credential. Dropping the bound is available as `--no-expiry`, but it
    /// has to be asked for.
    RotateToken {
        #[arg(long, env = "CAMELID_GATEWAY_IDENTITY_DB")]
        identity_db: PathBuf,
        /// The token to replace, or `-` to read it from stdin instead.
        /// Passing the plaintext token directly on the command line leaves
        /// it in shell history and briefly visible to other processes via
        /// `ps`; prefer `-` and pipe it in.
        token: String,
        /// Give the replacement this lifetime in seconds instead of the one
        /// the presented token was issued with.
        #[arg(long, conflicts_with = "no_expiry")]
        expires_in_seconds: Option<NonZeroU64>,
        /// Give the replacement no expiry at all, discarding whatever bound
        /// the presented token carried. This converts a time-limited
        /// credential into a permanent one, so it is never the default.
        #[arg(long)]
        no_expiry: bool,
    },
    /// Revoke a bearer token so it no longer resolves.
    RevokeToken {
        #[arg(long, env = "CAMELID_GATEWAY_IDENTITY_DB")]
        identity_db: PathBuf,
        /// The token to revoke, or `-` to read it from stdin instead.
        /// Passing the plaintext token directly on the command line leaves
        /// it in shell history and briefly visible to other processes via
        /// `ps`; prefer `-` and pipe it in.
        token: String,
    },
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    match Cli::parse().command {
        Command::Serve {
            upstream,
            model_route,
            max_model_selection_body_bytes,
            model_selection_memory_budget_bytes,
            addr,
            max_in_flight,
            max_connection_seconds,
            identity_db,
            audit_log,
            usage_log,
            org_request_quota,
            org_request_quota_window_seconds,
        } => {
            serve(ServeArgs {
                upstream,
                model_routes: model_route,
                max_model_selection_body_bytes,
                model_selection_memory_budget_bytes,
                addr,
                max_in_flight,
                max_connection_seconds,
                identity_db,
                logs: GatewayLogArgs {
                    audit_log,
                    usage_log,
                },
                org_request_quota: OrgQuotaArgs {
                    limit: org_request_quota,
                    window_seconds: org_request_quota_window_seconds,
                },
            })
            .await
        }
        Command::CreateUser { identity_db, name } => create_user(&identity_db, &name),
        Command::ListUsers { identity_db } => list_users(&identity_db),
        Command::CreateOrganization { identity_db, name } => {
            create_organization(&identity_db, &name)
        }
        Command::ListOrganizations {
            identity_db,
            principal,
        } => list_organizations(&identity_db, &principal),
        Command::AddPrincipalToOrganization {
            identity_db,
            principal,
            organization,
        } => add_principal_to_organization(&identity_db, &principal, &organization),
        Command::RemovePrincipalFromOrganization {
            identity_db,
            principal,
            organization,
        } => remove_principal_from_organization(&identity_db, &principal, &organization),
        Command::IssueToken {
            identity_db,
            principal,
            organization,
            expires_in_seconds,
        } => issue_token(
            &identity_db,
            &principal,
            organization.as_deref(),
            expires_in_seconds,
        ),
        Command::RotateToken {
            identity_db,
            token,
            expires_in_seconds,
            no_expiry,
        } => rotate_token(&identity_db, &token, expires_in_seconds, no_expiry),
        Command::RevokeToken { identity_db, token } => revoke_token(&identity_db, &token),
    }
}

/// The `--org-request-quota` / `--org-request-quota-window-seconds` pair,
/// grouped so [`serve`] takes one argument for them instead of two that must
/// always travel together.
struct OrgQuotaArgs {
    limit: Option<NonZeroU32>,
    window_seconds: NonZeroU64,
}

/// Append-only gateway telemetry sinks. Audit remains a response-head and
/// identity-correlation record; usage is a terminal raw-payload record.
struct GatewayLogArgs {
    audit_log: Option<PathBuf>,
    usage_log: Option<PathBuf>,
}

/// Fully parsed `serve` configuration. Grouping this before startup keeps the
/// CLI match and the operating path from drifting as gateway modes grow.
struct ServeArgs {
    upstream: Option<String>,
    model_routes: Vec<String>,
    max_model_selection_body_bytes: NonZeroUsize,
    model_selection_memory_budget_bytes: NonZeroUsize,
    addr: SocketAddr,
    max_in_flight: NonZeroUsize,
    max_connection_seconds: u64,
    identity_db: Option<PathBuf>,
    logs: GatewayLogArgs,
    org_request_quota: OrgQuotaArgs,
}

type GatewayLogs = (Option<Arc<GatewayLog>>, Option<Arc<GatewayLog>>);

enum ConfiguredServeRouting {
    Passthrough(UpstreamOrigin),
    Catalog(ModelCatalog),
}

enum ServeRouting {
    Passthrough(UpstreamOrigin),
    Catalog(VerifiedModelCatalog),
}

fn parse_serve_routing(
    upstream: Option<String>,
    model_routes: Vec<String>,
) -> Result<ConfiguredServeRouting, Box<dyn std::error::Error>> {
    match (upstream, model_routes.is_empty()) {
        (Some(upstream), true) => Ok(ConfiguredServeRouting::Passthrough(UpstreamOrigin::parse(
            &upstream,
        )?)),
        (None, false) => {
            let routes = model_routes
                .into_iter()
                .map(|route| parse_model_route(&route))
                .collect::<Result<Vec<_>, _>>()?;
            Ok(ConfiguredServeRouting::Catalog(ModelCatalog::new(routes)?))
        }
        // Clap enforces this for command-line callers. Retaining the check
        // makes `serve` correct for direct callers and protects future CLI
        // changes from silently choosing a routing mode.
        (Some(_), false) => Err("--upstream and --model-route cannot be combined".into()),
        (None, true) => Err("either --upstream or at least one --model-route is required".into()),
    }
}

async fn verify_serve_routing(
    routing: ConfiguredServeRouting,
) -> Result<ServeRouting, Box<dyn std::error::Error>> {
    match routing {
        ConfiguredServeRouting::Passthrough(upstream) => Ok(ServeRouting::Passthrough(upstream)),
        ConfiguredServeRouting::Catalog(catalog) => Ok(ServeRouting::Catalog(
            catalog.verify_backend_model_ids().await?,
        )),
    }
}

fn parse_model_route(value: &str) -> Result<(String, UpstreamOrigin), Box<dyn std::error::Error>> {
    let (model_id, origin) = value
        .split_once('=')
        .ok_or("--model-route must use MODEL_ID=http://ORIGIN")?;
    if model_id.is_empty() || origin.is_empty() {
        return Err("--model-route must use a non-empty MODEL_ID and ORIGIN".into());
    }
    Ok((model_id.to_string(), UpstreamOrigin::parse(origin)?))
}

fn open_gateway_logs(logs: &GatewayLogArgs) -> Result<GatewayLogs, Box<dyn std::error::Error>> {
    let audit = logs
        .audit_log
        .as_deref()
        .map(GatewayLog::open)
        .transpose()?;
    let usage = logs
        .usage_log
        .as_deref()
        .map(GatewayLog::open)
        .transpose()?;
    if let (Some(audit), Some(usage)) = (&audit, &usage) {
        if audit.has_same_destination(usage) {
            return Err("--audit-log and --usage-log must name different files".into());
        }
    }
    Ok((audit, usage))
}

async fn serve(args: ServeArgs) -> Result<(), Box<dyn std::error::Error>> {
    let ServeArgs {
        upstream,
        model_routes,
        max_model_selection_body_bytes,
        model_selection_memory_budget_bytes,
        addr,
        max_in_flight,
        max_connection_seconds,
        identity_db,
        logs,
        org_request_quota,
    } = args;
    let configured_routing = parse_serve_routing(upstream, model_routes)?;
    let selection_limits = match &configured_routing {
        ConfiguredServeRouting::Passthrough(_) => None,
        ConfiguredServeRouting::Catalog(_) => {
            let limits = ModelSelectionLimits::new(
                max_model_selection_body_bytes,
                model_selection_memory_budget_bytes,
            )?;
            // The catalog selector never rewrites a client request. Requiring
            // exact backend ids before binding avoids a gateway that accepts a
            // public alias and only fails later at the selected replica.
            Some(limits)
        }
    };
    let routing = verify_serve_routing(configured_routing).await?;
    let (audit, usage) = open_gateway_logs(&logs)?;
    // Cloned before the sinks are moved into the router and the auth mode:
    // both must still be reachable after `serve` returns so shutdown can drain
    // what is still queued.
    let audit_sink = audit.clone();
    let usage_sink = usage.clone();
    if let Some(audit) = &audit {
        tracing::info!(path = %audit.path().display(), "gateway request audit log enabled");
    }
    if let Some(usage) = &usage {
        tracing::info!(path = %usage.path().display(), "gateway transport usage log enabled");
    }
    let auth = match identity_db {
        Some(path) => {
            tracing::info!(path = %path.display(), "gateway auth enforcement enabled");
            tracing::warn!(
                "bearer tokens are sent as plain HTTP headers; this gateway does not \
                 terminate TLS. Put a TLS-terminating ingress/reverse proxy (or mTLS) in \
                 front of it, or restrict --addr to a trusted network, or a captured \
                 token can be replayed until it is revoked or, if it was issued with \
                 --expires-in-seconds, until it expires."
            );
            let quota = org_request_quota.limit.map(|limit| {
                tracing::info!(
                    limit = limit.get(),
                    window_seconds = org_request_quota.window_seconds.get(),
                    "gateway per-organization request quota enabled"
                );
                Arc::new(OrgQuota::new(limit, org_request_quota.window_seconds))
            });
            GatewayAuth::RequireToken {
                store: Arc::new(SqliteIdentityStore::open(&path)?),
                quota,
                usage,
            }
        }
        None => {
            tracing::warn!(
                "no --identity-db configured; the gateway is forwarding every request unauthenticated"
            );
            GatewayAuth::Disabled
        }
    };
    let router = match routing {
        ServeRouting::Passthrough(upstream) => {
            router_with_options(upstream, max_in_flight, auth, audit)
        }
        ServeRouting::Catalog(catalog) => {
            let selection_limits = selection_limits
                .expect("catalog routing creates validated selection limits before binding");
            tracing::info!(
                model_count = catalog.len(),
                max_model_selection_body_bytes = selection_limits.max_body_bytes().get(),
                model_selection_memory_budget_bytes = selection_limits.memory_budget_bytes().get(),
                max_concurrent_model_selections = selection_limits.max_concurrent().get(),
                "gateway static model catalog enabled"
            );
            router_with_model_catalog(catalog, max_in_flight, selection_limits, auth, audit)
        }
    };
    let listener = tokio::net::TcpListener::bind(addr).await?;
    let max_connection_duration = Duration::from_secs(max_connection_seconds);
    // Shutdown happens in stages -- connections drain, then each configured
    // JSONL log drains -- and an orchestrator that only budgets for the first
    // stage kills the process mid-drain. Nothing here can read the configured
    // termination grace period, so state the requirement instead of assuming
    // it: an operator comparing this line against their manifest can see the
    // mismatch that would otherwise only show up as a quietly truncated log.
    let configured_logs = u64::from(audit_sink.is_some()) + u64::from(usage_sink.is_some());
    // Computed before the macro, not inside it. `tracing` only evaluates field
    // expressions when the level is enabled, so arithmetic left in the macro
    // runs or does not run depending on `RUST_LOG` -- which would make whether
    // this process starts a function of how it is configured to log. Saturating
    // because `--max-connection-seconds` is an unbounded `u64` and a caller may
    // pass one whose budget is not representable.
    let required_budget = required_shutdown_budget_seconds(max_connection_seconds, configured_logs);
    tracing::info!(
        %addr,
        max_in_flight = max_in_flight.get(),
        max_connection_seconds,
        log_flush_deadline_seconds = DEFAULT_LOG_FLUSH_DEADLINE.as_secs(),
        configured_logs,
        required_shutdown_budget_seconds = required_budget,
        "gateway listening; termination grace period must exceed the required \
         shutdown budget or logs will be killed mid-drain"
    );
    let served = camelid_enterprise_gateway::serve(
        listener,
        router,
        max_connection_duration,
        shutdown_signal(),
    )
    .await;
    // Drained after the connection drain, never before it: records for the
    // requests that finished during graceful shutdown are queued while `serve`
    // is still running, and those are exactly the ones an early flush would
    // miss. Run it even when `serve` failed -- whatever was accepted before
    // the failure is still evidence.
    flush_gateway_log("audit", audit_sink);
    flush_gateway_log("usage", usage_sink);
    served?;
    Ok(())
}

/// Seconds a deployment must allow between SIGTERM and SIGKILL for the gateway
/// to finish shutting down: the connection drain, then one flush deadline per
/// configured JSONL log.
///
/// Saturating rather than checked. This is a diagnostic an operator compares
/// against their orchestrator's grace period, and a nonsensical connection cap
/// should produce a nonsensical-looking budget, not stop the gateway from
/// starting. `u64::MAX` seconds is already unsatisfiable by any grace period,
/// which is the message either way.
fn required_shutdown_budget_seconds(max_connection_seconds: u64, configured_logs: u64) -> u64 {
    max_connection_seconds
        .saturating_add(configured_logs.saturating_mul(DEFAULT_LOG_FLUSH_DEADLINE.as_secs()))
}

/// Drains one JSONL sink at shutdown and reports what happened.
///
/// A timeout is logged at `error` rather than `warn`: the log is evidence, and
/// a short one that nobody was told about is worse than no log at all, because
/// it still looks complete to whatever aggregates it later. The reported count
/// is how far behind the writer was when the wait gave up, not a measurement of
/// what was ultimately lost -- the writer keeps draining until the process
/// exits, so some or all of those records may still land.
fn flush_gateway_log(kind: &'static str, log: Option<Arc<GatewayLog>>) {
    let Some(log) = log else {
        return;
    };
    match log.flush_and_stop(DEFAULT_LOG_FLUSH_DEADLINE) {
        LogFlush::Drained => {
            tracing::info!(
                kind,
                path = %log.path().display(),
                "gateway JSONL log drained at shutdown"
            );
        }
        LogFlush::TimedOut {
            pending_at_deadline,
        } => {
            tracing::error!(
                kind,
                path = %log.path().display(),
                pending_at_deadline,
                deadline_seconds = DEFAULT_LOG_FLUSH_DEADLINE.as_secs(),
                "gateway JSONL log did not finish draining before the flush \
                 deadline; this many accepted records were still queued and \
                 may not survive process exit"
            );
        }
        LogFlush::AlreadyStopped => {}
    }
}

fn create_user(identity_db: &Path, name: &str) -> Result<(), Box<dyn std::error::Error>> {
    let store = SqliteIdentityStore::open(identity_db)?;
    let principal = store.create_user(name)?;
    println!("{principal}");
    Ok(())
}

fn list_users(identity_db: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let store = SqliteIdentityStore::open(identity_db)?;
    for user in store.list_users()? {
        // Id first so the line stays usable with `cut -f1` even though names
        // may contain spaces; a tab keeps them separable when they do.
        println!("{}\t{}", user.principal_id(), user.name());
    }
    Ok(())
}

fn create_organization(identity_db: &Path, name: &str) -> Result<(), Box<dyn std::error::Error>> {
    let store = SqliteIdentityStore::open(identity_db)?;
    let organization = store.create_organization(name)?;
    println!("{organization}");
    Ok(())
}

fn list_organizations(
    identity_db: &Path,
    principal: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let store = SqliteIdentityStore::open(identity_db)?;
    for organization in
        store.organizations_for_principal(&PrincipalId::new(principal.to_string()))?
    {
        println!("{organization}");
    }
    Ok(())
}

fn add_principal_to_organization(
    identity_db: &Path,
    principal: &str,
    organization: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let store = SqliteIdentityStore::open(identity_db)?;
    store.add_principal_to_organization(
        &PrincipalId::new(principal.to_string()),
        &OrganizationId::new(organization.to_string()),
    )?;
    Ok(())
}

fn remove_principal_from_organization(
    identity_db: &Path,
    principal: &str,
    organization: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let store = SqliteIdentityStore::open(identity_db)?;
    store.remove_principal_from_organization(
        &PrincipalId::new(principal.to_string()),
        &OrganizationId::new(organization.to_string()),
    )?;
    Ok(())
}

fn issue_token(
    identity_db: &Path,
    principal: &str,
    organization: Option<&str>,
    expires_in_seconds: Option<NonZeroU64>,
) -> Result<(), Box<dyn std::error::Error>> {
    let store = SqliteIdentityStore::open(identity_db)?;
    let principal = PrincipalId::new(principal.to_string());
    let lifetime = expires_in_seconds.map_or(TokenLifetime::Never, TokenLifetime::expires_in);
    let token = match organization {
        Some(organization) => store.issue_token_for_organization(
            &principal,
            &OrganizationId::new(organization.to_string()),
            lifetime,
        )?,
        None => store.issue_token(&principal, lifetime)?,
    };
    println!("{token}");
    Ok(())
}

fn rotate_token(
    identity_db: &Path,
    token: &str,
    expires_in_seconds: Option<NonZeroU64>,
    no_expiry: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let token = read_secret_arg(token)?;
    let store = SqliteIdentityStore::open(identity_db)?;
    // Decided after the token is in hand, not before. `-` blocks on stdin,
    // which for an interactive paste is unbounded, and `expires_in` measures
    // from the moment it is called: computing the lifetime up front would date
    // the replacement from process start rather than from the rotation.
    //
    // Clap rejects the two flags together, so at most one arm applies. Absent
    // both, the replacement keeps the lifetime the presented token was issued
    // with: the default must not quietly convert a bounded credential into a
    // permanent one.
    let lifetime = match (expires_in_seconds, no_expiry) {
        (Some(seconds), _) => RotationLifetime::Replaced(TokenLifetime::expires_in(seconds)),
        (None, true) => RotationLifetime::Replaced(TokenLifetime::Never),
        (None, false) => RotationLifetime::Preserved,
    };
    let replacement = store.rotate_token(&token, lifetime)?;
    println!("{replacement}");
    Ok(())
}

fn revoke_token(identity_db: &Path, token: &str) -> Result<(), Box<dyn std::error::Error>> {
    let token = read_secret_arg(token)?;
    let store = SqliteIdentityStore::open(identity_db)?;
    store.revoke_token(&token)?;
    Ok(())
}

/// Reads a secret CLI argument, treating the literal value `-` as "read one
/// line from stdin instead". A secret passed directly as an argv value sits
/// in shell history and is briefly visible to other local processes via
/// `ps`; `-` avoids both.
fn read_secret_arg(value: &str) -> Result<String, Box<dyn std::error::Error>> {
    if value != "-" {
        return Ok(value.to_string());
    }
    let mut line = String::new();
    std::io::stdin().read_line(&mut line)?;
    Ok(line.trim_end_matches(['\r', '\n']).to_string())
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

    #[test]
    fn max_in_flight_must_be_nonzero() {
        let result = Cli::try_parse_from([
            "camelid-enterprise-gateway",
            "serve",
            "--upstream",
            "http://127.0.0.1:8181",
            "--max-in-flight",
            "0",
        ]);
        assert!(result.is_err());
    }

    #[test]
    fn max_in_flight_uses_the_pinned_default() {
        let cli = Cli::try_parse_from([
            "camelid-enterprise-gateway",
            "serve",
            "--upstream",
            "http://127.0.0.1:8181",
        ])
        .unwrap();
        let Command::Serve { max_in_flight, .. } = cli.command else {
            panic!("expected the parsed command to be Serve");
        };
        assert_eq!(max_in_flight, DEFAULT_MAX_IN_FLIGHT);
    }

    #[test]
    fn catalog_mode_requires_complete_unambiguous_routing_configuration() {
        let cli = Cli::try_parse_from([
            "camelid-enterprise-gateway",
            "serve",
            "--model-route",
            "alpha=http://127.0.0.1:8181",
            "--model-route",
            "bravo=http://127.0.0.1:8282",
        ])
        .unwrap();
        let Command::Serve {
            upstream,
            model_route,
            max_model_selection_body_bytes,
            model_selection_memory_budget_bytes,
            ..
        } = cli.command
        else {
            panic!("expected the parsed command to be Serve");
        };
        assert_eq!(upstream, None);
        assert_eq!(model_route.len(), 2);
        assert_eq!(
            max_model_selection_body_bytes,
            DEFAULT_MAX_MODEL_SELECTION_BODY_BYTES
        );
        assert_eq!(
            model_selection_memory_budget_bytes,
            DEFAULT_MODEL_SELECTION_MEMORY_BUDGET_BYTES
        );
        let routing = parse_serve_routing(upstream, model_route).unwrap();
        assert!(matches!(routing, ConfiguredServeRouting::Catalog(_)));

        assert!(Cli::try_parse_from([
            "camelid-enterprise-gateway",
            "serve",
            "--upstream",
            "http://127.0.0.1:8181",
            "--model-route",
            "alpha=http://127.0.0.1:8181",
        ])
        .is_err());
        assert!(Cli::try_parse_from(["camelid-enterprise-gateway", "serve"]).is_err());
        assert!(Cli::try_parse_from([
            "camelid-enterprise-gateway",
            "serve",
            "--model-route",
            "alpha=http://127.0.0.1:8181",
            "--model-selection-memory-budget-bytes",
            "0",
        ])
        .is_err());
        assert!(parse_serve_routing(
            None,
            vec![
                "alpha=http://127.0.0.1:8181".into(),
                "alpha=http://127.0.0.1:8282".into()
            ],
        )
        .is_err());
        assert!(parse_model_route("alpha-without-an-origin").is_err());
    }

    #[test]
    fn issue_token_accepts_an_explicit_organization() {
        let cli = Cli::try_parse_from([
            "camelid-enterprise-gateway",
            "issue-token",
            "--identity-db",
            "identity.sqlite",
            "usr_ada",
            "--organization",
            "org_acme",
        ])
        .unwrap();
        let Command::IssueToken {
            principal,
            organization,
            ..
        } = cli.command
        else {
            panic!("expected the parsed command to be IssueToken");
        };
        assert_eq!(principal, "usr_ada");
        assert_eq!(organization.as_deref(), Some("org_acme"));
    }

    #[test]
    fn organization_membership_commands_require_both_ids() {
        let cli = Cli::try_parse_from([
            "camelid-enterprise-gateway",
            "add-principal-to-organization",
            "--identity-db",
            "identity.sqlite",
            "usr_ada",
            "org_acme",
        ])
        .unwrap();
        let Command::AddPrincipalToOrganization {
            principal,
            organization,
            ..
        } = cli.command
        else {
            panic!("expected the parsed command to be AddPrincipalToOrganization");
        };
        assert_eq!(principal, "usr_ada");
        assert_eq!(organization, "org_acme");
    }

    #[test]
    fn org_request_quota_window_defaults_to_sixty_seconds() {
        let cli = Cli::try_parse_from([
            "camelid-enterprise-gateway",
            "serve",
            "--upstream",
            "http://127.0.0.1:8181",
        ])
        .unwrap();
        let Command::Serve {
            org_request_quota,
            org_request_quota_window_seconds,
            ..
        } = cli.command
        else {
            panic!("expected the parsed command to be Serve");
        };
        assert_eq!(org_request_quota, None);
        assert_eq!(
            org_request_quota_window_seconds,
            NonZeroU64::new(60).unwrap()
        );
    }

    #[test]
    fn org_request_quota_must_be_nonzero() {
        let result = Cli::try_parse_from([
            "camelid-enterprise-gateway",
            "serve",
            "--upstream",
            "http://127.0.0.1:8181",
            "--org-request-quota",
            "0",
        ]);
        assert!(result.is_err());
    }

    #[test]
    fn org_request_quota_window_must_be_nonzero() {
        let result = Cli::try_parse_from([
            "camelid-enterprise-gateway",
            "serve",
            "--upstream",
            "http://127.0.0.1:8181",
            "--org-request-quota-window-seconds",
            "0",
        ]);
        assert!(result.is_err());
    }

    #[test]
    fn org_request_quota_requires_identity_db() {
        let result = Cli::try_parse_from([
            "camelid-enterprise-gateway",
            "serve",
            "--upstream",
            "http://127.0.0.1:8181",
            "--org-request-quota",
            "10",
        ]);
        assert!(result.is_err());
    }

    #[test]
    fn usage_log_requires_identity_db() {
        let result = Cli::try_parse_from([
            "camelid-enterprise-gateway",
            "serve",
            "--upstream",
            "http://127.0.0.1:8181",
            "--usage-log",
            "usage.jsonl",
        ]);
        assert!(result.is_err());
    }

    #[test]
    fn shutdown_budget_saturates_instead_of_overflowing() {
        // The connection cap is an unbounded `u64` from the CLI. Computing the
        // budget with `+` overflowed, and because `tracing` only evaluates
        // field expressions when the level is enabled, that made startup
        // succeed or panic depending on `RUST_LOG` -- logging configuration
        // deciding process correctness.
        assert_eq!(required_shutdown_budget_seconds(300, 2), 310);
        assert_eq!(required_shutdown_budget_seconds(300, 0), 300);
        assert_eq!(required_shutdown_budget_seconds(u64::MAX, 2), u64::MAX);
        assert_eq!(
            required_shutdown_budget_seconds(u64::MAX, u64::MAX),
            u64::MAX
        );
        assert_eq!(required_shutdown_budget_seconds(0, u64::MAX), u64::MAX);
    }

    #[test]
    fn gateway_logs_must_use_distinct_destinations() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("gateway.jsonl");
        let error = match open_gateway_logs(&GatewayLogArgs {
            audit_log: Some(path.clone()),
            usage_log: Some(path),
        }) {
            Ok(_) => panic!("matching audit and usage logs must be rejected"),
            Err(error) => error,
        };
        assert_eq!(
            error.to_string(),
            "--audit-log and --usage-log must name different files"
        );
    }

    #[test]
    fn gateway_logs_fail_before_startup_when_the_parent_is_missing() {
        let dir = tempfile::tempdir().unwrap();
        let error = match open_gateway_logs(&GatewayLogArgs {
            audit_log: None,
            usage_log: Some(dir.path().join("missing").join("usage.jsonl")),
        }) {
            Ok(_) => panic!("an unwritable log path must be rejected before binding"),
            Err(error) => error,
        };
        assert!(
            error
                .to_string()
                .contains("The system cannot find the path specified")
                || error.to_string().contains("No such file or directory"),
            "unexpected log-open error: {error}"
        );
    }
}
