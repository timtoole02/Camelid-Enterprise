use camelid_enterprise_gateway::{
    router_with_options, GatewayAuth, OrgQuota, UpstreamOrigin, DEFAULT_MAX_CONNECTION_DURATION,
    DEFAULT_MAX_IN_FLIGHT,
};
use clap::{Parser, Subcommand};
use identity::{OrganizationId, PrincipalId, SqliteIdentityStore};
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
    /// Start the transparent gateway in front of one replica or cluster Service.
    Serve {
        /// Replica origin, including the http:// scheme and optional port.
        #[arg(long, env = "CAMELID_GATEWAY_UPSTREAM")]
        upstream: String,
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
        /// correlation id, the resolved principal (or null), method, path, and
        /// status. Join it to a replica's serving receipts on `request_id` to
        /// see which deterministic configuration served each caller's request.
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
            addr,
            max_in_flight,
            max_connection_seconds,
            identity_db,
            audit_log,
            usage_log,
            org_request_quota,
            org_request_quota_window_seconds,
        } => {
            serve(
                &upstream,
                addr,
                max_in_flight,
                max_connection_seconds,
                identity_db,
                GatewayLogArgs {
                    audit_log,
                    usage_log,
                },
                OrgQuotaArgs {
                    limit: org_request_quota,
                    window_seconds: org_request_quota_window_seconds,
                },
            )
            .await
        }
        Command::CreateUser { identity_db, name } => create_user(&identity_db, &name),
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
        } => issue_token(&identity_db, &principal, organization.as_deref()),
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

async fn serve(
    upstream: &str,
    addr: SocketAddr,
    max_in_flight: NonZeroUsize,
    max_connection_seconds: u64,
    identity_db: Option<PathBuf>,
    logs: GatewayLogArgs,
    org_request_quota: OrgQuotaArgs,
) -> Result<(), Box<dyn std::error::Error>> {
    let upstream = UpstreamOrigin::parse(upstream)?;
    let usage = match logs.usage_log {
        Some(path) => {
            tracing::info!(path = %path.display(), "gateway transport usage log enabled");
            Some(Arc::new(path))
        }
        None => None,
    };
    let auth = match identity_db {
        Some(path) => {
            tracing::info!(path = %path.display(), "gateway auth enforcement enabled");
            tracing::warn!(
                "bearer tokens are sent as plain HTTP headers; this gateway does not \
                 terminate TLS. Put a TLS-terminating ingress/reverse proxy (or mTLS) in \
                 front of it, or restrict --addr to a trusted network, or a captured \
                 token can be replayed indefinitely (tokens do not expire yet)."
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
            if usage.is_some() {
                return Err("--usage-log requires --identity-db".into());
            }
            tracing::warn!(
                "no --identity-db configured; the gateway is forwarding every request unauthenticated"
            );
            GatewayAuth::Disabled
        }
    };
    let audit = match logs.audit_log {
        Some(path) => {
            tracing::info!(path = %path.display(), "gateway request audit log enabled");
            Some(Arc::new(path))
        }
        None => None,
    };
    let listener = tokio::net::TcpListener::bind(addr).await?;
    let max_connection_duration = Duration::from_secs(max_connection_seconds);
    tracing::info!(
        %addr,
        max_in_flight = max_in_flight.get(),
        max_connection_seconds,
        "gateway listening"
    );
    camelid_enterprise_gateway::serve(
        listener,
        router_with_options(upstream, max_in_flight, auth, audit),
        max_connection_duration,
        shutdown_signal(),
    )
    .await?;
    Ok(())
}

fn create_user(identity_db: &Path, name: &str) -> Result<(), Box<dyn std::error::Error>> {
    let store = SqliteIdentityStore::open(identity_db)?;
    let principal = store.create_user(name)?;
    println!("{principal}");
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
) -> Result<(), Box<dyn std::error::Error>> {
    let store = SqliteIdentityStore::open(identity_db)?;
    let principal = PrincipalId::new(principal.to_string());
    let token = match organization {
        Some(organization) => store.issue_token_for_organization(
            &principal,
            &OrganizationId::new(organization.to_string()),
        )?,
        None => store.issue_token(&principal)?,
    };
    println!("{token}");
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
}
