use camelid_enterprise_gateway::{router, GatewayAuth, UpstreamOrigin};
use clap::{Parser, Subcommand};
use identity::{PrincipalId, SqliteIdentityStore};
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::sync::Arc;

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
        /// Path to a local identity database. When set, every request must
        /// carry `Authorization: Bearer <token>` resolving to a known
        /// principal, or the gateway rejects it before it ever reaches a
        /// replica. When omitted, the gateway forwards every request
        /// unauthenticated, unchanged from prior releases.
        #[arg(long, env = "CAMELID_GATEWAY_IDENTITY_DB")]
        identity_db: Option<PathBuf>,
    },
    /// Create a user in the identity database and print its opaque principal id.
    CreateUser {
        #[arg(long, env = "CAMELID_GATEWAY_IDENTITY_DB")]
        identity_db: PathBuf,
        /// Display label for the user. Never used as a lookup key.
        name: String,
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
            identity_db,
        } => serve(&upstream, addr, identity_db).await,
        Command::CreateUser { identity_db, name } => create_user(&identity_db, &name),
        Command::IssueToken {
            identity_db,
            principal,
        } => issue_token(&identity_db, &principal),
        Command::RevokeToken { identity_db, token } => revoke_token(&identity_db, &token),
    }
}

async fn serve(
    upstream: &str,
    addr: SocketAddr,
    identity_db: Option<PathBuf>,
) -> Result<(), Box<dyn std::error::Error>> {
    let upstream = UpstreamOrigin::parse(upstream)?;
    let auth = match identity_db {
        Some(path) => {
            tracing::info!(path = %path.display(), "gateway auth enforcement enabled");
            tracing::warn!(
                "bearer tokens are sent as plain HTTP headers; this gateway does not \
                 terminate TLS. Put a TLS-terminating ingress/reverse proxy (or mTLS) in \
                 front of it, or restrict --addr to a trusted network, or a captured \
                 token can be replayed indefinitely (tokens do not expire yet)."
            );
            GatewayAuth::RequireToken(Arc::new(SqliteIdentityStore::open(&path)?))
        }
        None => {
            tracing::warn!(
                "no --identity-db configured; the gateway is forwarding every request unauthenticated"
            );
            GatewayAuth::Disabled
        }
    };
    let listener = tokio::net::TcpListener::bind(addr).await?;
    tracing::info!(%addr, "gateway listening");
    axum::serve(listener, router(upstream, auth))
        .with_graceful_shutdown(shutdown_signal())
        .await?;
    Ok(())
}

fn create_user(identity_db: &Path, name: &str) -> Result<(), Box<dyn std::error::Error>> {
    let store = SqliteIdentityStore::open(identity_db)?;
    let principal = store.create_user(name)?;
    println!("{principal}");
    Ok(())
}

fn issue_token(identity_db: &Path, principal: &str) -> Result<(), Box<dyn std::error::Error>> {
    let store = SqliteIdentityStore::open(identity_db)?;
    let token = store.issue_token(&PrincipalId::new(principal.to_string()))?;
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
