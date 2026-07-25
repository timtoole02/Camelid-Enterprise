use camelid_enterprise_gateway::{
    router_with_max_in_flight, UpstreamOrigin, DEFAULT_MAX_CONNECTION_DURATION,
    DEFAULT_MAX_IN_FLIGHT,
};
use clap::{Parser, Subcommand};
use std::net::SocketAddr;
use std::num::NonZeroUsize;
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
        } => serve(&upstream, addr, max_in_flight, max_connection_seconds).await,
    }
}

async fn serve(
    upstream: &str,
    addr: SocketAddr,
    max_in_flight: NonZeroUsize,
    max_connection_seconds: u64,
) -> Result<(), Box<dyn std::error::Error>> {
    let upstream = UpstreamOrigin::parse(upstream)?;
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
        router_with_max_in_flight(upstream, max_in_flight),
        max_connection_duration,
        shutdown_signal(),
    )
    .await?;
    Ok(())
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
        let Command::Serve { max_in_flight, .. } = cli.command;
        assert_eq!(max_in_flight, DEFAULT_MAX_IN_FLIGHT);
    }
}
