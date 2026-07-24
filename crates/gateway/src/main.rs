use camelid_enterprise_gateway::{
    router_with_max_in_flight, UpstreamOrigin, DEFAULT_MAX_IN_FLIGHT,
};
use clap::{Parser, Subcommand};
use std::net::SocketAddr;

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
            value_parser = parse_positive_usize,
            env = "CAMELID_GATEWAY_MAX_IN_FLIGHT"
        )]
        max_in_flight: usize,
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
        } => serve(&upstream, addr, max_in_flight).await,
    }
}

fn parse_positive_usize(value: &str) -> Result<usize, String> {
    value
        .parse::<usize>()
        .ok()
        .filter(|parsed| *parsed > 0)
        .ok_or_else(|| "value must be a positive integer".to_string())
}

async fn serve(
    upstream: &str,
    addr: SocketAddr,
    max_in_flight: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let upstream = UpstreamOrigin::parse(upstream)?;
    let listener = tokio::net::TcpListener::bind(addr).await?;
    tracing::info!(%addr, max_in_flight, "gateway listening");
    axum::serve(listener, router_with_max_in_flight(upstream, max_in_flight))
        .with_graceful_shutdown(shutdown_signal())
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
