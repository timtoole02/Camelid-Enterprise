//! camelid-enterprise — multi-tenant serving distribution of the Camelid engine.
//!
//! This release ships the **deterministic lane**: whole-generation serialized
//! execution on the engine at a pinned revision under a frozen configuration
//! vector, with deterministic greedy output run to run. Requests the
//! replica cannot serve fail closed (typed 503 from the bounded engine queue);
//! there is no silent demotion to any other execution mode.

use camelid_enterprise::{apply_deterministic, attributed_router, ENGINE_PIN};
use clap::{Parser, Subcommand};
use std::net::SocketAddr;
use std::path::PathBuf;

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
        /// Rayon worker threads (recorded as part of the replica's identity).
        #[arg(long, env = "CAMELID_THREADS")]
        threads: Option<usize>,
        /// Append one JSONL serving receipt per request to this file.
        #[arg(long)]
        serving_receipts: Option<PathBuf>,
    },
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();
    let cli = Cli::parse();
    match cli.command {
        Command::Serve { model, addr, lane, threads, serving_receipts } => {
            serve(model, addr, &lane, threads, serving_receipts).await
        }
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
    eprintln!(
        "[lane] deterministic | engine pin {} | config vector sha256 {} | host {}",
        ENGINE_PIN,
        config.short(),
        host_summary
    );

    let model = model.canonicalize()?;
    let state = camelid::api::AppState::with_configured_threads(threads)
        .with_default_enable_thinking(false)
        .with_models_dir(None);
    let router = attributed_router(state, config.sha256, host_summary, serving_receipts);

    let listener = tokio::net::TcpListener::bind(addr).await?;
    eprintln!("[lane] listening on http://{addr}");

    // Load the startup model through the same HTTP surface a client uses, so
    // startup exercises exactly the serving path.
    let load_addr = addr;
    tokio::spawn(async move {
        let body = serde_json::json!({ "path": load_addr_model_path(&model) }).to_string();
        for attempt in 0..60u32 {
            tokio::time::sleep(std::time::Duration::from_millis(250 * (attempt.min(4) as u64 + 1))).await;
            let result = http_post_json(load_addr, "/api/models/load", &body).await;
            match result {
                Ok((status, _)) if status == 200 => {
                    eprintln!("[lane] model loaded; replica ready");
                    return;
                }
                Ok((status, response)) => {
                    eprintln!("[lane] FATAL: model load failed (HTTP {status}): {response}");
                    std::process::exit(1);
                }
                Err(_) if attempt < 59 => continue,
                Err(err) => {
                    eprintln!("[lane] FATAL: could not reach own listener to load model: {err}");
                    std::process::exit(1);
                }
            }
        }
    });

    axum::serve(listener, router)
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

fn load_addr_model_path(model: &std::path::Path) -> String {
    model.to_string_lossy().into_owned()
}

async fn http_post_json(
    addr: SocketAddr,
    path: &str,
    body: &str,
) -> Result<(u16, String), std::io::Error> {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    let mut stream = tokio::net::TcpStream::connect(addr).await?;
    let request = format!(
        "POST {path} HTTP/1.1\r\nHost: {addr}\r\nContent-Type: application/json\r\n\
         Content-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    );
    stream.write_all(request.as_bytes()).await?;
    let mut response = Vec::new();
    stream.read_to_end(&mut response).await?;
    let text = String::from_utf8_lossy(&response);
    let status = text
        .split_whitespace()
        .nth(1)
        .and_then(|s| s.parse::<u16>().ok())
        .unwrap_or(0);
    let payload = text
        .split_once("\r\n\r\n")
        .map(|(_, b)| b.trim().to_string())
        .unwrap_or_default();
    Ok((status, payload))
}
