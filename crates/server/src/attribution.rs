//! Lane attribution: every response is attributable to the lane that produced it.
//!
//! Three locations, so no consumer misses it:
//! - `x-camelid-lane` / `x-camelid-config-sha256` / `x-camelid-host` headers on
//!   every response (including streams);
//! - `camelid_lane` / `camelid_config_sha256` fields injected into non-streaming
//!   completion JSON bodies;
//! - an optional append-only serving-receipt log (JSONL), one line per request,
//!   carrying the lane, config vector, and host identity.
//!
//! Host identity is attributed but deliberately NOT folded into the config
//! vector hash: the config vector identifies a *configuration* (so two pools on
//! different hardware classes stay comparable by hash), while `x-camelid-host`
//! and the receipt's `host` field carry the hardware class the guarantee is
//! scoped to. Config identity and host identity are different claims.

use axum::{
    body::{to_bytes, Body},
    extract::{Request, State},
    http::header::{CONTENT_LENGTH, CONTENT_TYPE},
    http::{HeaderValue, StatusCode},
    middleware::Next,
    response::Response,
};
use std::io::Write;
use std::path::PathBuf;
use std::sync::Arc;

const BODY_LIMIT: usize = 64 * 1024 * 1024;

#[derive(Clone)]
pub struct Attribution {
    pub lane: &'static str,
    pub config_sha256: Arc<String>,
    /// Hardware-class identity for this replica (the same string the startup
    /// banner prints, e.g. `linux/x86_64 cores=16 simd=...`). Attributed on
    /// every response and receipt; never an input to `config_sha256`.
    pub host: Arc<String>,
    pub receipts: Option<Arc<PathBuf>>,
}

fn is_completion_path(path: &str) -> bool {
    matches!(path, "/v1/chat/completions" | "/v1/completions")
}

pub async fn attribute(
    State(ctx): State<Attribution>,
    req: Request,
    next: Next,
) -> Response {
    let path = req.uri().path().to_string();
    let method = req.method().to_string();
    let mut resp = next.run(req).await;

    let short = &ctx.config_sha256[..12];
    resp.headers_mut().insert(
        "x-camelid-lane",
        HeaderValue::from_static(ctx.lane),
    );
    if let Ok(v) = HeaderValue::from_str(short) {
        resp.headers_mut().insert("x-camelid-config-sha256", v);
    }
    if let Ok(v) = HeaderValue::from_str(ctx.host.as_str()) {
        resp.headers_mut().insert("x-camelid-host", v);
    }

    let is_json = resp
        .headers()
        .get(CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .map(|v| v.starts_with("application/json"))
        .unwrap_or(false);

    if is_completion_path(&path) && is_json {
        let (mut parts, body) = resp.into_parts();
        match to_bytes(body, BODY_LIMIT).await {
            Ok(bytes) => {
                let rewritten = match serde_json::from_slice::<serde_json::Value>(&bytes) {
                    Ok(mut value) => {
                        if let Some(obj) = value.as_object_mut() {
                            obj.insert("camelid_lane".into(), ctx.lane.into());
                            obj.insert("camelid_config_sha256".into(), short.into());
                        }
                        serde_json::to_vec(&value).unwrap_or_else(|_| bytes.to_vec())
                    }
                    Err(_) => bytes.to_vec(),
                };
                parts.headers.remove(CONTENT_LENGTH);
                resp = Response::from_parts(parts, Body::from(rewritten));
            }
            Err(_) => {
                // Attribution must not corrupt a response it could not buffer;
                // fail the request rather than emit an unattributed body.
                let mut failed = Response::new(Body::from(
                    r#"{"error":{"message":"response exceeded the attribution buffer limit","type":"server_error"}}"#,
                ));
                *failed.status_mut() = StatusCode::INTERNAL_SERVER_ERROR;
                // The response we manufacture to PROTECT attribution must itself
                // carry the full attribution set — lane, config vector, and host.
                let headers = failed.headers_mut();
                headers.insert("x-camelid-lane", HeaderValue::from_static(ctx.lane));
                if let Ok(v) = HeaderValue::from_str(short) {
                    headers.insert("x-camelid-config-sha256", v);
                }
                if let Ok(v) = HeaderValue::from_str(ctx.host.as_str()) {
                    headers.insert("x-camelid-host", v);
                }
                resp = failed;
            }
        }
    }

    if let Some(log) = &ctx.receipts {
        let line = serde_json::json!({
            "ts": std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs_f64())
                .unwrap_or(0.0),
            "method": method,
            "path": path,
            "status": resp.status().as_u16(),
            "lane": ctx.lane,
            "config_sha256": ctx.config_sha256.as_str(),
            "host": ctx.host.as_str(),
        });
        let log = Arc::clone(log);
        // Best-effort, off the request path's async context.
        tokio::task::spawn_blocking(move || {
            if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(&*log) {
                let _ = writeln!(f, "{line}");
            }
        });
    }

    resp
}
