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

#[cfg(test)]
mod tests {
    use super::{attribute, Attribution, BODY_LIMIT};
    use axum::body::{to_bytes, Body};
    use axum::http::{header::CONTENT_TYPE, Request, StatusCode};
    use axum::middleware::from_fn_with_state;
    use axum::response::Response;
    use axum::routing::{get, post};
    use axum::{Json, Router};
    use std::path::{Path, PathBuf};
    use std::sync::Arc;
    use tower::ServiceExt;

    const TEST_SHA: &str = "30d77c2608036f8475372ace9ec125ffc5fa16d8d63f0355a08c32c69f4449b7";
    const TEST_HOST: &str = "linux/x86_64 cores=8 simd=avx2+fma";

    fn ctx(receipts: Option<Arc<PathBuf>>) -> Attribution {
        Attribution {
            lane: "deterministic",
            config_sha256: Arc::new(TEST_SHA.to_string()),
            host: Arc::new(TEST_HOST.to_string()),
            receipts,
        }
    }

    /// Attach the middleware with a receiptless context.
    fn attributed(router: Router) -> Router {
        router.layer(from_fn_with_state(ctx(None), attribute))
    }

    fn header(resp: &Response, name: &str) -> String {
        resp.headers()
            .get(name)
            .unwrap_or_else(|| panic!("missing header {name}"))
            .to_str()
            .unwrap()
            .to_string()
    }

    async fn read_body(resp: Response) -> String {
        let bytes = to_bytes(resp.into_body(), BODY_LIMIT).await.unwrap();
        String::from_utf8(bytes.to_vec()).unwrap()
    }

    /// Every response carries lane, config-vector (12-char short form), and
    /// host headers — even on a non-completion path, whose body is left alone.
    #[tokio::test]
    async fn headers_on_every_response_body_untouched_off_completion_paths() {
        let app = attributed(Router::new().route(
            "/v1/models",
            get(|| async { Json(serde_json::json!({ "object": "list", "data": [] })) }),
        ));
        let resp = app
            .oneshot(Request::get("/v1/models").body(Body::empty()).unwrap())
            .await
            .unwrap();

        assert_eq!(header(&resp, "x-camelid-lane"), "deterministic");
        assert_eq!(header(&resp, "x-camelid-config-sha256"), TEST_SHA[..12]);
        assert_eq!(header(&resp, "x-camelid-host"), TEST_HOST);

        let body = read_body(resp).await;
        assert!(
            !body.contains("camelid_lane"),
            "a non-completion body must not be rewritten: {body}"
        );
    }

    /// A JSON completion body gains `camelid_lane` and `camelid_config_sha256`
    /// (matching the header short form), and its original fields survive.
    #[tokio::test]
    async fn completion_json_body_is_attributed_in_place() {
        let app = attributed(Router::new().route(
            "/v1/chat/completions",
            post(|| async { Json(serde_json::json!({ "id": "chatcmpl-1", "choices": [] })) }),
        ));
        let resp = app
            .oneshot(Request::post("/v1/chat/completions").body(Body::empty()).unwrap())
            .await
            .unwrap();

        let header_sha = header(&resp, "x-camelid-config-sha256");
        let body: serde_json::Value = serde_json::from_str(&read_body(resp).await).unwrap();
        assert_eq!(body["camelid_lane"], "deterministic");
        assert_eq!(body["camelid_config_sha256"], TEST_SHA[..12]);
        assert_eq!(body["camelid_config_sha256"].as_str().unwrap(), header_sha.as_str());
        assert_eq!(body["id"], "chatcmpl-1", "original fields must be preserved");
    }

    /// A non-JSON completion body is passed through byte-for-byte; only headers
    /// are added.
    #[tokio::test]
    async fn non_json_completion_body_is_passed_through() {
        let app = attributed(Router::new().route(
            "/v1/completions",
            post(|| async { ([(CONTENT_TYPE, "text/plain")], "hello") }),
        ));
        let resp = app
            .oneshot(Request::post("/v1/completions").body(Body::empty()).unwrap())
            .await
            .unwrap();

        assert_eq!(header(&resp, "x-camelid-host"), TEST_HOST);
        assert_eq!(read_body(resp).await, "hello");
    }

    /// A JSON completion body that is not an object (here, an array) must not be
    /// corrupted by the object-only injection path.
    #[tokio::test]
    async fn non_object_json_completion_body_is_preserved() {
        let app = attributed(Router::new().route(
            "/v1/completions",
            post(|| async { Json(serde_json::json!(["a", "b"])) }),
        ));
        let resp = app
            .oneshot(Request::post("/v1/completions").body(Body::empty()).unwrap())
            .await
            .unwrap();

        assert_eq!(header(&resp, "x-camelid-lane"), "deterministic");
        let body: serde_json::Value = serde_json::from_str(&read_body(resp).await).unwrap();
        assert_eq!(body, serde_json::json!(["a", "b"]));
    }

    /// A completion body larger than the attribution buffer limit cannot be
    /// buffered, so the middleware fails closed with a 500 that still carries
    /// the full attribution header set — the guarantee must not be droppable by
    /// an oversized upstream body.
    #[tokio::test]
    async fn oversized_completion_body_fails_closed_but_stays_attributed() {
        let app = attributed(Router::new().route(
            "/v1/completions",
            post(|| async {
                ([(CONTENT_TYPE, "application/json")], vec![b'x'; BODY_LIMIT + 1])
            }),
        ));
        let resp = app
            .oneshot(Request::post("/v1/completions").body(Body::empty()).unwrap())
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::INTERNAL_SERVER_ERROR);
        assert_eq!(header(&resp, "x-camelid-lane"), "deterministic");
        assert_eq!(header(&resp, "x-camelid-config-sha256"), TEST_SHA[..12]);
        assert_eq!(header(&resp, "x-camelid-host"), TEST_HOST);
    }

    /// With receipts enabled, one JSONL line per request is appended, carrying
    /// method, path, status, lane, the full config digest, and host.
    #[tokio::test]
    async fn receipt_line_records_the_request() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("receipts.jsonl");
        let app = Router::new()
            .route("/v1/models", get(|| async { Json(serde_json::json!({ "object": "list" })) }))
            .layer(from_fn_with_state(ctx(Some(Arc::new(path.clone()))), attribute));

        let resp = app
            .oneshot(Request::get("/v1/models").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        // The receipt is written off the request path on a spawn_blocking task,
        // so poll (bounded) for it rather than assuming it has landed.
        let line = read_receipt(&path).await;
        let receipt: serde_json::Value = serde_json::from_str(line.trim()).unwrap();
        assert_eq!(receipt["method"], "GET");
        assert_eq!(receipt["path"], "/v1/models");
        assert_eq!(receipt["status"], 200);
        assert_eq!(receipt["lane"], "deterministic");
        assert_eq!(receipt["config_sha256"], TEST_SHA);
        assert_eq!(receipt["host"], TEST_HOST);
    }

    async fn read_receipt(path: &Path) -> String {
        for _ in 0..200 {
            if let Ok(contents) = std::fs::read_to_string(path) {
                if !contents.trim().is_empty() {
                    return contents;
                }
            }
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        }
        panic!("receipt was not written within the timeout");
    }
}
