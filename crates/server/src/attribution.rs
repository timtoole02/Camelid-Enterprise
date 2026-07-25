//! Lane attribution: every response is attributable to the lane that produced
//! it, to the weights it was produced from, and to the machine state it was
//! produced on.
//!
//! Three locations, so no consumer misses it:
//! - `x-camelid-lane` / `x-camelid-config-sha256` / `x-camelid-admission-sha256`
//!   / `x-camelid-model-sha256` / `x-camelid-host` / `x-camelid-worker-threads`
//!   headers on every response (including streams);
//! - `camelid_lane` / `camelid_config_sha256` / `camelid_admission_sha256` /
//!   `camelid_model_sha256` / `camelid_host` / `camelid_worker_threads` fields
//!   injected into non-streaming completion JSON bodies;
//! - an optional append-only serving-receipt log (JSONL), one line per request,
//!   carrying the same facts as `lane`, `config_sha256`, `admission_sha256`,
//!   `model_sha256`, `host` and `worker_threads`.
//!
//! These are separate fields because they are separate claims, and none stands
//! in for another.
//!
//! **Config identity** is the vector this replica wrote, at the engine revision
//! it wrote it for. It is a compile-time constant, byte-identical across every
//! conforming replica, which is exactly what makes it worth comparing.
//!
//! **Admission identity** is what this replica would have *refused*, which the
//! configuration digest cannot express: under deny-by-default the allow list is
//! the admission surface, so a build that gained one permit could start under a
//! variable that moves the forward pass and publish everything a clean replica
//! publishes. It is a compile-time constant for the same reason config identity
//! is, and it is a second field rather than more bytes in the first so that a
//! client comparing two replicas learns *which* of the two claims moved.
//!
//! **Host identity** is attributed but deliberately NOT folded into the config
//! vector hash: the config vector identifies a *configuration* (so two pools on
//! different hardware classes stay comparable by hash), while `x-camelid-host`,
//! `camelid_host` and the receipt's `host` field carry the hardware class the
//! guarantee is scoped to.
//!
//! It is on all three surfaces and not only on the header, which is a correction
//! rather than a flourish. The hardware class is the axis the reproducibility
//! scope is *stated over*: two replicas serving the same weights under the same
//! configuration at the same pool width can emit different tokens when one has a
//! SIMD feature the other lacks and the engine routes through a different
//! kernel. With `camelid_host` absent, those two replicas' completion bodies
//! were identical over every attribution field, so a client verifying
//! attribution from the body — which is what the README demonstrates, and the
//! only surface a response-body-shaped client library exposes without extra
//! work — could not tell them apart on the one axis that had moved.
//!
//! **Model identity** is the content of the weights, and it is here because it
//! is the one input that changes every token while changing nothing else. The
//! engine's own `model` field is the GGUF's `general.name` metadata string, so
//! two files sharing a name and differing in every tensor are indistinguishable
//! through it, and the configuration digest is a compile-time constant that
//! cannot see an operator's `--model` at all. Without this field two replicas
//! serving different weights publish byte-identical attribution while emitting
//! unrelated token streams — the precise claim this surface exists to make
//! checkable.
//!
//! **Worker-pool width** is what the process actually came up as, and it is in
//! neither digest for a reason worth stating: fold a machine property into a
//! digest and it stops being constant across conforming replicas, so an 8-core
//! and a 16-core replica running an identical audited configuration would
//! publish different config digests and every consumer would need the
//! per-hardware table the field exists to eliminate. Two questions, two fields,
//! neither impersonating the other.
//!
//! The honest consequence of that split, stated in the open rather than left to
//! be discovered: two replicas may publish the same `config_sha256` and still
//! emit different tokens. A client that cares compares all of these fields, not
//! one of them.

use axum::{
    body::{to_bytes, Body},
    extract::{Request, State},
    http::header::{CONTENT_LENGTH, CONTENT_TYPE},
    http::{HeaderValue, StatusCode},
    middleware::Next,
    response::Response,
};
use sha2::{Digest, Sha256};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;

const BODY_LIMIT: usize = 64 * 1024 * 1024;

/// Read size for the model digest. Large enough that the syscall is not the
/// bottleneck, small enough to be irrelevant next to the weights themselves.
const DIGEST_CHUNK: usize = 1024 * 1024;

/// Number of leading hex characters published on the wire.
///
/// 48 bits: a comparison and audit handle, not a cryptographic binding. It is
/// short because it rides on every response including every streamed one, and
/// the operation it serves is equality — "is this the replica I audited?", "are
/// these two replicas the same?" — which a prefix answers. Anything that needs
/// the real thing reads the receipt, which carries all 64.
const SHORT_HEX: usize = 12;

/// What a file was, well enough to notice it being swapped underneath us.
///
/// Length plus, on unix, the device and inode: replacing a file by rename gives
/// the path a new inode, so this changes even when the replacement is the same
/// size. Off unix it degrades to length alone, which is weaker and is the reason
/// this is a hardening rather than a control.
#[derive(Clone, PartialEq, Eq)]
struct FileFingerprint {
    len: u64,
    #[cfg(unix)]
    dev: u64,
    #[cfg(unix)]
    ino: u64,
}

impl FileFingerprint {
    fn of(meta: &std::fs::Metadata) -> Self {
        #[cfg(unix)]
        {
            use std::os::unix::fs::MetadataExt;
            Self { len: meta.len(), dev: meta.dev(), ino: meta.ino() }
        }
        #[cfg(not(unix))]
        {
            Self { len: meta.len() }
        }
    }
}

/// The weights this replica serves, identified by content.
///
/// A path is not an identity — two replicas can both serve `/models/model.gguf`
/// and mean different files — and neither is the GGUF's `general.name`, which is
/// metadata an editor can copy between files without touching a tensor. This is
/// a plain SHA-256 over the file's bytes, so it moves when and only when the
/// weights move, and an operator verifies it with `shasum -a 256`.
///
/// Whole file, not the tensor block alone. Tensor-only would be smaller and is
/// wrong in the direction that matters: the metadata block carries the tokenizer
/// vocabulary, the rotary base and scaling, and the normalization epsilon, all
/// of which change output. A tensor-only digest would publish one identity for
/// two files that emit different token streams — this field's purpose,
/// inverted — and it would need a canonical serialization no standard tool can
/// reproduce. Whole-file over-triggers only on inert metadata edits, and that is
/// the safe direction of error: a false "these differ" prompts a look, a false
/// "these are the same" is an undetected substitution.
#[derive(Clone)]
pub struct ModelIdentity {
    /// Full 64 hex, for receipts — an audit line carries the whole value so it
    /// can be checked against a file without a second lookup.
    sha256: Arc<String>,
    /// Leading [`SHORT_HEX`] characters, pre-rendered so stamping a response is
    /// an infallible refcount bump rather than a parse on the request path.
    short: HeaderValue,
    /// What the digested file was, captured from the handle that was digested.
    fingerprint: FileFingerprint,
}

impl ModelIdentity {
    /// Digest the GGUF at `path`.
    ///
    /// Fails closed. A replica that cannot read its own weights well enough to
    /// identify them has nothing to publish about what produced its tokens, and
    /// it is about to fail the load anyway — failing here says why in one line
    /// rather than surfacing later as a load error with no context.
    pub fn of_file(path: &Path) -> Result<Self, String> {
        let unreadable = |err: std::io::Error| {
            format!("cannot read the model at {} to identify it: {err}", path.display())
        };
        let mut file = std::fs::File::open(path).map_err(unreadable)?;
        // From the open handle, not from the path: the fingerprint has to
        // describe the bytes actually hashed, and a second `stat` by path could
        // already be describing a different file.
        let fingerprint = FileFingerprint::of(&file.metadata().map_err(unreadable)?);
        let mut hasher = Sha256::new();
        let mut buf = vec![0u8; DIGEST_CHUNK];
        loop {
            let read = file.read(&mut buf).map_err(unreadable)?;
            if read == 0 {
                break;
            }
            hasher.update(&buf[..read]);
        }
        Self::new(format!("{:x}", hasher.finalize()), fingerprint)
    }

    fn new(sha256: String, fingerprint: FileFingerprint) -> Result<Self, String> {
        // Infallible for hex, and checked rather than assumed, because the
        // alternative to an error here is a response that silently carries one
        // fewer identity field than the one next to it.
        let short = HeaderValue::from_str(sha256.get(..SHORT_HEX).unwrap_or_default())
            .map_err(|_| "the model digest does not encode as a header value".to_string())?;
        Ok(Self { sha256: Arc::new(sha256), short, fingerprint })
    }

    /// The published short form.
    pub fn short(&self) -> &str {
        &self.sha256[..SHORT_HEX]
    }

    /// Fail if the file at `path` is no longer the one that was digested.
    ///
    /// Called after the startup load returns. The engine loads by path string
    /// rather than by handle, so between hashing and loading there is a window
    /// in which a rename could hand the loader different bytes than the ones
    /// this replica is about to publish a digest for. This closes it.
    ///
    /// A hardening, not the primary control: the primary controls are a
    /// read-only model mount and a served surface with no route, and no request
    /// body, that can change which weights answer.
    ///
    /// What it does not cover, stated rather than left to be inferred, and
    /// stated at the size it actually is. It compares length and, on unix,
    /// device and inode, so a **same-length, in-place rewrite** of the file
    /// passes it. Replacement by rename is caught, and that is the accident a
    /// build pipeline actually produces; `dd conv=notrunc` over a live GGUF is
    /// not.
    ///
    /// And the exposure is not the interval this check sits in. It is tempting
    /// to describe the gap as a hashing-versus-loading window, which would make
    /// a rewrite outside that window harmless; it is not, because the load does
    /// not finish reading the file. At the pinned engine, tensors are bound as
    /// descriptors and their rows are read from a cached file handle inside the
    /// forward pass, so the GGUF is a live input for as long as the process
    /// serves. A same-length rewrite an hour into serving changes the tokens
    /// while `x-camelid-model-sha256`, `camelid_model_sha256` and the receipt's
    /// `model_sha256` all go on describing the bytes hashed at startup — and a
    /// second replica legitimately serving those original bytes is then
    /// byte-indistinguishable from this one. One check at startup cannot close
    /// that; against a writer who can reach the file, the answer is the
    /// read-only mount, which both deployments in this tree provide.
    pub fn verify_unchanged(&self, path: &Path) -> Result<(), String> {
        let meta = std::fs::metadata(path).map_err(|err| {
            format!("cannot re-check the model at {} after loading it: {err}", path.display())
        })?;
        if FileFingerprint::of(&meta) == self.fingerprint {
            return Ok(());
        }
        Err(format!(
            "the model at {} changed between being identified and being loaded. This replica \
             publishes {} on every response, and can no longer vouch that it describes the bytes \
             that were loaded, so it refuses to serve.",
            path.display(),
            self.short()
        ))
    }
}

/// The resolved width of the process-global data-parallel worker pool.
///
/// "Resolved" is the whole point: this is the width the process came up with,
/// read back from the pool, never the `Option` the CLI was handed. A replica
/// that echoed the flag would publish a number that is wrong in exactly the case
/// anyone would check it — when the flag was absent.
///
/// It is the width of the *global* pool. The engine derives narrower or wider
/// phase-specific pools from it at first generation, so this is not a promise
/// that every kernel runs at this width; it is the one number those are derived
/// from, and the number that differs between two replicas started differently.
#[derive(Clone)]
pub struct WorkerThreads {
    count: usize,
    /// Pre-rendered for the same reason as the model digest's short form.
    header: HeaderValue,
}

impl WorkerThreads {
    pub fn resolved(count: usize) -> Self {
        let header = HeaderValue::from_str(&count.to_string())
            .expect("a decimal integer is always a valid header value");
        Self { count, header }
    }

    pub fn count(&self) -> usize {
        self.count
    }
}

#[derive(Clone)]
pub struct Attribution {
    pub lane: &'static str,
    pub config_sha256: Arc<String>,
    /// Identity of the admission policy this build enforces — what it would have
    /// refused, as against what it applied. Compile-time constant, like
    /// `config_sha256`, and deliberately not fused with it: fusing would move a
    /// digest documented as "this exact vector" for a change that alters no
    /// applied value, and would leave a client unable to tell which claim moved.
    pub admission_sha256: Arc<String>,
    /// Content identity of the weights, fixed before the listener bound.
    pub model: ModelIdentity,
    /// Hardware-class identity for this replica (the same string the startup
    /// banner prints, e.g. `linux/x86_64 cores=16 simd=...`). Attributed on
    /// every response and receipt; never an input to `config_sha256`.
    pub host: Arc<String>,
    /// Resolved worker-pool width. Machine state, like `host`, and published for
    /// the same reason: it is not in any digest and cannot be recovered from
    /// one.
    pub workers: WorkerThreads,
    pub receipts: Option<Arc<PathBuf>>,
}

impl Attribution {
    fn short(&self) -> &str {
        &self.config_sha256[..SHORT_HEX]
    }

    fn admission_short(&self) -> &str {
        &self.admission_sha256[..SHORT_HEX]
    }

    /// Stamp the attribution headers.
    ///
    /// Called on the response the handler produced *and* on the one this
    /// middleware synthesizes when a body cannot be buffered, because a failure
    /// this middleware causes is still a response this replica emitted and still
    /// has to say so. One function so the two cannot drift — the failure path
    /// dropping a field is precisely the kind of gap that survives review.
    fn stamp(&self, resp: &mut Response) {
        let headers = resp.headers_mut();
        headers.insert("x-camelid-lane", HeaderValue::from_static(self.lane));
        if let Ok(value) = HeaderValue::from_str(self.short()) {
            headers.insert("x-camelid-config-sha256", value);
        }
        if let Ok(value) = HeaderValue::from_str(self.admission_short()) {
            headers.insert("x-camelid-admission-sha256", value);
        }
        if let Ok(value) = HeaderValue::from_str(self.host.as_str()) {
            headers.insert("x-camelid-host", value);
        }
        // No fallible conversion on these two: both were rendered and validated
        // at startup, so there is no failure left to swallow.
        headers.insert("x-camelid-model-sha256", self.model.short.clone());
        headers.insert("x-camelid-worker-threads", self.workers.header.clone());
    }

    /// Add the attribution fields to a completion body.
    ///
    /// The digests go in at their published short length so a caller comparing
    /// the body against the header is comparing the same string, and the thread
    /// count goes in as a JSON number rather than a string because it is one.
    ///
    /// Every field the headers carry, and for the same reason the three surfaces
    /// exist: a consumer that reads only one of them must not be missing a claim
    /// the others make. `camelid_host` is here because the hardware class is the
    /// axis the reproducibility scope is stated over, and nothing else in a
    /// completion body varies with it — the engine's own metadata is derived
    /// from the model and the configuration, and the timings are noise.
    fn inject(&self, value: &mut serde_json::Value) {
        let Some(obj) = value.as_object_mut() else {
            return;
        };
        obj.insert("camelid_lane".into(), self.lane.into());
        obj.insert("camelid_config_sha256".into(), self.short().into());
        obj.insert("camelid_admission_sha256".into(), self.admission_short().into());
        obj.insert("camelid_model_sha256".into(), self.model.short().into());
        obj.insert("camelid_host".into(), self.host.as_str().into());
        obj.insert("camelid_worker_threads".into(), self.workers.count.into());
    }

    /// One serving-receipt line.
    ///
    /// The identity fields are repeated on every line even though none of them
    /// can change while the process is up. A receipt whose single line does not
    /// fully identify the replica that produced it is not a receipt — it is a
    /// line that has to be joined against a log nobody kept. The digests are
    /// full length here, because this is the copy an audit reads.
    fn receipt(&self, method: &str, path: &str, status: u16, ts: f64) -> serde_json::Value {
        serde_json::json!({
            "ts": ts,
            "method": method,
            "path": path,
            "status": status,
            "lane": self.lane,
            "config_sha256": self.config_sha256.as_str(),
            "admission_sha256": self.admission_sha256.as_str(),
            "model_sha256": self.model.sha256.as_str(),
            "host": self.host.as_str(),
            "worker_threads": self.workers.count,
        })
    }
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

    ctx.stamp(&mut resp);

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
                        ctx.inject(&mut value);
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
                // carry the full attribution set and be a well-formed, labeled
                // JSON error body.
                failed
                    .headers_mut()
                    .insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
                ctx.stamp(&mut failed);
                resp = failed;
            }
        }
    }

    if let Some(log) = &ctx.receipts {
        let ts = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs_f64())
            .unwrap_or(0.0);
        let line = ctx.receipt(&method, &path, resp.status().as_u16(), ts);
        let log = Arc::clone(log);
        // Best-effort, off the request path's async context. Write the whole
        // record — JSON plus newline — in a single `write_all` so concurrent
        // appends from simultaneous requests cannot interleave into a corrupt
        // line. Detached: a receipt may land after the client has its bytes, and
        // on a stop the drain does not wait for it.
        tokio::task::spawn_blocking(move || {
            if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(&*log) {
                let mut record = line.to_string();
                record.push('\n');
                let _ = f.write_all(record.as_bytes());
            }
        });
    }

    resp
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::to_bytes;
    use axum::http::{header::CONTENT_TYPE, Request, StatusCode};
    use axum::middleware::from_fn_with_state;
    use axum::response::Response;
    use axum::routing::{get, post};
    use axum::{Json, Router};
    use std::path::Path;
    use tower::ServiceExt;

    const TEST_SHA: &str = "30d77c2608036f8475372ace9ec125ffc5fa16d8d63f0355a08c32c69f4449b7";
    /// Distinct from every other digest here, so no assertion can pass by
    /// reading a neighbouring field.
    const TEST_ADMISSION_SHA: &str =
        "45121fb83fef631f8464c32dada6100b23f0a0af80347031f812803ee9ec2a09";
    const OTHER_ADMISSION_SHA: &str =
        "aabbccddeeff00112233445566778899aabbccddeeff00112233445566778899";
    const TEST_HOST: &str = "linux/x86_64 cores=8 simd=avx2+fma";
    /// A model digest that is not the configuration digest, so an assertion
    /// cannot pass by reading the wrong field.
    const TEST_MODEL_SHA: &str =
        "9f2b1c7a4e0d63558a1e2f4c7b8d90a3c5e6f70819a2b3c4d5e6f708192a3b4c";
    const OTHER_MODEL_SHA: &str =
        "1122334455667788990011223344556677889900112233445566778899001122";
    const TEST_THREADS: usize = 6;
    const SSE_BODY: &str =
        "data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\ndata: [DONE]\n\n";

    /// A model identity with no file behind it, for the middleware tests. The
    /// digest-of-a-real-file path is exercised separately, against a real file.
    fn model(sha: &str) -> ModelIdentity {
        ModelIdentity::new(sha.to_string(), FileFingerprint::of(&std::fs::metadata(".").unwrap()))
            .unwrap()
    }

    fn ctx(receipts: Option<Arc<PathBuf>>) -> Attribution {
        ctx_for(TEST_MODEL_SHA, receipts)
    }

    fn ctx_for(model_sha: &str, receipts: Option<Arc<PathBuf>>) -> Attribution {
        ctx_full(model_sha, TEST_ADMISSION_SHA, receipts)
    }

    fn ctx_full(
        model_sha: &str,
        admission_sha: &str,
        receipts: Option<Arc<PathBuf>>,
    ) -> Attribution {
        Attribution {
            lane: "deterministic",
            config_sha256: Arc::new(TEST_SHA.to_string()),
            admission_sha256: Arc::new(admission_sha.to_string()),
            model: model(model_sha),
            host: Arc::new(TEST_HOST.to_string()),
            workers: WorkerThreads::resolved(TEST_THREADS),
            receipts,
        }
    }

    /// The same replica on a different hardware class: only `host` moves.
    fn ctx_on_host(host: &str, receipts: Option<Arc<PathBuf>>) -> Attribution {
        Attribution {
            host: Arc::new(host.to_string()),
            ..ctx_full(TEST_MODEL_SHA, TEST_ADMISSION_SHA, receipts)
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

    /// Every response carries lane, config vector, admission policy, model
    /// digest, host and worker width — even on a non-completion path, whose body
    /// is left alone.
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
        assert_eq!(header(&resp, "x-camelid-admission-sha256"), TEST_ADMISSION_SHA[..12]);
        assert_eq!(header(&resp, "x-camelid-model-sha256"), TEST_MODEL_SHA[..12]);
        assert_eq!(header(&resp, "x-camelid-host"), TEST_HOST);
        assert_eq!(header(&resp, "x-camelid-worker-threads"), "6");

        let body: serde_json::Value = serde_json::from_str(&read_body(resp).await).unwrap();
        assert!(
            body.get("camelid_lane").is_none(),
            "a non-completion body must not be attributed: {body}"
        );
        assert_eq!(body["object"], "list", "original fields must be preserved");
    }

    /// Two replicas serving different weights must not publish the same
    /// attribution. This is the whole reason the model digest exists: everything
    /// else about these two replicas is byte-identical.
    #[tokio::test]
    async fn two_model_digests_render_differently_everywhere_they_are_published() {
        async fn responses(model_sha: &str) -> (String, serde_json::Value) {
            let app = Router::new()
                .route(
                    "/v1/chat/completions",
                    post(|| async { Json(serde_json::json!({ "id": "chatcmpl-1" })) }),
                )
                .layer(from_fn_with_state(ctx_for(model_sha, None), attribute));
            let resp = app
                .oneshot(Request::post("/v1/chat/completions").body(Body::empty()).unwrap())
                .await
                .unwrap();
            let header_value = header(&resp, "x-camelid-model-sha256");
            let body: serde_json::Value =
                serde_json::from_str(&read_body(resp).await).unwrap();
            (header_value, body)
        }

        let (one_header, one_body) = responses(TEST_MODEL_SHA).await;
        let (two_header, two_body) = responses(OTHER_MODEL_SHA).await;

        assert_eq!(one_header, TEST_MODEL_SHA[..12]);
        assert_eq!(two_header, OTHER_MODEL_SHA[..12]);
        assert_ne!(one_header, two_header, "the header must distinguish the weights");
        assert_ne!(
            one_body["camelid_model_sha256"], two_body["camelid_model_sha256"],
            "the body must distinguish the weights"
        );
        // …while every other published field stays identical, which is what
        // makes the model digest load-bearing rather than redundant.
        assert_eq!(one_body["camelid_config_sha256"], two_body["camelid_config_sha256"]);
        assert_eq!(one_body["camelid_worker_threads"], two_body["camelid_worker_threads"]);
    }

    /// Two builds whose allow lists differ must not publish identical
    /// attribution, and this is the only field that can tell them apart:
    /// everything else about them — the applied vector, the weights, the host,
    /// the width — is byte-identical, which is exactly the shape of the attack
    /// the admission digest exists for. A build that gained one permit would
    /// start under a variable that moves the forward pass; without this field a
    /// client has no way to ask what a replica would have refused.
    #[tokio::test]
    async fn two_admission_policies_render_differently_everywhere_they_are_published() {
        async fn responses(admission_sha: &str) -> (String, serde_json::Value, String) {
            let dir = tempfile::tempdir().unwrap();
            let receipts = dir.path().join("receipts.jsonl");
            let app = Router::new()
                .route(
                    "/v1/chat/completions",
                    post(|| async { Json(serde_json::json!({ "id": "chatcmpl-1" })) }),
                )
                .layer(from_fn_with_state(
                    ctx_full(TEST_MODEL_SHA, admission_sha, Some(Arc::new(receipts.clone()))),
                    attribute,
                ));
            let resp = app
                .oneshot(Request::post("/v1/chat/completions").body(Body::empty()).unwrap())
                .await
                .unwrap();
            let header_value = header(&resp, "x-camelid-admission-sha256");
            let body: serde_json::Value =
                serde_json::from_str(&read_body(resp).await).unwrap();
            let receipt = read_receipts(&receipts, 1).await.remove(0);
            (header_value, body, receipt["admission_sha256"].as_str().unwrap().to_string())
        }

        let (one_header, one_body, one_receipt) = responses(TEST_ADMISSION_SHA).await;
        let (two_header, two_body, two_receipt) = responses(OTHER_ADMISSION_SHA).await;

        assert_eq!(one_header, TEST_ADMISSION_SHA[..12]);
        assert_ne!(one_header, two_header, "the header must distinguish the policies");
        assert_ne!(
            one_body["camelid_admission_sha256"], two_body["camelid_admission_sha256"],
            "the body must distinguish the policies"
        );
        // Full length on the receipt, like every other digest there.
        assert_eq!(one_receipt, TEST_ADMISSION_SHA);
        assert_eq!(two_receipt, OTHER_ADMISSION_SHA);

        // …while every other published field is identical, which is what makes
        // this field load-bearing rather than redundant.
        assert_eq!(one_body["camelid_config_sha256"], two_body["camelid_config_sha256"]);
        assert_eq!(one_body["camelid_model_sha256"], two_body["camelid_model_sha256"]);
        assert_eq!(one_body["camelid_worker_threads"], two_body["camelid_worker_threads"]);
    }

    /// Two replicas on different hardware classes must not publish identical
    /// attribution on any of the three surfaces — and the body is the one this
    /// was wrong on.
    ///
    /// The hardware class is the axis the reproducibility scope is stated over.
    /// Same weights, same configuration, same admission policy, same pool width,
    /// one host with a SIMD feature the other lacks: the engine can route
    /// through a different kernel and the two emit different tokens. Everything
    /// else these two replicas publish is byte-identical, which is exactly what
    /// makes `host` load-bearing rather than decorative, and exactly why leaving
    /// it off the body left a client that verifies from the body — the surface
    /// the README's own example reads — unable to see the axis that moved.
    #[tokio::test]
    async fn two_hosts_render_differently_everywhere_they_are_published() {
        const OTHER_HOST: &str = "linux/x86_64 cores=8 simd=avx2+avx512f+avx512vnni+fma";

        async fn responses(host: &str) -> (String, serde_json::Value, String) {
            let dir = tempfile::tempdir().unwrap();
            let receipts = dir.path().join("receipts.jsonl");
            let app = Router::new()
                .route(
                    "/v1/chat/completions",
                    post(|| async { Json(serde_json::json!({ "id": "chatcmpl-1" })) }),
                )
                .layer(from_fn_with_state(
                    ctx_on_host(host, Some(Arc::new(receipts.clone()))),
                    attribute,
                ));
            let resp = app
                .oneshot(Request::post("/v1/chat/completions").body(Body::empty()).unwrap())
                .await
                .unwrap();
            let header_value = header(&resp, "x-camelid-host");
            let body: serde_json::Value =
                serde_json::from_str(&read_body(resp).await).unwrap();
            let receipt = read_receipts(&receipts, 1).await.remove(0);
            (header_value, body, receipt["host"].as_str().unwrap().to_string())
        }

        let (one_header, one_body, one_receipt) = responses(TEST_HOST).await;
        let (two_header, two_body, two_receipt) = responses(OTHER_HOST).await;

        assert_eq!(one_header, TEST_HOST);
        assert_ne!(one_header, two_header, "the header must distinguish the hosts");
        assert_eq!(one_body["camelid_host"], TEST_HOST);
        assert_ne!(
            one_body["camelid_host"], two_body["camelid_host"],
            "the completion body must distinguish the hosts"
        );
        assert_eq!(one_receipt, TEST_HOST);
        assert_ne!(one_receipt, two_receipt, "the receipt must distinguish the hosts");

        // …while every other published field is identical, which is what makes
        // this field the only thing that could have told them apart.
        assert_eq!(one_body["camelid_config_sha256"], two_body["camelid_config_sha256"]);
        assert_eq!(one_body["camelid_admission_sha256"], two_body["camelid_admission_sha256"]);
        assert_eq!(one_body["camelid_model_sha256"], two_body["camelid_model_sha256"]);
        assert_eq!(one_body["camelid_worker_threads"], two_body["camelid_worker_threads"]);
    }

    /// A JSON completion body gains all six fields, each matching its header
    /// where both are published, and its original fields survive.
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

        let header_config = header(&resp, "x-camelid-config-sha256");
        let header_admission = header(&resp, "x-camelid-admission-sha256");
        let header_model = header(&resp, "x-camelid-model-sha256");
        let header_host = header(&resp, "x-camelid-host");
        let header_threads = header(&resp, "x-camelid-worker-threads");
        let body: serde_json::Value = serde_json::from_str(&read_body(resp).await).unwrap();
        assert_eq!(body["camelid_lane"], "deterministic");
        assert_eq!(body["camelid_config_sha256"], TEST_SHA[..12]);
        assert_eq!(body["camelid_config_sha256"].as_str().unwrap(), header_config);
        assert_eq!(body["camelid_admission_sha256"], TEST_ADMISSION_SHA[..12]);
        assert_eq!(body["camelid_admission_sha256"].as_str().unwrap(), header_admission);
        assert_eq!(body["camelid_model_sha256"], TEST_MODEL_SHA[..12]);
        assert_eq!(body["camelid_model_sha256"].as_str().unwrap(), header_model);
        assert_eq!(body["camelid_host"], TEST_HOST);
        assert_eq!(body["camelid_host"].as_str().unwrap(), header_host);
        // A number in the body, a decimal string on the wire — the same fact in
        // each format's own idiom.
        assert_eq!(body["camelid_worker_threads"], serde_json::json!(TEST_THREADS));
        assert!(body["camelid_worker_threads"].is_number(), "thread count must not be a string");
        assert_eq!(body["camelid_worker_threads"].as_u64().unwrap().to_string(), header_threads);
        assert_eq!(body["id"], "chatcmpl-1", "original fields must be preserved");
    }

    /// The three surfaces carry the same claims, derived rather than listed.
    ///
    /// This is the test that would have caught `camelid_host` being missing from
    /// the body, and the reason it is written this way: every per-field test
    /// checks the surfaces its own field is on, so a field that is on two of the
    /// three passes all of them. Here the header set is read off the response
    /// and each name is *mechanically* required to appear in the body and in the
    /// receipt — `x-camelid-worker-threads` → `camelid_worker_threads` →
    /// `worker_threads` — so a seventh header added to `stamp` alone fails here
    /// without anyone remembering to extend a list.
    ///
    /// Values are checked with `starts_with` in the receipt's direction because
    /// the digests are deliberately published short on the wire and full in the
    /// receipt; the body carries the short form, so there it is equality.
    #[tokio::test]
    async fn every_attribution_header_has_a_matching_body_and_receipt_field() {
        let dir = tempfile::tempdir().unwrap();
        let receipts = dir.path().join("receipts.jsonl");
        let app = Router::new()
            .route(
                "/v1/chat/completions",
                post(|| async { Json(serde_json::json!({ "id": "chatcmpl-1" })) }),
            )
            .layer(from_fn_with_state(ctx(Some(Arc::new(receipts.clone()))), attribute));
        let resp = app
            .oneshot(Request::post("/v1/chat/completions").body(Body::empty()).unwrap())
            .await
            .unwrap();

        let published: Vec<(String, String)> = resp
            .headers()
            .iter()
            .filter(|(name, _)| name.as_str().starts_with("x-camelid-"))
            .map(|(name, value)| {
                (name.as_str().to_string(), value.to_str().unwrap().to_string())
            })
            .collect();
        assert!(published.len() >= 6, "expected the full header set: {published:?}");

        let body: serde_json::Value = serde_json::from_str(&read_body(resp).await).unwrap();
        let receipt = read_receipts(&receipts, 1).await.remove(0);

        /// A JSON value as the wire would spell it, so a number and a header
        /// string compare as the one fact they are.
        fn rendered(value: &serde_json::Value) -> String {
            value.as_str().map(ToOwned::to_owned).unwrap_or_else(|| value.to_string())
        }

        for (name, value) in published {
            let body_field = name.replace("x-camelid-", "camelid_").replace('-', "_");
            let receipt_field = name.replace("x-camelid-", "").replace('-', "_");
            let in_body = body
                .get(&body_field)
                .unwrap_or_else(|| panic!("{name} is published on the wire but not as {body_field} in the completion body"));
            let in_receipt = receipt
                .get(&receipt_field)
                .unwrap_or_else(|| panic!("{name} is published on the wire but not as {receipt_field} in the receipt"));
            assert_eq!(rendered(in_body), value, "{body_field} disagrees with {name}");
            assert!(
                rendered(in_receipt).starts_with(&value),
                "{receipt_field} disagrees with {name}: {in_receipt} vs {value}"
            );
        }
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
        assert_eq!(header(&resp, "x-camelid-model-sha256"), TEST_MODEL_SHA[..12]);
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

    /// A streaming (`text/event-stream`) completion response is passed through
    /// untouched — attribution never buffers or rewrites a stream — while the
    /// identity headers are still set. A stream is where the body fields are
    /// unavailable, so it is where the headers matter most.
    #[tokio::test]
    async fn event_stream_completion_body_is_passed_through() {
        let app = attributed(Router::new().route(
            "/v1/chat/completions",
            post(|| async { ([(CONTENT_TYPE, "text/event-stream")], SSE_BODY) }),
        ));
        let resp = app
            .oneshot(Request::post("/v1/chat/completions").body(Body::empty()).unwrap())
            .await
            .unwrap();

        assert_eq!(header(&resp, "x-camelid-lane"), "deterministic");
        assert_eq!(header(&resp, "x-camelid-config-sha256"), TEST_SHA[..12]);
        assert_eq!(header(&resp, "x-camelid-admission-sha256"), TEST_ADMISSION_SHA[..12]);
        assert_eq!(header(&resp, "x-camelid-model-sha256"), TEST_MODEL_SHA[..12]);
        assert_eq!(header(&resp, "x-camelid-worker-threads"), "6");
        assert_eq!(read_body(resp).await, SSE_BODY);
    }

    /// The stream is really passed through and not merely reassembled — the
    /// claim `event_stream_completion_body_is_passed_through` makes and cannot
    /// check.
    ///
    /// That test drives the router with `oneshot` and reads the body with
    /// `to_bytes`, which collapses the stream inside the harness whether or not
    /// the middleware collapsed it first, so it compares identical bytes in both
    /// worlds: delete the `is_json` guard and it still passes, while in
    /// production every streamed completion has become a single bounded blocking
    /// read before the first token reaches the client.
    ///
    /// A stream past the buffer ceiling is where the two worlds disagree
    /// observably. Passed through, it is served; drained through `to_bytes`, it
    /// trips the limit and becomes the manufactured 500. That is also a real
    /// case rather than a contrivance — a long generation's SSE stream is not
    /// bounded by anything this middleware knows about, so buffering it is a
    /// ceiling on how much a client may be sent.
    #[tokio::test]
    async fn a_stream_past_the_buffer_ceiling_is_still_passed_through() {
        let streamed = vec![b'x'; BODY_LIMIT + 1];
        let expected = streamed.len();
        let app = attributed(Router::new().route(
            "/v1/chat/completions",
            post(|| async move { ([(CONTENT_TYPE, "text/event-stream")], streamed) }),
        ));
        let resp = app
            .oneshot(Request::post("/v1/chat/completions").body(Body::empty()).unwrap())
            .await
            .unwrap();

        assert_eq!(
            resp.status(),
            StatusCode::OK,
            "a stream longer than the buffer was buffered, and failed closed as if the engine \
             had produced an oversized JSON body"
        );
        assert_eq!(header(&resp, "x-camelid-model-sha256"), TEST_MODEL_SHA[..12]);
        let bytes = to_bytes(resp.into_body(), BODY_LIMIT * 2).await.unwrap();
        assert_eq!(bytes.len(), expected, "the stream must reach the client whole");
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
        // The fields added by this change must survive the failure path too —
        // that path synthesizes its own response, which is exactly where a new
        // field gets forgotten.
        assert_eq!(header(&resp, "x-camelid-model-sha256"), TEST_MODEL_SHA[..12]);
        assert_eq!(header(&resp, "x-camelid-admission-sha256"), TEST_ADMISSION_SHA[..12]);
        assert_eq!(header(&resp, "x-camelid-worker-threads"), "6");
        // The manufactured 500 must itself be a well-formed, labeled JSON error.
        assert_eq!(header(&resp, "content-type"), "application/json");
        let body: serde_json::Value = serde_json::from_str(&read_body(resp).await).unwrap();
        assert_eq!(body["error"]["type"], "server_error");
        assert!(
            body["error"]["message"]
                .as_str()
                .unwrap()
                .contains("attribution buffer limit"),
            "unexpected error body: {body}"
        );
    }

    /// With receipts enabled, each request appends exactly one JSONL line
    /// carrying method, path, status, lane, the full config digest, the full
    /// model digest, host, worker width and a numeric timestamp. Two requests
    /// must yield two lines — append, not truncate — and concurrent appends must
    /// not corrupt a line.
    #[tokio::test]
    async fn receipts_append_one_line_per_request() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("receipts.jsonl");
        let app = Router::new()
            .route("/v1/models", get(|| async { Json(serde_json::json!({ "object": "list" })) }))
            .route(
                "/v1/completions",
                post(|| async { Json(serde_json::json!({ "id": "x" })) }),
            )
            .layer(from_fn_with_state(ctx(Some(Arc::new(path.clone()))), attribute));

        app.clone()
            .oneshot(Request::get("/v1/models").body(Body::empty()).unwrap())
            .await
            .unwrap();
        app.oneshot(Request::post("/v1/completions").body(Body::empty()).unwrap())
            .await
            .unwrap();

        let receipts = read_receipts(&path, 2).await;
        for r in &receipts {
            assert_eq!(r["lane"], "deterministic");
            assert_eq!(r["config_sha256"], TEST_SHA);
            assert_eq!(r["admission_sha256"], TEST_ADMISSION_SHA);
            // Full 64 on the receipt, not the 12 the wire carries: this is the
            // copy an audit compares against a file.
            assert_eq!(r["model_sha256"], TEST_MODEL_SHA);
            assert_eq!(r["host"], TEST_HOST);
            assert_eq!(r["worker_threads"], serde_json::json!(TEST_THREADS));
            assert_eq!(r["status"], 200);
            assert!(r["ts"].is_number(), "ts must be numeric: {r}");
        }
        // The two append writes may land in either order, so match on the set.
        let seen: std::collections::HashSet<(String, String)> = receipts
            .iter()
            .map(|r| {
                (
                    r["method"].as_str().unwrap().to_string(),
                    r["path"].as_str().unwrap().to_string(),
                )
            })
            .collect();
        assert!(seen.contains(&("GET".to_string(), "/v1/models".to_string())));
        assert!(seen.contains(&("POST".to_string(), "/v1/completions".to_string())));
    }

    async fn read_receipts(path: &Path, expected: usize) -> Vec<serde_json::Value> {
        for _ in 0..200 {
            if let Ok(contents) = std::fs::read_to_string(path) {
                let lines: Vec<&str> = contents.lines().filter(|l| !l.trim().is_empty()).collect();
                if lines.len() >= expected {
                    return lines.iter().map(|l| serde_json::from_str(l).unwrap()).collect();
                }
            }
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        }
        panic!("expected {expected} receipt line(s) were not written within the timeout");
    }

    /// The digest is over the file's bytes, computed the way an operator would
    /// check it. The expected value is the SHA-256 of the literal contents, so
    /// this fails if the read is ever chunked wrongly or the hex rendering
    /// changes — not merely if two calls happen to agree with each other.
    #[test]
    fn the_model_digest_is_the_sha256_of_the_file_contents() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("model.gguf");
        std::fs::File::create(&path).unwrap().write_all(b"weights").unwrap();

        // `printf 'weights' | shasum -a 256`
        const EXPECTED: &str =
            "9a129038d9a00aed0cf6a7ea059ca50a813449061ab87848cf1a13eafdf33b2c";
        let identity = ModelIdentity::of_file(&path).unwrap();
        assert_eq!(identity.sha256.as_str(), EXPECTED);
        assert_eq!(identity.short(), &EXPECTED[..12]);
    }

    /// Content, not name: two files at different paths with identical bytes have
    /// one identity, and editing a byte changes it.
    #[test]
    fn the_model_digest_follows_content_not_path() {
        let dir = tempfile::tempdir().unwrap();
        let a = dir.path().join("a.gguf");
        let b = dir.path().join("b.gguf");
        let changed = dir.path().join("changed.gguf");
        std::fs::write(&a, b"same bytes").unwrap();
        std::fs::write(&b, b"same bytes").unwrap();
        std::fs::write(&changed, b"same byteS").unwrap();

        let a = ModelIdentity::of_file(&a).unwrap();
        let b = ModelIdentity::of_file(&b).unwrap();
        let changed = ModelIdentity::of_file(&changed).unwrap();
        assert_eq!(a.sha256, b.sha256, "identical bytes are one model");
        assert_ne!(a.sha256, changed.sha256, "one changed byte is a different model");
    }

    /// A file larger than one read chunk must digest to the same value as the
    /// hasher would produce over the whole slice — the chunked read is an
    /// implementation detail and must not be visible in the output.
    #[test]
    fn the_digest_is_unaffected_by_the_chunk_boundary() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("big.gguf");
        let contents: Vec<u8> = (0..(DIGEST_CHUNK * 2 + 12345)).map(|i| (i % 251) as u8).collect();
        std::fs::write(&path, &contents).unwrap();

        let whole = format!("{:x}", Sha256::digest(&contents));
        assert_eq!(ModelIdentity::of_file(&path).unwrap().sha256.as_str(), whole);
    }

    /// A replica that cannot identify its weights must not start. Failing here
    /// names the file; failing later names nothing useful.
    #[test]
    fn an_unreadable_model_is_a_startup_error_naming_the_path() {
        let dir = tempfile::tempdir().unwrap();
        let missing = dir.path().join("absent.gguf");
        let err = ModelIdentity::of_file(&missing)
            .err()
            .expect("a model that cannot be read must not yield an identity");
        assert!(err.contains("absent.gguf"), "the error must name the file: {err}");
        assert!(err.contains("identify"), "the error must say what failed: {err}");
    }

    /// Replace-by-rename between hashing and loading is the accident a build
    /// pipeline produces, and the one the fingerprint is shaped to catch.
    ///
    /// `#[cfg(unix)]`, and that is the assertion rather than a portability
    /// convenience: catching a *same-length* replacement is what the device and
    /// inode buy, and off unix the fingerprint degrades to length alone, so this
    /// case genuinely is not detected there. Running it on Windows would not
    /// test a weaker guarantee, it would fail — which is why the crate ships the
    /// weaker guarantee documented on `verify_unchanged` rather than a test
    /// implying otherwise.
    #[test]
    #[cfg(unix)]
    fn a_model_swapped_after_it_was_identified_is_refused() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("model.gguf");
        let replacement = dir.path().join("other.gguf");
        std::fs::write(&path, b"original weights").unwrap();
        std::fs::write(&replacement, b"swapped weights!").unwrap();

        let identity = ModelIdentity::of_file(&path).unwrap();
        identity.verify_unchanged(&path).expect("an untouched file must verify");

        // Same length, different bytes, swapped in by rename — the case a
        // length-only check would miss on unix.
        std::fs::rename(&replacement, &path).unwrap();
        let err = identity.verify_unchanged(&path).unwrap_err();
        assert!(err.contains("changed between"), "unexpected message: {err}");
        assert!(err.contains(identity.short()), "the refusal must name what it published: {err}");
    }

    /// A replacement of a *different* length is caught on every platform, which
    /// is the part of the check that is not unix-specific. Asserted separately
    /// so the crate has coverage of `verify_unchanged` on the Windows target it
    /// ships support for.
    #[test]
    fn a_model_replaced_at_a_different_length_is_refused_on_every_platform() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("model.gguf");
        std::fs::write(&path, b"original weights").unwrap();

        let identity = ModelIdentity::of_file(&path).unwrap();
        identity.verify_unchanged(&path).expect("an untouched file must verify");

        std::fs::write(&path, b"weights, but longer than they were").unwrap();
        let err = identity.verify_unchanged(&path).unwrap_err();
        assert!(err.contains("changed between"), "unexpected message: {err}");
    }

    /// The documented limit, asserted so it cannot quietly become a claim: a
    /// same-length rewrite in place leaves length, device and inode untouched,
    /// so the check passes over bytes that are no longer the ones hashed. The
    /// control against a writer who can reach the file is the read-only mount.
    #[test]
    fn a_same_length_rewrite_in_place_is_not_detected() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("model.gguf");
        std::fs::write(&path, b"original weights").unwrap();

        let identity = ModelIdentity::of_file(&path).unwrap();
        std::fs::write(&path, b"rewritten bytes!").unwrap();
        assert_eq!(std::fs::metadata(&path).unwrap().len(), 16);
        assert_ne!(
            ModelIdentity::of_file(&path).unwrap().sha256,
            identity.sha256,
            "the bytes really did change"
        );
        identity
            .verify_unchanged(&path)
            .expect("documented limit: a same-length in-place rewrite is not detected");
    }

    /// The width published is the one it was told, as a number and as a string,
    /// and the two encodings are one value.
    #[test]
    fn the_worker_width_renders_the_same_in_both_encodings() {
        for count in [1usize, 8, 128] {
            let workers = WorkerThreads::resolved(count);
            assert_eq!(workers.count(), count);
            assert_eq!(workers.header.to_str().unwrap(), count.to_string());
        }
    }
}
