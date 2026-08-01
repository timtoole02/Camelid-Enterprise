use axum::body::{to_bytes, Body};
use axum::http::{header::CONTENT_TYPE, Request, StatusCode};
use camelid_enterprise::{
    apply_deterministic, replica_router, Attribution, ModelIdentity, WorkerThreads,
};
use camelid_enterprise_gateway::{
    router_with_model_catalog, GatewayAuth, GatewayLog, ModelCatalog, ModelSelectionLimits,
    UpstreamOrigin, DEFAULT_MAX_IN_FLIGHT, DEFAULT_MAX_MODEL_SELECTION_BODY_BYTES,
    DEFAULT_MODEL_SELECTION_MEMORY_BUDGET_BYTES,
};
use hyper_util::client::legacy::connect::HttpConnector;
use hyper_util::client::legacy::Client;
use hyper_util::rt::TokioExecutor;
use engine_core::runtime::LoadedModel;
use serde_json::{json, Value};
use std::net::SocketAddr;
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;
use tokio::task::JoinHandle;

const MODEL_ENV: &str = "CAMELID_ENTERPRISE_TEST_MODEL";
const STEP_TIMEOUT: Duration = Duration::from_secs(900);

struct TestServer {
    addr: SocketAddr,
    task: JoinHandle<()>,
}

impl Drop for TestServer {
    fn drop(&mut self) {
        self.task.abort();
    }
}

async fn spawn_router(router: axum::Router) -> TestServer {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let task = tokio::spawn(async move {
        axum::serve(listener, router).await.unwrap();
    });
    TestServer { addr, task }
}

async fn spawn_gateway(router: axum::Router) -> TestServer {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let task = tokio::spawn(async move {
        camelid_enterprise_gateway::serve(listener, router, STEP_TIMEOUT, std::future::pending())
            .await
            .unwrap();
    });
    TestServer { addr, task }
}

fn client() -> Client<HttpConnector, Body> {
    Client::builder(TokioExecutor::new()).build_http()
}

async fn send(request: Request<Body>) -> hyper::Response<hyper::body::Incoming> {
    tokio::time::timeout(STEP_TIMEOUT, client().request(request))
        .await
        .expect("model-backed gateway HTTP step exceeded 900 seconds")
        .unwrap()
}

async fn response_json(response: hyper::Response<hyper::body::Incoming>) -> Value {
    let bytes = tokio::time::timeout(
        STEP_TIMEOUT,
        to_bytes(Body::new(response.into_body()), 16 * 1024 * 1024),
    )
    .await
    .expect("model-backed gateway response body exceeded 900 seconds")
    .unwrap();
    serde_json::from_slice(&bytes).unwrap()
}

async fn read_jsonl(path: &Path, expected_records: usize) -> Vec<Value> {
    for _ in 0..200 {
        if let Ok(contents) = std::fs::read_to_string(path) {
            let records: Vec<Value> = contents
                .lines()
                .filter(|line| !line.trim().is_empty())
                .map(|line| serde_json::from_str(line).unwrap())
                .collect();
            if records.len() >= expected_records {
                return records;
            }
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    panic!(
        "expected {expected_records} JSONL records at {}",
        path.display()
    );
}

#[tokio::test]
#[ignore = "requires CAMELID_ENTERPRISE_TEST_MODEL to name a compatible local GGUF"]
async fn real_model_routes_through_a_verified_static_catalog() {
    let model = std::env::var(MODEL_ENV)
        .unwrap_or_else(|_| panic!("{MODEL_ENV} must name a compatible local GGUF"));
    let model = std::path::PathBuf::from(model)
        .canonicalize()
        .unwrap_or_else(|error| panic!("could not resolve {MODEL_ENV}: {error}"));
    let config = apply_deterministic().expect("the test process must apply the lane contract");
    let expected_sha = config.sha256.clone();
    let dir = tempfile::tempdir().unwrap();
    let receipt_path = dir.path().join("replica-receipts.jsonl");
    let identity = Attribution {
        lane: "deterministic",
        config_sha256: Arc::new(config.sha256),
        admission_sha256: Arc::new(config.admission_sha256),
        model: ModelIdentity::of_file(&model).expect("the model file must be readable"),
        host: Arc::new("gateway-catalog-model-test/host".to_string()),
        workers: WorkerThreads::resolved(rayon::current_num_threads()),
        receipts: Some(Arc::new(receipt_path.clone())),
    };
    let loaded_model = LoadedModel::load(&model)
        .expect("the in-tree runtime must load the catalog test model");
    let (replica, expected_model_id) = replica_router(loaded_model, &model, identity, 1)
        .expect("the production replica composition must accept the catalog test model");
    let replica = spawn_router(replica).await;

    let discovery = send(
        Request::get(format!("http://{}/v1/models", replica.addr))
            .body(Body::empty())
            .unwrap(),
    )
    .await;
    assert_eq!(discovery.status(), StatusCode::OK);
    let discovery = response_json(discovery).await;
    let model_id = discovery["data"]
        .as_array()
        .and_then(|models| models.first())
        .and_then(|model| model["id"].as_str())
        .expect("the loaded replica must advertise exactly one model id")
        .to_string();
    assert_eq!(model_id, expected_model_id);

    let catalog = ModelCatalog::new([(
        model_id.clone(),
        UpstreamOrigin::parse(&format!("http://{}", replica.addr)).unwrap(),
    )])
    .unwrap();
    let catalog = catalog.verify_backend_model_ids().await.unwrap();
    let audit_path = dir.path().join("gateway-audit.jsonl");
    let selection_limits = ModelSelectionLimits::new(
        DEFAULT_MAX_MODEL_SELECTION_BODY_BYTES,
        DEFAULT_MODEL_SELECTION_MEMORY_BUDGET_BYTES,
    )
    .unwrap();
    let gateway = spawn_gateway(router_with_model_catalog(
        catalog,
        DEFAULT_MAX_IN_FLIGHT,
        selection_limits,
        None,
        GatewayAuth::Disabled,
        Some(GatewayLog::open(&audit_path).unwrap()),
    ))
    .await;

    let response = send(
        Request::post(format!("http://{}/v1/chat/completions", gateway.addr))
            .header(CONTENT_TYPE, "application/json")
            .body(Body::from(
                json!({
                    "model": model_id,
                    "messages": [{ "role": "user", "content": "Reply with one word." }],
                    "temperature": 0,
                    "max_tokens": 2,
                    "stream": false
                })
                .to_string(),
            ))
            .unwrap(),
    )
    .await;
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(response.headers()["x-camelid-lane"], "deterministic");
    assert_eq!(
        response.headers()["x-camelid-config-sha256"],
        &expected_sha[..12]
    );
    let completion = response_json(response).await;
    assert_eq!(completion["model"], model_id);
    assert_eq!(completion["camelid_lane"], "deterministic");
    assert_eq!(completion["camelid_config_sha256"], &expected_sha[..12]);

    let audit = read_jsonl(&audit_path, 1).await;
    assert_eq!(audit[0]["model_id"], model_id);
    assert_eq!(audit[0]["status"], 200);
    let request_id = audit[0]["request_id"]
        .as_str()
        .expect("gateway audit record must have a request id");
    let receipts = read_jsonl(&receipt_path, 3).await;
    let receipt = receipts
        .iter()
        .find(|receipt| receipt["request_id"] == request_id)
        .expect("the replica receipt must carry the gateway audit request id");
    assert_eq!(receipt["path"], "/v1/chat/completions");
    assert_eq!(receipt["status"], 200);
    assert_eq!(receipt["config_sha256"], expected_sha);
}
