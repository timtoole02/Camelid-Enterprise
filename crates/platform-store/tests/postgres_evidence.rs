//! Integration tests for evidence aggregation against a real PostgreSQL.
//!
//! The database harness, and the reason these tests skip without a URL, live in
//! `support`.

use camelid_enterprise_platform_store::{
    EvidenceStream, IngestSummary, Ingested, PlatformStore, PLATFORM_SCHEMA_VERSION,
};

mod support;
use support::TestDatabase;

const REQUEST: &str = "req_0123456789abcdef0123456789abcdef";

fn audit_line(request_id: &str, organization: Option<&str>) -> String {
    let organization = match organization {
        Some(organization) => format!("\"{organization}\""),
        None => "null".to_string(),
    };
    format!(
        "{{\"ts\":1754100000.125,\"request_id\":\"{request_id}\",\"principal\":\"prn_a\",\
         \"organization\":{organization},\"reason\":null,\"method\":\"POST\",\
         \"path\":\"/v1/chat/completions\",\"model_id\":\"alpha\",\"status\":200}}"
    )
}

fn usage_line(request_id: &str) -> String {
    format!(
        "{{\"ts\":1754100001.5,\"started_ts\":1754100000.1,\"duration_ms\":1400,\
         \"request_id\":\"{request_id}\",\"principal\":\"prn_a\",\"organization\":\"org_a\",\
         \"method\":\"POST\",\"path\":\"/v1/chat/completions\",\"model_id\":\"alpha\",\
         \"response_head_status\":200,\"request_bytes\":412,\"response_bytes\":8192,\
         \"stream_outcome\":\"completed\"}}"
    )
}

fn receipt_line(request_id: Option<&str>) -> String {
    let request_id = match request_id {
        Some(request_id) => format!("\"{request_id}\""),
        None => "null".to_string(),
    };
    format!(
        "{{\"ts\":1754100001.4,\"request_id\":{request_id},\"method\":\"POST\",\
         \"path\":\"/v1/chat/completions\",\"status\":200,\"lane\":\"deterministic\",\
         \"config_sha256\":\"aa11\",\"admission_sha256\":\"bb22\",\"posture\":\"cpu\",\
         \"engine_sha256\":\"cc33\",\"model_sha256\":\"dd44\",\
         \"host\":\"linux/x86_64 cores=16\",\"worker_threads\":8}}"
    )
}

/// Guards against this whole file quietly becoming a no-op.
#[tokio::test]
async fn postgres_evidence_tests_are_not_silently_skipped() {
    support::assert_not_silently_skipped();
}

/// The point of the whole exercise: three files written by three processes that
/// never spoke to each other become one row.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn one_request_is_reconstructed_from_all_three_streams() {
    let Some(database) = TestDatabase::create().await else {
        return;
    };
    let store = PlatformStore::connect(&database.config())
        .await
        .expect("store");

    store
        .ingest_evidence(
            EvidenceStream::GatewayAudit,
            &audit_line(REQUEST, Some("org_a")),
        )
        .await
        .expect("audit");
    store
        .ingest_evidence(EvidenceStream::GatewayUsage, &usage_line(REQUEST))
        .await
        .expect("usage");
    store
        .ingest_evidence(EvidenceStream::ReplicaReceipt, &receipt_line(Some(REQUEST)))
        .await
        .expect("receipt");

    let row = database
        .client()
        .await
        .query_one(
            "SELECT a.organization, \
                    a.record->>'status', \
                    u.record->>'stream_outcome', \
                    u.record->>'response_bytes', \
                    r.record->>'model_sha256', \
                    r.record->>'posture' \
             FROM gateway_audit a \
             JOIN gateway_usage u USING (request_id) \
             JOIN replica_receipt r USING (request_id) \
             WHERE a.request_id = $1",
            &[&REQUEST],
        )
        .await
        .expect("the three streams join on request_id");

    assert_eq!(row.get::<_, Option<String>>(0).as_deref(), Some("org_a"));
    assert_eq!(row.get::<_, Option<String>>(1).as_deref(), Some("200"));
    assert_eq!(
        row.get::<_, Option<String>>(2).as_deref(),
        Some("completed")
    );
    assert_eq!(row.get::<_, Option<String>>(3).as_deref(), Some("8192"));
    assert_eq!(row.get::<_, Option<String>>(4).as_deref(), Some("dd44"));
    // Added to the receipt by ADR 0004 and never mentioned in this crate's
    // schema: a record keeps what it says, not what this build understood.
    assert_eq!(row.get::<_, Option<String>>(5).as_deref(), Some("cpu"));

    database.drop_database().await;
}

/// Reading a growing file from the top is the intended way to run this, so the
/// overlap has to converge rather than duplicate.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn ingesting_the_same_file_twice_stores_it_once() {
    let Some(database) = TestDatabase::create().await else {
        return;
    };
    let store = PlatformStore::connect(&database.config())
        .await
        .expect("store");
    let file = format!(
        "{}\n{}\n",
        audit_line(REQUEST, Some("org_a")),
        audit_line("req_ffffffffffffffffffffffffffffffff", Some("org_b"))
    );

    let first = store
        .ingest_evidence(EvidenceStream::GatewayAudit, &file)
        .await
        .expect("first pass");
    let second = store
        .ingest_evidence(EvidenceStream::GatewayAudit, &file)
        .await
        .expect("second pass");

    assert_eq!(
        first,
        IngestSummary {
            stored: 2,
            already_present: 0,
            unreadable: 0
        }
    );
    assert_eq!(
        second,
        IngestSummary {
            stored: 0,
            already_present: 2,
            unreadable: 0
        }
    );
    assert_eq!(count(&database, "gateway_audit").await, 2);

    database.drop_database().await;
}

/// What a killed process actually leaves behind. One half-written last line
/// must not cost the operator the complete records in front of it.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_truncated_final_line_does_not_cost_the_lines_before_it() {
    let Some(database) = TestDatabase::create().await else {
        return;
    };
    let store = PlatformStore::connect(&database.config())
        .await
        .expect("store");
    let whole = audit_line("req_cccccccccccccccccccccccccccccccc", Some("org_a"));
    let file = format!(
        "{}\n{}\n{}",
        audit_line(REQUEST, Some("org_a")),
        whole,
        &whole[..whole.len() / 2]
    );

    let summary = store
        .ingest_evidence(EvidenceStream::GatewayAudit, &file)
        .await
        .expect("a torn tail is not a failed file");

    assert_eq!(
        summary,
        IngestSummary {
            stored: 2,
            already_present: 0,
            unreadable: 1
        }
    );
    assert_eq!(count(&database, "gateway_audit").await, 2);

    database.drop_database().await;
}

/// A replica reached directly is never given a correlation id. Its receipt is
/// still the record of what served that request; it simply cannot be joined.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_receipt_without_a_correlation_id_is_still_stored() {
    let Some(database) = TestDatabase::create().await else {
        return;
    };
    let store = PlatformStore::connect(&database.config())
        .await
        .expect("store");

    assert_eq!(
        store
            .ingest_evidence_line(EvidenceStream::ReplicaReceipt, &receipt_line(None))
            .await
            .expect("store"),
        Ingested::Stored
    );

    let unjoinable: i64 = database
        .client()
        .await
        .query_one(
            "SELECT count(*) FROM replica_receipt WHERE request_id IS NULL",
            &[],
        )
        .await
        .expect("query")
        .get(0);
    assert_eq!(unjoinable, 1);

    // The same tolerance would be a hole in the gateway's streams, where the
    // gateway mints the id itself and a line without one is a defect.
    let missing = store
        .ingest_evidence_line(EvidenceStream::GatewayAudit, &audit_line("", None))
        .await
        .expect("a malformed line is reported, not raised");
    assert!(
        matches!(&missing, Ingested::Unreadable(why) if why.contains("request_id")),
        "expected an unreadable audit line, got {missing:?}"
    );
    assert_eq!(count(&database, "gateway_audit").await, 0);

    database.drop_database().await;
}

/// Authentication can be off, or a request can be refused before identity is
/// established. The audit line still exists and still has to land.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn an_audit_line_with_no_organization_is_still_stored() {
    let Some(database) = TestDatabase::create().await else {
        return;
    };
    let store = PlatformStore::connect(&database.config())
        .await
        .expect("store");

    store
        .ingest_evidence(EvidenceStream::GatewayAudit, &audit_line(REQUEST, None))
        .await
        .expect("ingest");

    let anonymous: i64 = database
        .client()
        .await
        .query_one(
            "SELECT count(*) FROM gateway_audit WHERE organization IS NULL",
            &[],
        )
        .await
        .expect("query")
        .get(0);
    assert_eq!(anonymous, 1);

    database.drop_database().await;
}

/// The upgrade an existing deployment actually performs: a store already
/// holding quota state gains the evidence tables and loses none of it.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_store_already_holding_quota_gains_the_evidence_tables() {
    let Some(database) = TestDatabase::create().await else {
        return;
    };
    PlatformStore::connect(&database.config())
        .await
        .expect("store");

    // Rewind to what a pod running the previous release left behind.
    let client = database.client().await;
    client
        .batch_execute(
            "DROP TABLE gateway_audit, gateway_usage, replica_receipt; \
             UPDATE platform_schema_version SET version = 1; \
             INSERT INTO quota_config (request_limit, window_seconds) VALUES (10, 60); \
             INSERT INTO quota_windows VALUES ('org_a', 1754100000, 7);",
        )
        .await
        .expect("rewind to v1");

    let store = PlatformStore::connect(&database.config())
        .await
        .expect("a v1 store must be upgraded, not refused");
    store
        .ingest_evidence(
            EvidenceStream::GatewayAudit,
            &audit_line(REQUEST, Some("org_a")),
        )
        .await
        .expect("ingest");

    let version: i32 = client
        .query_one("SELECT version FROM platform_schema_version", &[])
        .await
        .expect("query")
        .get(0);
    assert_eq!(version, PLATFORM_SCHEMA_VERSION);
    let spent: i64 = client
        .query_one(
            "SELECT request_count FROM quota_windows WHERE organization_id = 'org_a'",
            &[],
        )
        .await
        .expect("query")
        .get(0);
    assert_eq!(spent, 7, "the upgrade discarded a live quota counter");
    assert_eq!(count(&database, "gateway_audit").await, 1);

    database.drop_database().await;
}

async fn count(database: &TestDatabase, table: &str) -> i64 {
    database
        .client()
        .await
        .query_one(&format!("SELECT count(*) FROM {table}"), &[])
        .await
        .expect("count")
        .get(0)
}
