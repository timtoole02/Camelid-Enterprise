//! Local identity primitive for Camelid Enterprise.
//!
//! This crate deliberately does not model roles, quotas, or federation yet. It
//! provides the principal, organization, and credential indirection every
//! later identity feature (gateway enforcement, per-organization quotas, and
//! audit correlation) is built on:
//!
//! **an opaque bearer token resolves to an opaque [`AuthenticatedContext`].**
//!
//! Callers never see a raw database row id as a principal, and the store
//! never compares plaintext tokens: only a token's SHA-256 hash is ever
//! persisted or looked up. This keeps the shape stable while the storage
//! backend, the identity model (users today, orgs/RBAC/SSO later), and the
//! enforcement point (nothing today, the gateway later) all change
//! independently. [`TokenStore`] retains its principal-only compatibility
//! contract while gateway integration is introduced separately.

use rusqlite::{Connection, OptionalExtension};
use sha2::{Digest, Sha256};
use std::fmt;
use std::path::Path;
use std::sync::{Mutex, MutexGuard};

/// Opaque identifier for a principal (today: exactly one user per token).
///
/// Deliberately not the storage row id: generated independently at creation
/// time so the storage backend can change without changing what a principal
/// *means* to callers above this crate.
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct PrincipalId(String);

impl PrincipalId {
    /// Wraps an existing principal id string, e.g. one an operator captured
    /// from [`SqliteIdentityStore::create_user`]'s return value, so it can be
    /// passed back into [`SqliteIdentityStore::issue_token`]. Does not create
    /// a user or validate anything by itself; operations against a principal
    /// id that names no user fail with [`IdentityError::Storage`].
    pub fn new(id: String) -> Self {
        Self(id)
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for PrincipalId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// Opaque identifier for an organization.
///
/// Like [`PrincipalId`], this is generated independently of SQLite row ids so
/// storage details never become part of a caller-visible identity.
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct OrganizationId(String);

impl OrganizationId {
    /// Wraps an organization id previously returned by this store. Operations
    /// against an unknown organization return [`IdentityError::Storage`].
    pub fn new(id: String) -> Self {
        Self(id)
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for OrganizationId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// The principal and organization authenticated by one bearer token.
///
/// Tokens intentionally select an organization rather than relying on a
/// principal-wide default. A principal that belongs to multiple organizations
/// therefore needs an explicitly scoped token for each intended context.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AuthenticatedContext {
    principal_id: PrincipalId,
    organization_id: OrganizationId,
}

impl AuthenticatedContext {
    pub fn principal_id(&self) -> &PrincipalId {
        &self.principal_id
    }

    pub fn organization_id(&self) -> &OrganizationId {
        &self.organization_id
    }
}

#[derive(Debug)]
pub enum IdentityError {
    /// The token does not resolve to any principal (never issued, or
    /// revoked). Deliberately does not distinguish the two: a caller must not
    /// be able to tell "never existed" from "revoked".
    InvalidToken,
    Storage(String),
}

impl fmt::Display for IdentityError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidToken => write!(f, "token does not resolve to any principal"),
            Self::Storage(msg) => write!(f, "identity storage error: {msg}"),
        }
    }
}

impl std::error::Error for IdentityError {}

impl From<rusqlite::Error> for IdentityError {
    fn from(err: rusqlite::Error) -> Self {
        Self::Storage(err.to_string())
    }
}

/// Resolves an opaque bearer token to the [`PrincipalId`] it belongs to.
///
/// The only contract: token in, principal out, or [`IdentityError::InvalidToken`].
/// Implementations must never expose the storage row backing a token.
pub trait TokenStore: Send + Sync {
    fn resolve(&self, token: &str) -> Result<PrincipalId, IdentityError>;
}

const TOKEN_BYTES: usize = 32; // 256-bit bearer token
const PRINCIPAL_BYTES: usize = 16; // 128-bit opaque principal id
const ORGANIZATION_BYTES: usize = 16; // 128-bit opaque organization id
const TOKEN_PREFIX: &str = "cme_"; // Camelid Enterprise token marker
const IDENTITY_SCHEMA_VERSION: i64 = 2;

/// SQLite-backed identity store: users and their hashed tokens.
///
/// Holds one connection behind a mutex. SQLite serializes writes regardless of
/// how many handles reach it, so this does not give up real concurrency; it
/// only makes the store `Send + Sync` honestly instead of pretending
/// otherwise.
pub struct SqliteIdentityStore {
    conn: Mutex<Connection>,
}

impl SqliteIdentityStore {
    /// Opens (creating if absent) the identity database at `path` and ensures
    /// its schema exists.
    pub fn open(path: &Path) -> Result<Self, IdentityError> {
        let conn = Connection::open(path)?;
        // WAL lets `serve`'s long-lived auth-lookup connection and a
        // short-lived CLI process (create-user/issue-token/revoke-token)
        // touch the same database file concurrently without one blocking on
        // the other's rollback-journal lock. rusqlite already sets a
        // 5-second `sqlite3_busy_timeout` on every connection it opens, so a
        // transient conflict under either journal mode retries before
        // failing, not on it. SQLite silently keeps the prior mode instead of
        // erroring if WAL cannot be enabled (e.g. certain network
        // filesystems), so check the mode it actually reports back rather
        // than assuming the pragma took effect.
        let mode: String =
            conn.pragma_update_and_check(None, "journal_mode", "WAL", |row| row.get(0))?;
        if !mode.eq_ignore_ascii_case("wal") {
            return Err(IdentityError::Storage(format!(
                "identity database at {} could not be switched to WAL journal mode (reported: {mode})",
                path.display()
            )));
        }
        Self::from_connection(conn)
    }

    /// In-memory store. For tests and single-process smoke checks only:
    /// nothing survives the process.
    pub fn open_in_memory() -> Result<Self, IdentityError> {
        Self::from_connection(Connection::open_in_memory()?)
    }

    fn from_connection(mut conn: Connection) -> Result<Self, IdentityError> {
        conn.execute_batch("PRAGMA foreign_keys = ON;")?;
        initialize_schema(&mut conn)?;
        Ok(Self {
            conn: Mutex::new(conn),
        })
    }

    /// Locks the store's connection, recovering from poisoning instead of
    /// propagating it.
    ///
    /// A `std::sync::Mutex` never un-poisons itself once a panic occurs while
    /// it is held. Propagating that poisoning here would mean one unrelated
    /// panic during any identity operation permanently breaks every later
    /// one for the life of the process — in the gateway, that means every
    /// request fails once this lock is poisoned, until restart. Every
    /// operation in this file executes one self-contained statement (or a
    /// `SELECT` immediately followed by one `INSERT`/`DELETE`) and holds no
    /// transaction state across calls, so reusing a guard recovered from a
    /// poisoned lock is safe.
    fn locked_connection(&self) -> MutexGuard<'_, Connection> {
        self.conn
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// Creates a user and returns its opaque [`PrincipalId`]. `name` is a
    /// display label only; it is never used as a lookup key.
    pub fn create_user(&self, name: &str) -> Result<PrincipalId, IdentityError> {
        let principal = PrincipalId(random_id("usr"));
        let organization = OrganizationId(random_organization_id());
        let mut conn = self.locked_connection();
        let transaction = conn.transaction()?;
        let now = unix_now();
        transaction.execute(
            "INSERT INTO users (principal_id, name, created_at_unix) VALUES (?1, ?2, ?3)",
            rusqlite::params![principal.0, name, now],
        )?;
        transaction.execute(
            "INSERT INTO organizations (organization_id, name, created_at_unix) VALUES (?1, ?2, ?3)",
            rusqlite::params![organization.0, format!("Personal organization for {name}"), now],
        )?;
        transaction.execute(
            "INSERT INTO organization_memberships (organization_id, principal_id, created_at_unix) VALUES (?1, ?2, ?3)",
            rusqlite::params![organization.0, principal.0, now],
        )?;
        transaction.commit()?;
        Ok(principal)
    }

    /// Creates an organization and returns its opaque identifier. `name` is a
    /// display label only; callers must use the returned id for relationships.
    pub fn create_organization(&self, name: &str) -> Result<OrganizationId, IdentityError> {
        let organization = OrganizationId(random_organization_id());
        let conn = self.locked_connection();
        conn.execute(
            "INSERT INTO organizations (organization_id, name, created_at_unix) VALUES (?1, ?2, ?3)",
            rusqlite::params![organization.0, name, unix_now()],
        )?;
        Ok(organization)
    }

    /// Creates a principal as a member of an existing organization.
    pub fn create_user_in_organization(
        &self,
        name: &str,
        organization: &OrganizationId,
    ) -> Result<PrincipalId, IdentityError> {
        let principal = PrincipalId(random_id("usr"));
        let mut conn = self.locked_connection();
        let transaction = conn.transaction()?;
        require_organization(&transaction, organization)?;
        let now = unix_now();
        transaction.execute(
            "INSERT INTO users (principal_id, name, created_at_unix) VALUES (?1, ?2, ?3)",
            rusqlite::params![principal.0, name, now],
        )?;
        transaction.execute(
            "INSERT INTO organization_memberships (organization_id, principal_id, created_at_unix) VALUES (?1, ?2, ?3)",
            rusqlite::params![organization.0, principal.0, now],
        )?;
        transaction.commit()?;
        Ok(principal)
    }

    /// Adds an existing principal to an existing organization.
    pub fn add_principal_to_organization(
        &self,
        principal: &PrincipalId,
        organization: &OrganizationId,
    ) -> Result<(), IdentityError> {
        let mut conn = self.locked_connection();
        let transaction = conn.transaction()?;
        require_organization(&transaction, organization)?;
        let principal_exists: Option<i64> = transaction
            .query_row(
                "SELECT 1 FROM users WHERE principal_id = ?1",
                rusqlite::params![principal.0],
                |row| row.get(0),
            )
            .optional()?;
        if principal_exists.is_none() {
            return Err(IdentityError::Storage(format!(
                "no such principal: {}",
                principal.0
            )));
        }
        transaction.execute(
            "INSERT OR IGNORE INTO organization_memberships (organization_id, principal_id, created_at_unix)
             VALUES (?1, ?2, ?3)",
            rusqlite::params![organization.0, principal.0, unix_now()],
        )?;
        transaction.commit()?;
        Ok(())
    }

    /// Issues a new bearer token for `principal` and returns its plaintext
    /// value. This is the only moment the plaintext is ever available: only
    /// its SHA-256 hash is persisted, so a stolen database backup cannot be
    /// used to authenticate.
    pub fn issue_token(&self, principal: &PrincipalId) -> Result<String, IdentityError> {
        let organizations: Vec<String> = {
            let conn = self.locked_connection();
            let mut statement = conn.prepare(
                "SELECT organization_id FROM organization_memberships WHERE principal_id = ?1 ORDER BY organization_id",
            )?;
            let organizations = statement
                .query_map(rusqlite::params![principal.0], |row| row.get(0))?
                .collect::<Result<Vec<_>, _>>()?;
            organizations
        };
        match organizations.as_slice() {
            [organization_id] => self.issue_token_for_organization(
                principal,
                &OrganizationId(organization_id.clone()),
            ),
            [] => Err(IdentityError::Storage(format!(
                "no such principal: {}",
                principal.0
            ))),
            _ => Err(IdentityError::Storage(format!(
                "principal {} belongs to multiple organizations; issue an organization-scoped token",
                principal.0
            ))),
        }
    }

    /// Issues a bearer token for one principal in one organization. Both the
    /// principal and organization must exist, and their membership is required.
    pub fn issue_token_for_organization(
        &self,
        principal: &PrincipalId,
        organization: &OrganizationId,
    ) -> Result<String, IdentityError> {
        let conn = self.locked_connection();
        let is_member: Option<i64> = conn
            .query_row(
                "SELECT 1 FROM organization_memberships WHERE principal_id = ?1 AND organization_id = ?2",
                rusqlite::params![principal.0, organization.0],
                |row| row.get(0),
            )
            .optional()?;
        if is_member.is_none() {
            return Err(IdentityError::Storage(format!(
                "principal {} is not a member of organization {}",
                principal.0, organization.0
            )));
        }

        let token = format!("{TOKEN_PREFIX}{}", random_hex(TOKEN_BYTES));
        conn.execute(
            "INSERT INTO tokens (token_hash, principal_id, organization_id, created_at_unix) VALUES (?1, ?2, ?3, ?4)",
            rusqlite::params![hash_token(&token), principal.0, organization.0, unix_now()],
        )?;
        Ok(token)
    }

    /// Resolves a bearer token to both the authenticated principal and its
    /// explicitly selected organization.
    pub fn resolve_context(&self, token: &str) -> Result<AuthenticatedContext, IdentityError> {
        let conn = self.locked_connection();
        let context: Option<(String, String)> = conn
            .query_row(
                "SELECT token.principal_id, token.organization_id
                 FROM tokens AS token
                 INNER JOIN organization_memberships AS membership
                   ON membership.principal_id = token.principal_id
                  AND membership.organization_id = token.organization_id
                 WHERE token.token_hash = ?1",
                rusqlite::params![hash_token(token)],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .optional()?;
        context
            .map(|(principal_id, organization_id)| AuthenticatedContext {
                principal_id: PrincipalId(principal_id),
                organization_id: OrganizationId(organization_id),
            })
            .ok_or(IdentityError::InvalidToken)
    }

    /// Revokes a token. A no-op if the token was already invalid.
    pub fn revoke_token(&self, token: &str) -> Result<(), IdentityError> {
        let conn = self.locked_connection();
        conn.execute(
            "DELETE FROM tokens WHERE token_hash = ?1",
            rusqlite::params![hash_token(token)],
        )?;
        Ok(())
    }
}

impl TokenStore for SqliteIdentityStore {
    fn resolve(&self, token: &str) -> Result<PrincipalId, IdentityError> {
        self.resolve_context(token)
            .map(|context| context.principal_id)
    }
}

fn initialize_schema(conn: &mut Connection) -> Result<(), IdentityError> {
    let schema_version: i64 = conn.pragma_query_value(None, "user_version", |row| row.get(0))?;
    if schema_version > IDENTITY_SCHEMA_VERSION {
        return Err(IdentityError::Storage(format!(
            "identity database schema version {schema_version} is newer than this binary supports ({IDENTITY_SCHEMA_VERSION})"
        )));
    }
    if !table_exists(conn, "tokens")? {
        create_schema_v2(conn)?;
    } else if !table_has_column(conn, "tokens", "organization_id")? {
        migrate_v1_to_v2(conn)?;
    } else {
        create_schema_v2(conn)?;
    }
    conn.pragma_update(None, "user_version", IDENTITY_SCHEMA_VERSION)?;
    verify_foreign_keys(conn)
}

fn create_schema_v2(conn: &Connection) -> Result<(), IdentityError> {
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS users (
            principal_id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            created_at_unix INTEGER NOT NULL
        );
        CREATE TABLE IF NOT EXISTS organizations (
            organization_id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            created_at_unix INTEGER NOT NULL
        );
        CREATE TABLE IF NOT EXISTS organization_memberships (
            organization_id TEXT NOT NULL REFERENCES organizations(organization_id),
            principal_id TEXT NOT NULL REFERENCES users(principal_id),
            created_at_unix INTEGER NOT NULL,
            PRIMARY KEY (organization_id, principal_id)
        );
        CREATE TABLE IF NOT EXISTS tokens (
            token_hash TEXT PRIMARY KEY,
            principal_id TEXT NOT NULL REFERENCES users(principal_id),
            organization_id TEXT NOT NULL REFERENCES organizations(organization_id),
            created_at_unix INTEGER NOT NULL,
            FOREIGN KEY (organization_id, principal_id)
                REFERENCES organization_memberships(organization_id, principal_id)
        );
        CREATE INDEX IF NOT EXISTS idx_tokens_principal ON tokens(principal_id);
        CREATE INDEX IF NOT EXISTS idx_memberships_principal ON organization_memberships(principal_id);",
    )?;
    Ok(())
}

fn migrate_v1_to_v2(conn: &mut Connection) -> Result<(), IdentityError> {
    let transaction = conn.transaction()?;
    transaction.execute_batch(
        "ALTER TABLE tokens ADD COLUMN organization_id TEXT;
        CREATE TABLE organizations (
            organization_id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            created_at_unix INTEGER NOT NULL
        );
        CREATE TABLE organization_memberships (
            organization_id TEXT NOT NULL REFERENCES organizations(organization_id),
            principal_id TEXT NOT NULL REFERENCES users(principal_id),
            created_at_unix INTEGER NOT NULL,
            PRIMARY KEY (organization_id, principal_id)
        );",
    )?;

    let principals: Vec<(String, String)> = {
        let mut statement =
            transaction.prepare("SELECT principal_id, name FROM users ORDER BY principal_id")?;
        let principals = statement
            .query_map([], |row| Ok((row.get(0)?, row.get(1)?)))?
            .collect::<Result<Vec<_>, _>>()?;
        principals
    };
    for (principal_id, name) in principals {
        let organization_id = random_organization_id();
        let now = unix_now();
        transaction.execute(
            "INSERT INTO organizations (organization_id, name, created_at_unix) VALUES (?1, ?2, ?3)",
            rusqlite::params![organization_id, format!("Personal organization for {name}"), now],
        )?;
        transaction.execute(
            "INSERT INTO organization_memberships (organization_id, principal_id, created_at_unix) VALUES (?1, ?2, ?3)",
            rusqlite::params![organization_id, principal_id, now],
        )?;
        transaction.execute(
            "UPDATE tokens SET organization_id = ?1 WHERE principal_id = ?2",
            rusqlite::params![organization_id, principal_id],
        )?;
    }
    transaction.execute_batch(
        "CREATE TABLE tokens_v2 (
            token_hash TEXT PRIMARY KEY,
            principal_id TEXT NOT NULL REFERENCES users(principal_id),
            organization_id TEXT NOT NULL REFERENCES organizations(organization_id),
            created_at_unix INTEGER NOT NULL,
            FOREIGN KEY (organization_id, principal_id)
                REFERENCES organization_memberships(organization_id, principal_id)
        );
        INSERT INTO tokens_v2 (token_hash, principal_id, organization_id, created_at_unix)
            SELECT token_hash, principal_id, organization_id, created_at_unix FROM tokens;
        DROP TABLE tokens;
        ALTER TABLE tokens_v2 RENAME TO tokens;
        CREATE INDEX idx_tokens_principal ON tokens(principal_id);
        CREATE INDEX idx_memberships_principal ON organization_memberships(principal_id);",
    )?;
    transaction.commit()?;
    Ok(())
}

fn table_exists(conn: &Connection, name: &str) -> Result<bool, IdentityError> {
    Ok(conn
        .query_row(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?1",
            rusqlite::params![name],
            |_| Ok(()),
        )
        .optional()?
        .is_some())
}

fn table_has_column(conn: &Connection, table: &str, column: &str) -> Result<bool, IdentityError> {
    let mut statement = conn.prepare(&format!("PRAGMA table_info({table})"))?;
    let columns = statement
        .query_map([], |row| row.get::<_, String>(1))?
        .collect::<Result<Vec<_>, _>>()?;
    Ok(columns.iter().any(|candidate| candidate == column))
}

fn verify_foreign_keys(conn: &Connection) -> Result<(), IdentityError> {
    let violation: Option<(String, i64, String)> = conn
        .query_row("PRAGMA foreign_key_check", [], |row| {
            Ok((row.get(0)?, row.get(1)?, row.get(2)?))
        })
        .optional()?;
    if let Some((table, row_id, parent)) = violation {
        return Err(IdentityError::Storage(format!(
            "identity database foreign-key violation in {table} row {row_id} referencing {parent}"
        )));
    }
    Ok(())
}

fn require_organization(
    transaction: &rusqlite::Transaction<'_>,
    organization: &OrganizationId,
) -> Result<(), IdentityError> {
    let exists: Option<i64> = transaction
        .query_row(
            "SELECT 1 FROM organizations WHERE organization_id = ?1",
            rusqlite::params![organization.0],
            |row| row.get(0),
        )
        .optional()?;
    if exists.is_none() {
        return Err(IdentityError::Storage(format!(
            "no such organization: {}",
            organization.0
        )));
    }
    Ok(())
}

fn hash_token(token: &str) -> String {
    to_hex(&Sha256::digest(token.as_bytes()))
}

fn random_id(label: &str) -> String {
    format!("{label}_{}", random_hex(PRINCIPAL_BYTES))
}

fn random_organization_id() -> String {
    format!("org_{}", random_hex(ORGANIZATION_BYTES))
}

fn random_hex(len_bytes: usize) -> String {
    let mut buf = vec![0u8; len_bytes];
    getrandom::fill(&mut buf).expect("OS random source unavailable");
    to_hex(&buf)
}

fn to_hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn unix_now() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("system clock is before the Unix epoch")
        .as_secs() as i64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn issued_token_resolves_to_the_issuing_principal() {
        let store = SqliteIdentityStore::open_in_memory().unwrap();
        let principal = store.create_user("ada").unwrap();
        let token = store.issue_token(&principal).unwrap();

        assert_eq!(store.resolve(&token).unwrap(), principal);
    }

    #[test]
    fn unknown_token_is_rejected() {
        let store = SqliteIdentityStore::open_in_memory().unwrap();
        let error = store.resolve("cme_not-a-real-token").unwrap_err();
        assert!(matches!(error, IdentityError::InvalidToken));
    }

    #[test]
    fn revoked_token_no_longer_resolves() {
        let store = SqliteIdentityStore::open_in_memory().unwrap();
        let principal = store.create_user("ada").unwrap();
        let token = store.issue_token(&principal).unwrap();

        store.revoke_token(&token).unwrap();

        assert!(matches!(
            store.resolve(&token).unwrap_err(),
            IdentityError::InvalidToken
        ));
    }

    #[test]
    fn revoking_an_unknown_token_is_not_an_error() {
        let store = SqliteIdentityStore::open_in_memory().unwrap();
        store.revoke_token("cme_never-issued").unwrap();
    }

    /// `open` asks SQLite for WAL mode explicitly (rather than assuming the
    /// pragma took effect) so a concurrent CLI process and a running `serve`
    /// can touch the same file without contending for the default
    /// rollback-journal lock. Prove it actually switches on a real file, not
    /// just that the call returns `Ok`.
    #[test]
    fn opening_a_file_backed_store_enables_wal_mode() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("identity.sqlite");

        let store = SqliteIdentityStore::open(&path).unwrap();

        let mode: String = store
            .conn
            .lock()
            .unwrap()
            .pragma_query_value(None, "journal_mode", |row| row.get(0))
            .unwrap();
        assert_eq!(mode.to_ascii_lowercase(), "wal");
    }

    #[test]
    fn issuing_a_token_for_an_unknown_principal_fails() {
        let store = SqliteIdentityStore::open_in_memory().unwrap();
        let ghost = PrincipalId("usr_does_not_exist".to_string());
        assert!(store.issue_token(&ghost).is_err());
    }

    #[test]
    fn a_principal_can_hold_multiple_independent_tokens() {
        let store = SqliteIdentityStore::open_in_memory().unwrap();
        let principal = store.create_user("ada").unwrap();
        let first = store.issue_token(&principal).unwrap();
        let second = store.issue_token(&principal).unwrap();

        assert_ne!(first, second);
        assert_eq!(store.resolve(&first).unwrap(), principal);
        assert_eq!(store.resolve(&second).unwrap(), principal);

        store.revoke_token(&first).unwrap();
        assert!(store.resolve(&first).is_err());
        assert_eq!(store.resolve(&second).unwrap(), principal);
    }

    #[test]
    fn tokens_are_not_stored_in_plaintext() {
        let store = SqliteIdentityStore::open_in_memory().unwrap();
        let principal = store.create_user("ada").unwrap();
        let token = store.issue_token(&principal).unwrap();

        let conn = store.conn.lock().unwrap();
        let stored_hash: String = conn
            .query_row("SELECT token_hash FROM tokens LIMIT 1", [], |row| {
                row.get(0)
            })
            .unwrap();
        assert_ne!(stored_hash, token);
        assert_eq!(stored_hash, hash_token(&token));
    }

    /// A `std::sync::Mutex` never un-poisons itself once a panic occurs while
    /// it is held. Giving up on a poisoned lock would permanently break every
    /// identity operation for the life of the process. Prove recovery works:
    /// poison the store's connection lock exactly like a panicking holder
    /// would, then confirm the store keeps working afterward.
    #[test]
    fn store_survives_a_poisoned_lock() {
        let store = std::sync::Arc::new(SqliteIdentityStore::open_in_memory().unwrap());
        let principal = store.create_user("ada").unwrap();

        let poison_store = std::sync::Arc::clone(&store);
        let poison_thread = std::thread::spawn(move || {
            let _guard = poison_store.conn.lock().unwrap();
            panic!("deliberately poisoning the identity connection lock for this test");
        });
        assert!(
            poison_thread.join().is_err(),
            "the poisoning thread must have panicked"
        );

        let token = store.issue_token(&principal).unwrap();
        assert_eq!(store.resolve(&token).unwrap(), principal);
    }

    #[test]
    fn distinct_principals_and_tokens_are_generated_each_time() {
        let store = SqliteIdentityStore::open_in_memory().unwrap();
        let a = store.create_user("ada").unwrap();
        let b = store.create_user("grace").unwrap();
        assert_ne!(a, b);
    }

    #[test]
    fn a_token_resolves_to_one_explicit_organization_context() {
        let store = SqliteIdentityStore::open_in_memory().unwrap();
        let organization = store.create_organization("Acme").unwrap();
        let principal = store
            .create_user_in_organization("ada", &organization)
            .unwrap();
        let token = store
            .issue_token_for_organization(&principal, &organization)
            .unwrap();

        let context = store.resolve_context(&token).unwrap();
        assert_eq!(context.principal_id(), &principal);
        assert_eq!(context.organization_id(), &organization);
        assert_eq!(store.resolve(&token).unwrap(), principal);
    }

    #[test]
    fn a_multi_organization_principal_needs_an_explicitly_scoped_token() {
        let store = SqliteIdentityStore::open_in_memory().unwrap();
        let first = store.create_organization("Acme").unwrap();
        let second = store.create_organization("Example").unwrap();
        let principal = store.create_user_in_organization("ada", &first).unwrap();
        store
            .add_principal_to_organization(&principal, &second)
            .unwrap();
        store
            .add_principal_to_organization(&principal, &second)
            .unwrap();

        let error = store.issue_token(&principal).unwrap_err();
        assert!(error.to_string().contains("multiple organizations"));

        let first_token = store
            .issue_token_for_organization(&principal, &first)
            .unwrap();
        let second_token = store
            .issue_token_for_organization(&principal, &second)
            .unwrap();
        assert_eq!(
            store
                .resolve_context(&first_token)
                .unwrap()
                .organization_id(),
            &first
        );
        assert_eq!(
            store
                .resolve_context(&second_token)
                .unwrap()
                .organization_id(),
            &second
        );
    }

    #[test]
    fn issuing_an_organization_token_requires_membership() {
        let store = SqliteIdentityStore::open_in_memory().unwrap();
        let first = store.create_organization("Acme").unwrap();
        let second = store.create_organization("Example").unwrap();
        let principal = store.create_user_in_organization("ada", &first).unwrap();

        let error = store
            .issue_token_for_organization(&principal, &second)
            .unwrap_err();
        assert!(error.to_string().contains("is not a member"));
    }

    #[test]
    fn legacy_database_migration_preserves_existing_token_authentication() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("identity.sqlite");
        let principal = PrincipalId::new("usr_legacy".to_string());
        let token = "cme_legacy_token";
        {
            let connection = Connection::open(&path).unwrap();
            connection
                .execute_batch(
                    "PRAGMA foreign_keys = ON;
                     CREATE TABLE users (
                        principal_id TEXT PRIMARY KEY,
                        name TEXT NOT NULL,
                        created_at_unix INTEGER NOT NULL
                     );
                     CREATE TABLE tokens (
                        token_hash TEXT PRIMARY KEY,
                        principal_id TEXT NOT NULL REFERENCES users(principal_id),
                        created_at_unix INTEGER NOT NULL
                     );
                     CREATE INDEX idx_tokens_principal ON tokens(principal_id);",
                )
                .unwrap();
            connection
                .execute(
                    "INSERT INTO users (principal_id, name, created_at_unix) VALUES (?1, ?2, ?3)",
                    rusqlite::params![principal.as_str(), "ada", 1_i64],
                )
                .unwrap();
            connection
                .execute(
                    "INSERT INTO tokens (token_hash, principal_id, created_at_unix) VALUES (?1, ?2, ?3)",
                    rusqlite::params![hash_token(token), principal.as_str(), 1_i64],
                )
                .unwrap();
        }

        let store = SqliteIdentityStore::open(&path).unwrap();
        let context = store.resolve_context(token).unwrap();

        assert_eq!(context.principal_id(), &principal);
        assert!(context.organization_id().as_str().starts_with("org_"));
        assert_eq!(store.resolve(token).unwrap(), principal);
        assert!(store.issue_token(context.principal_id()).is_ok());
    }

    #[test]
    fn newer_database_schema_is_rejected_without_modification() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("identity.sqlite");
        let connection = Connection::open(&path).unwrap();
        connection
            .pragma_update(None, "user_version", IDENTITY_SCHEMA_VERSION + 1)
            .unwrap();
        drop(connection);

        let error = match SqliteIdentityStore::open(&path) {
            Ok(_) => panic!("a newer identity schema must be rejected"),
            Err(error) => error,
        };
        assert!(error
            .to_string()
            .contains("newer than this binary supports"));
    }
}
