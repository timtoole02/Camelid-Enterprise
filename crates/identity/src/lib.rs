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
//! contract for callers that have not yet adopted organization-aware policy.

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
#[non_exhaustive]
pub enum IdentityError {
    /// The token does not resolve to any principal (never issued, or
    /// revoked). Deliberately does not distinguish the two: a caller must not
    /// be able to tell "never existed" from "revoked".
    InvalidToken,
    UnknownPrincipal,
    UnknownOrganization,
    NoOrganizationMembership,
    AmbiguousOrganization,
    NotOrganizationMember,
    UnsupportedSchemaVersion {
        found: i64,
        supported: i64,
    },
    MigrationIntegrity(String),
    Storage(String),
}

impl fmt::Display for IdentityError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidToken => write!(f, "token does not resolve to any principal"),
            Self::UnknownPrincipal => write!(f, "principal does not exist"),
            Self::UnknownOrganization => write!(f, "organization does not exist"),
            Self::NoOrganizationMembership => write!(f, "principal belongs to no organizations"),
            Self::AmbiguousOrganization => write!(
                f,
                "principal belongs to multiple organizations; select an organization explicitly"
            ),
            Self::NotOrganizationMember => {
                write!(f, "principal is not a member of the organization")
            }
            Self::UnsupportedSchemaVersion { found, supported } => write!(
                f,
                "identity database schema version {found} is newer than this binary supports ({supported})"
            ),
            Self::MigrationIntegrity(message) => {
                write!(f, "identity database cannot be migrated safely: {message}")
            }
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
        // Do this read-only compatibility check before the WAL pragma. A
        // binary must not alter the journal mode of a database it cannot
        // understand or migrate.
        ensure_supported_schema_version(&conn)?;
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
    /// Operations can use SQLite transactions while holding this guard. If a
    /// panic unwinds through one, `Transaction::drop` rolls it back before the
    /// mutex guard is released; rollback failures cannot be reported during
    /// drop, but SQLite will not leave that uncommitted transaction active on
    /// the reused connection. Reusing a recovered guard is therefore safe for
    /// this connection-local, self-contained unit of work.
    fn locked_connection(&self) -> MutexGuard<'_, Connection> {
        self.conn
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// Creates a user, its personal organization, and their membership in one
    /// transaction. `name` is a display label only; it is never a lookup key.
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

    /// Adds an existing principal to an existing organization. Repeating the
    /// same addition is idempotent.
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
            return Err(IdentityError::UnknownPrincipal);
        }
        transaction.execute(
            "INSERT OR IGNORE INTO organization_memberships (organization_id, principal_id, created_at_unix)
             VALUES (?1, ?2, ?3)",
            rusqlite::params![organization.0, principal.0, unix_now()],
        )?;
        transaction.commit()?;
        Ok(())
    }

    /// Returns every organization the principal belongs to, in stable id order.
    pub fn organizations_for_principal(
        &self,
        principal: &PrincipalId,
    ) -> Result<Vec<OrganizationId>, IdentityError> {
        let conn = self.locked_connection();
        let exists: Option<i64> = conn
            .query_row(
                "SELECT 1 FROM users WHERE principal_id = ?1",
                rusqlite::params![principal.0],
                |row| row.get(0),
            )
            .optional()?;
        if exists.is_none() {
            return Err(IdentityError::UnknownPrincipal);
        }
        let mut statement = conn.prepare(
            "SELECT organization_id FROM organization_memberships
             WHERE principal_id = ?1 ORDER BY organization_id",
        )?;
        let organizations = statement
            .query_map(rusqlite::params![principal.0], |row| {
                Ok(OrganizationId(row.get(0)?))
            })?
            .collect::<Result<Vec<_>, _>>()?;
        Ok(organizations)
    }

    /// Removes a principal from an organization and revokes every token scoped
    /// to that membership. Repeating a removal is reported as a typed caller
    /// error rather than silently leaving uncertain access state.
    pub fn remove_principal_from_organization(
        &self,
        principal: &PrincipalId,
        organization: &OrganizationId,
    ) -> Result<(), IdentityError> {
        let mut conn = self.locked_connection();
        let transaction = conn.transaction()?;
        if !principal_exists(&transaction, principal)? {
            return Err(IdentityError::UnknownPrincipal);
        }
        require_organization(&transaction, organization)?;
        let removed = transaction.execute(
            "DELETE FROM organization_memberships
             WHERE principal_id = ?1 AND organization_id = ?2",
            rusqlite::params![principal.0, organization.0],
        )?;
        if removed == 0 {
            return Err(IdentityError::NotOrganizationMember);
        }
        transaction.commit()?;
        Ok(())
    }

    /// Issues a token for a principal that belongs to exactly one organization.
    /// Multi-organization principals must use [`Self::issue_token_for_organization`]
    /// so the selected tenant is unambiguous.
    pub fn issue_token(&self, principal: &PrincipalId) -> Result<String, IdentityError> {
        let mut conn = self.locked_connection();
        let transaction = conn.transaction()?;
        let organizations: Vec<String> = {
            let mut statement = transaction.prepare(
                "SELECT organization_id FROM organization_memberships WHERE principal_id = ?1 ORDER BY organization_id",
            )?;
            let organizations = statement
                .query_map(rusqlite::params![principal.0], |row| row.get(0))?
                .collect::<Result<Vec<_>, _>>()?;
            organizations
        };
        let organization = match organizations.as_slice() {
            [organization_id] => OrganizationId(organization_id.clone()),
            [] if principal_exists(&transaction, principal)? => {
                return Err(IdentityError::NoOrganizationMembership);
            }
            [] => return Err(IdentityError::UnknownPrincipal),
            _ => return Err(IdentityError::AmbiguousOrganization),
        };
        let token = issue_token_in_transaction(&transaction, principal, &organization)?;
        transaction.commit()?;
        Ok(token)
    }

    /// Issues a bearer token for one principal in one organization. Both the
    /// principal and organization must exist, and their membership is required.
    pub fn issue_token_for_organization(
        &self,
        principal: &PrincipalId,
        organization: &OrganizationId,
    ) -> Result<String, IdentityError> {
        let mut conn = self.locked_connection();
        let transaction = conn.transaction()?;
        let token = issue_token_in_transaction(&transaction, principal, organization)?;
        transaction.commit()?;
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

fn ensure_supported_schema_version(conn: &Connection) -> Result<(), IdentityError> {
    let schema_version: i64 = conn.pragma_query_value(None, "user_version", |row| row.get(0))?;
    if schema_version > IDENTITY_SCHEMA_VERSION {
        return Err(IdentityError::UnsupportedSchemaVersion {
            found: schema_version,
            supported: IDENTITY_SCHEMA_VERSION,
        });
    }
    Ok(())
}

fn principal_exists(
    transaction: &rusqlite::Transaction<'_>,
    principal: &PrincipalId,
) -> Result<bool, IdentityError> {
    Ok(transaction
        .query_row(
            "SELECT 1 FROM users WHERE principal_id = ?1",
            rusqlite::params![principal.0],
            |_| Ok(()),
        )
        .optional()?
        .is_some())
}

fn issue_token_in_transaction(
    transaction: &rusqlite::Transaction<'_>,
    principal: &PrincipalId,
    organization: &OrganizationId,
) -> Result<String, IdentityError> {
    if !principal_exists(transaction, principal)? {
        return Err(IdentityError::UnknownPrincipal);
    }
    require_organization(transaction, organization)?;
    let is_member: Option<i64> = transaction
        .query_row(
            "SELECT 1 FROM organization_memberships WHERE principal_id = ?1 AND organization_id = ?2",
            rusqlite::params![principal.0, organization.0],
            |row| row.get(0),
        )
        .optional()?;
    if is_member.is_none() {
        return Err(IdentityError::NotOrganizationMember);
    }

    let token = format!("{TOKEN_PREFIX}{}", random_hex(TOKEN_BYTES));
    transaction.execute(
        "INSERT INTO tokens (token_hash, principal_id, organization_id, created_at_unix) VALUES (?1, ?2, ?3, ?4)",
        rusqlite::params![hash_token(&token), principal.0, organization.0, unix_now()],
    )?;
    Ok(token)
}

fn initialize_schema(conn: &mut Connection) -> Result<(), IdentityError> {
    let schema_version: i64 = conn.pragma_query_value(None, "user_version", |row| row.get(0))?;
    match schema_version {
        0 => initialize_unversioned_database(conn),
        1 => migrate_v1_to_v2(conn),
        2 => validate_v2_schema(conn),
        _ => Err(IdentityError::UnsupportedSchemaVersion {
            found: schema_version,
            supported: IDENTITY_SCHEMA_VERSION,
        }),
    }
}

fn initialize_unversioned_database(conn: &mut Connection) -> Result<(), IdentityError> {
    let users_exists = table_exists(conn, "users")?;
    let tokens_exists = table_exists(conn, "tokens")?;
    let organizations_exists = table_exists(conn, "organizations")?;
    let memberships_exist = table_exists(conn, "organization_memberships")?;
    match (
        users_exists,
        tokens_exists,
        organizations_exists,
        memberships_exist,
    ) {
        (false, false, false, false) => {
            create_schema_v2(conn)?;
            conn.pragma_update(None, "user_version", IDENTITY_SCHEMA_VERSION)?;
            Ok(())
        }
        (true, true, false, false) => migrate_v1_to_v2(conn),
        _ => Err(IdentityError::MigrationIntegrity(
            "unversioned database does not match the supported legacy identity schema".into(),
        )),
    }
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
            principal_id TEXT NOT NULL,
            organization_id TEXT NOT NULL,
            created_at_unix INTEGER NOT NULL,
            FOREIGN KEY (organization_id, principal_id)
                REFERENCES organization_memberships(organization_id, principal_id)
                ON DELETE CASCADE
        );
        CREATE INDEX IF NOT EXISTS idx_tokens_principal ON tokens(principal_id);
        CREATE INDEX IF NOT EXISTS idx_memberships_principal ON organization_memberships(principal_id);",
    )?;
    Ok(())
}

fn migrate_v1_to_v2(conn: &mut Connection) -> Result<(), IdentityError> {
    let transaction = conn.transaction()?;
    let orphaned_principal: Option<String> = transaction
        .query_row(
            "SELECT tokens.principal_id
             FROM tokens
             LEFT JOIN users ON users.principal_id = tokens.principal_id
             WHERE users.principal_id IS NULL
             LIMIT 1",
            [],
            |row| row.get(0),
        )
        .optional()?;
    if orphaned_principal.is_some() {
        return Err(IdentityError::MigrationIntegrity(
            "a legacy token references a missing principal".into(),
        ));
    }
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
            principal_id TEXT NOT NULL,
            organization_id TEXT NOT NULL,
            created_at_unix INTEGER NOT NULL,
            FOREIGN KEY (organization_id, principal_id)
                REFERENCES organization_memberships(organization_id, principal_id)
                ON DELETE CASCADE
        );
        INSERT INTO tokens_v2 (token_hash, principal_id, organization_id, created_at_unix)
            SELECT token_hash, principal_id, organization_id, created_at_unix FROM tokens;
        DROP TABLE tokens;
        ALTER TABLE tokens_v2 RENAME TO tokens;
        CREATE INDEX idx_tokens_principal ON tokens(principal_id);
        CREATE INDEX idx_memberships_principal ON organization_memberships(principal_id);
        PRAGMA user_version = 2;",
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

fn validate_v2_schema(conn: &Connection) -> Result<(), IdentityError> {
    for query in [
        "SELECT principal_id, name, created_at_unix FROM users LIMIT 0",
        "SELECT organization_id, name, created_at_unix FROM organizations LIMIT 0",
        "SELECT organization_id, principal_id, created_at_unix FROM organization_memberships LIMIT 0",
        "SELECT token_hash, principal_id, organization_id, created_at_unix FROM tokens LIMIT 0",
    ] {
        conn.prepare(query)?;
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
        return Err(IdentityError::UnknownOrganization);
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
        assert!(matches!(
            store.issue_token(&ghost).unwrap_err(),
            IdentityError::UnknownPrincipal
        ));
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
        let a_organizations = store.organizations_for_principal(&a).unwrap();
        let b_organizations = store.organizations_for_principal(&b).unwrap();
        assert_eq!(a_organizations.len(), 1);
        assert_eq!(b_organizations.len(), 1);
        assert_ne!(a_organizations, b_organizations);
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

        assert_eq!(
            store.organizations_for_principal(&principal).unwrap(),
            vec![first.clone(), second.clone()]
        );
        assert!(matches!(
            store.issue_token(&principal).unwrap_err(),
            IdentityError::AmbiguousOrganization
        ));

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

        assert!(matches!(
            store
                .issue_token_for_organization(&principal, &second)
                .unwrap_err(),
            IdentityError::NotOrganizationMember
        ));
    }

    #[test]
    fn legacy_database_migration_preserves_every_users_tokens() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("identity.sqlite");
        let ada = PrincipalId::new("usr_legacy_ada".to_string());
        let grace = PrincipalId::new("usr_legacy_grace".to_string());
        let tokens = [
            ("cme_legacy_ada_one", &ada),
            ("cme_legacy_ada_two", &ada),
            ("cme_legacy_grace", &grace),
        ];
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
            for (principal, name) in [(&ada, "ada"), (&grace, "grace")] {
                connection
                    .execute(
                        "INSERT INTO users (principal_id, name, created_at_unix) VALUES (?1, ?2, ?3)",
                        rusqlite::params![principal.as_str(), name, 1_i64],
                    )
                    .unwrap();
            }
            for &(token, principal) in &tokens {
                connection
                    .execute(
                        "INSERT INTO tokens (token_hash, principal_id, created_at_unix) VALUES (?1, ?2, ?3)",
                        rusqlite::params![hash_token(token), principal.as_str(), 1_i64],
                    )
                    .unwrap();
            }
        }

        let store = SqliteIdentityStore::open(&path).unwrap();
        let ada_organization = store.organizations_for_principal(&ada).unwrap();
        let grace_organization = store.organizations_for_principal(&grace).unwrap();
        assert_eq!(ada_organization.len(), 1);
        assert_eq!(grace_organization.len(), 1);
        assert_ne!(ada_organization, grace_organization);
        for &(token, principal) in &tokens {
            let context = store.resolve_context(token).unwrap();
            assert_eq!(context.principal_id(), principal);
            assert_eq!(store.resolve(token).unwrap(), *principal);
        }
    }

    #[test]
    fn newer_database_schema_is_rejected_without_modification() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("identity.sqlite");
        let connection = Connection::open(&path).unwrap();
        connection
            .execute_batch(
                "CREATE TABLE sentinel (value TEXT); INSERT INTO sentinel VALUES ('unchanged');",
            )
            .unwrap();
        connection
            .pragma_update(None, "user_version", IDENTITY_SCHEMA_VERSION + 1)
            .unwrap();
        drop(connection);
        let before = std::fs::read(&path).unwrap();
        assert!(!path.with_extension("sqlite-wal").exists());
        assert!(!path.with_extension("sqlite-shm").exists());

        let error = match SqliteIdentityStore::open(&path) {
            Ok(_) => panic!("a newer identity schema must be rejected"),
            Err(error) => error,
        };
        assert!(matches!(
            error,
            IdentityError::UnsupportedSchemaVersion {
                found,
                supported: IDENTITY_SCHEMA_VERSION
            } if found == IDENTITY_SCHEMA_VERSION + 1
        ));
        assert_eq!(std::fs::read(&path).unwrap(), before);
        let connection = Connection::open(&path).unwrap();
        assert_eq!(
            connection
                .pragma_query_value(None, "user_version", |row| row.get::<_, i64>(0))
                .unwrap(),
            IDENTITY_SCHEMA_VERSION + 1
        );
        assert_eq!(
            connection
                .query_row("SELECT value FROM sentinel", [], |row| row
                    .get::<_, String>(0))
                .unwrap(),
            "unchanged"
        );
    }

    #[test]
    fn removing_membership_revokes_only_that_organizations_tokens() {
        let store = SqliteIdentityStore::open_in_memory().unwrap();
        let first = store.create_organization("Acme").unwrap();
        let second = store.create_organization("Example").unwrap();
        let principal = store.create_user_in_organization("ada", &first).unwrap();
        store
            .add_principal_to_organization(&principal, &second)
            .unwrap();
        let first_token = store
            .issue_token_for_organization(&principal, &first)
            .unwrap();
        let second_token = store
            .issue_token_for_organization(&principal, &second)
            .unwrap();

        store
            .remove_principal_from_organization(&principal, &first)
            .unwrap();

        assert!(matches!(
            store.resolve_context(&first_token),
            Err(IdentityError::InvalidToken)
        ));
        assert_eq!(
            store
                .resolve_context(&second_token)
                .unwrap()
                .organization_id(),
            &second
        );
        assert!(matches!(
            store
                .remove_principal_from_organization(&principal, &first)
                .unwrap_err(),
            IdentityError::NotOrganizationMember
        ));
    }

    #[test]
    fn context_resolution_rejects_a_corrupt_token_without_membership() {
        let store = SqliteIdentityStore::open_in_memory().unwrap();
        let principal = store.create_user("ada").unwrap();
        let token = store.issue_token(&principal).unwrap();
        let organization = store
            .organizations_for_principal(&principal)
            .unwrap()
            .pop()
            .unwrap();
        let conn = store.locked_connection();
        conn.pragma_update(None, "foreign_keys", "OFF").unwrap();
        conn.execute(
            "DELETE FROM organization_memberships WHERE principal_id = ?1 AND organization_id = ?2",
            rusqlite::params![principal.as_str(), organization.as_str()],
        )
        .unwrap();
        conn.pragma_update(None, "foreign_keys", "ON").unwrap();
        drop(conn);

        assert!(matches!(
            store.resolve_context(&token),
            Err(IdentityError::InvalidToken)
        ));
    }

    #[test]
    fn concurrent_scoped_token_issuance_remains_consistent() {
        const ISSUERS: usize = 48;
        let store = std::sync::Arc::new(SqliteIdentityStore::open_in_memory().unwrap());
        let organization = store.create_organization("Acme").unwrap();
        let principal = store
            .create_user_in_organization("ada", &organization)
            .unwrap();
        let mut threads = Vec::new();
        for _ in 0..ISSUERS {
            let store = std::sync::Arc::clone(&store);
            let principal = principal.clone();
            let organization = organization.clone();
            threads.push(std::thread::spawn(move || {
                store
                    .issue_token_for_organization(&principal, &organization)
                    .unwrap()
            }));
        }
        let tokens: Vec<String> = threads
            .into_iter()
            .map(|thread| thread.join().unwrap())
            .collect();
        assert_eq!(tokens.len(), ISSUERS);
        assert_eq!(
            tokens
                .iter()
                .collect::<std::collections::HashSet<_>>()
                .len(),
            ISSUERS
        );
        for token in tokens {
            assert_eq!(
                store.resolve_context(&token).unwrap().organization_id(),
                &organization
            );
        }
    }
}
