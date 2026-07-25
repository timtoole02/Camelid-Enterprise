//! Local identity primitive for Camelid Enterprise.
//!
//! This crate deliberately does not model orgs, roles, quotas, or federation
//! yet. It provides exactly one indirection every later identity feature (the
//! gateway's auth enforcement, per-user quotas, receipts-with-identity, and so
//! on) is built on:
//!
//! **an opaque bearer token resolves to an opaque [`PrincipalId`], and
//! nothing else.**
//!
//! Callers never see a raw database row id as a principal, and the store
//! never compares plaintext tokens: only a token's SHA-256 hash is ever
//! persisted or looked up. This keeps the shape stable while the storage
//! backend, the identity model (users today, orgs/RBAC/SSO later), and the
//! enforcement point (nothing today, the gateway later) all change
//! independently.

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
const TOKEN_PREFIX: &str = "cme_"; // Camelid Enterprise token marker

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

    fn from_connection(conn: Connection) -> Result<Self, IdentityError> {
        conn.execute_batch(
            "PRAGMA foreign_keys = ON;
            CREATE TABLE IF NOT EXISTS users (
                principal_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                created_at_unix INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS tokens (
                token_hash TEXT PRIMARY KEY,
                principal_id TEXT NOT NULL REFERENCES users(principal_id),
                created_at_unix INTEGER NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_tokens_principal ON tokens(principal_id);",
        )?;
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
        let conn = self.locked_connection();
        conn.execute(
            "INSERT INTO users (principal_id, name, created_at_unix) VALUES (?1, ?2, ?3)",
            rusqlite::params![principal.0, name, unix_now()],
        )?;
        Ok(principal)
    }

    /// Issues a new bearer token for `principal` and returns its plaintext
    /// value. This is the only moment the plaintext is ever available: only
    /// its SHA-256 hash is persisted, so a stolen database backup cannot be
    /// used to authenticate.
    pub fn issue_token(&self, principal: &PrincipalId) -> Result<String, IdentityError> {
        let conn = self.locked_connection();
        let exists: Option<i64> = conn
            .query_row(
                "SELECT 1 FROM users WHERE principal_id = ?1",
                rusqlite::params![principal.0],
                |row| row.get(0),
            )
            .optional()?;
        if exists.is_none() {
            return Err(IdentityError::Storage(format!(
                "no such principal: {}",
                principal.0
            )));
        }

        let token = format!("{TOKEN_PREFIX}{}", random_hex(TOKEN_BYTES));
        conn.execute(
            "INSERT INTO tokens (token_hash, principal_id, created_at_unix) VALUES (?1, ?2, ?3)",
            rusqlite::params![hash_token(&token), principal.0, unix_now()],
        )?;
        Ok(token)
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
        let conn = self.locked_connection();
        let principal_id: Option<String> = conn
            .query_row(
                "SELECT principal_id FROM tokens WHERE token_hash = ?1",
                rusqlite::params![hash_token(token)],
                |row| row.get(0),
            )
            .optional()?;
        principal_id
            .map(PrincipalId)
            .ok_or(IdentityError::InvalidToken)
    }
}

fn hash_token(token: &str) -> String {
    to_hex(&Sha256::digest(token.as_bytes()))
}

fn random_id(label: &str) -> String {
    format!("{label}_{}", random_hex(PRINCIPAL_BYTES))
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
}
