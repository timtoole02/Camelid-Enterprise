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
//!
//! A token may carry an expiry, and any token can be exchanged for a fresh
//! secret in one transaction. Both are opt-in and neither changes what an
//! existing credential does: a token issued without an expiry never expires,
//! which is exactly what every token issued before this existed already did.
//!
//! Three bounds on that, none of them accidental:
//!
//! - Expiry is evaluated against the local system clock, so a clock moved
//!   backwards extends every outstanding token by the size of the move.
//!   Revocation consults no clock and is the control that does not share this
//!   property.
//! - Rotation needs the plaintext of the token being replaced, so it is
//!   reachable only by something holding that credential -- today, an operator
//!   at the machine running the CLI. There is no remote refresh: a client that
//!   is told its token expired cannot obtain a replacement over the network.
//! - Lapsed tokens are not collected. An expired row stays until it is rotated
//!   or its membership is removed, and [`SqliteIdentityStore::resolve_context`]
//!   keeps finding it in order to report [`IdentityError::ExpiredToken`]
//!   rather than a misleading [`IdentityError::InvalidToken`]. Short lifetimes
//!   therefore accumulate rows. That is a deliberate trade -- an accurate
//!   refusal over a self-pruning table -- and sweeping them is left to a later
//!   change with a policy behind it, not folded into the read path where it
//!   would let anyone force a write by replaying a dead token.

use rusqlite::{Connection, OptionalExtension};
use sha2::{Digest, Sha256};
use std::fmt;
use std::num::NonZeroU64;
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
    /// The token was issued and its membership is intact, but its expiry has
    /// passed.
    ///
    /// Deliberately distinct from [`Self::InvalidToken`], which the variant
    /// above is careful not to be. The reasoning differs because the audience
    /// does: a legitimate client cannot act on "unauthorized" but can act on
    /// "expired" by obtaining a new credential, and that is the entire point
    /// of issuing credentials with a lifetime. What it discloses is bounded to
    /// the bearer of one specific token: that the token they already hold was
    /// once real. Separating the two is only an oracle to a caller who can put
    /// a *guessed* token into it, and a guess must first land on one of 2^256
    /// values before the distinction says anything at all.
    ExpiredToken,
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
            Self::ExpiredToken => write!(f, "token has expired"),
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
/// The only contract: token in, principal out, or a refusal. A token that is
/// unknown or revoked is [`IdentityError::InvalidToken`]; one that was issued
/// with a lifetime that has since passed is [`IdentityError::ExpiredToken`].
/// Implementations must never expose the storage row backing a token.
pub trait TokenStore: Send + Sync {
    fn resolve(&self, token: &str) -> Result<PrincipalId, IdentityError>;
}

const TOKEN_BYTES: usize = 32; // 256-bit bearer token
const PRINCIPAL_BYTES: usize = 16; // 128-bit opaque principal id
const ORGANIZATION_BYTES: usize = 16; // 128-bit opaque organization id
const TOKEN_PREFIX: &str = "cme_"; // Camelid Enterprise token marker
const IDENTITY_SCHEMA_VERSION: i64 = 3;

/// The absolute expiry instant `seconds` from now, for callers that think in
/// lifetimes rather than in instants.
///
/// Expiry is stored absolute rather than as a duration so that a token's
/// validity does not depend on when it is next read. The lifetime is nonzero
/// because a zero-second lifetime issues a credential that is already dead:
/// that is a misconfiguration, and it is better refused where it is written
/// than honored at the point of use.
///
/// Both this and expiry checking read the system clock, so a backwards clock
/// jump extends the validity of every outstanding token by the size of the
/// jump. Revocation, which does not consult the clock, remains the control
/// that does not depend on it.
pub fn expires_in(seconds: NonZeroU64) -> i64 {
    unix_now().saturating_add(i64::try_from(seconds.get()).unwrap_or(i64::MAX))
}

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
        let transaction = begin_write(&mut conn)?;
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
        let transaction = begin_write(&mut conn)?;
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
        let transaction = begin_write(&mut conn)?;
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
        let transaction = begin_write(&mut conn)?;
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

    /// Issues a non-expiring token for a principal that belongs to exactly one
    /// organization. Multi-organization principals must use
    /// [`Self::issue_token_for_organization`] so the selected tenant is
    /// unambiguous.
    pub fn issue_token(&self, principal: &PrincipalId) -> Result<String, IdentityError> {
        self.issue_token_expiring_at(principal, None)
    }

    /// Issues a token for a principal that belongs to exactly one organization,
    /// valid until `expires_at_unix`. `None` issues a token that never expires.
    pub fn issue_token_expiring_at(
        &self,
        principal: &PrincipalId,
        expires_at_unix: Option<i64>,
    ) -> Result<String, IdentityError> {
        let mut conn = self.locked_connection();
        let transaction = begin_write(&mut conn)?;
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
        let token =
            issue_token_in_transaction(&transaction, principal, &organization, expires_at_unix)?;
        transaction.commit()?;
        Ok(token)
    }

    /// Issues a non-expiring bearer token for one principal in one
    /// organization. Both the principal and organization must exist, and their
    /// membership is required.
    pub fn issue_token_for_organization(
        &self,
        principal: &PrincipalId,
        organization: &OrganizationId,
    ) -> Result<String, IdentityError> {
        self.issue_token_for_organization_expiring_at(principal, organization, None)
    }

    /// Issues a bearer token for one principal in one organization, valid until
    /// `expires_at_unix`. `None` issues a token that never expires.
    pub fn issue_token_for_organization_expiring_at(
        &self,
        principal: &PrincipalId,
        organization: &OrganizationId,
        expires_at_unix: Option<i64>,
    ) -> Result<String, IdentityError> {
        let mut conn = self.locked_connection();
        let transaction = begin_write(&mut conn)?;
        let token =
            issue_token_in_transaction(&transaction, principal, organization, expires_at_unix)?;
        transaction.commit()?;
        Ok(token)
    }

    /// Exchanges a valid token for a fresh secret with the same principal and
    /// organization, and invalidates the presented one. `expires_at_unix`
    /// applies to the replacement only; `None` makes the replacement
    /// non-expiring regardless of what the presented token carried.
    ///
    /// Issue and revoke happen in one transaction, so there is no instant at
    /// which both secrets are live and none at which neither is: a caller that
    /// receives an error still holds a working token, and a caller that
    /// receives a token holds the only working one. That is what makes this
    /// safe to run against a credential currently in use.
    ///
    /// Rotation requires the plaintext of the token being replaced, exactly as
    /// [`Self::revoke_token`] does. It is a credential refresh for whoever
    /// holds the credential, not an administrative reset of someone else's.
    pub fn rotate_token(
        &self,
        token: &str,
        expires_at_unix: Option<i64>,
    ) -> Result<String, IdentityError> {
        let mut conn = self.locked_connection();
        let transaction = begin_write(&mut conn)?;
        // An expired token cannot buy a live one. Otherwise expiry would bound
        // only the credential rather than the access it grants, and any lapsed
        // token would remain a permanent key to a new one.
        let credential = lookup_credential(&transaction, token)?.still_valid(unix_now())?;
        let principal = PrincipalId(credential.principal_id);
        let organization = OrganizationId(credential.organization_id);
        let replacement =
            issue_token_in_transaction(&transaction, &principal, &organization, expires_at_unix)?;
        let retired = transaction.execute(
            "DELETE FROM tokens WHERE token_hash = ?1",
            rusqlite::params![hash_token(token)],
        )?;
        // The lookup above already proved this row exists and this transaction
        // holds the write lock, so this cannot currently be anything but 1.
        // It is asserted rather than assumed because the invariant -- that a
        // rotation retires exactly the credential it replaced, no more and no
        // fewer -- is the reason a caller can trust the operation at all, and
        // a future change that broke it would otherwise do so silently.
        if retired != 1 {
            return Err(IdentityError::Storage(format!(
                "rotation retired {retired} tokens instead of exactly 1"
            )));
        }
        transaction.commit()?;
        Ok(replacement)
    }

    /// Resolves a bearer token to both the authenticated principal and its
    /// explicitly selected organization.
    pub fn resolve_context(&self, token: &str) -> Result<AuthenticatedContext, IdentityError> {
        let conn = self.locked_connection();
        let credential = lookup_credential(&conn, token)?.still_valid(unix_now())?;
        Ok(AuthenticatedContext {
            principal_id: PrincipalId(credential.principal_id),
            organization_id: OrganizationId(credential.organization_id),
        })
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

/// A token row that exists and whose membership is still intact.
///
/// Existence and validity are separated on purpose. The lookup answers "is
/// this a real credential for a current member", which is a question about
/// stored state; expiry answers "is it usable now", which is a question about
/// the clock. Rotation needs both answers and needs them to be different
/// answers, so the split is in the type rather than buried in one SQL
/// predicate.
struct Credential {
    principal_id: String,
    organization_id: String,
    expires_at_unix: Option<i64>,
}

impl Credential {
    /// Consumes the credential if it has not expired at `now`.
    ///
    /// The expiry instant is exclusive: a token that expires at `T` is already
    /// refused at `T`. Between two defensible boundaries, this picks the one
    /// that never leaves a credential usable past the instant it was issued
    /// to be usable until.
    fn still_valid(self, now: i64) -> Result<Self, IdentityError> {
        match self.expires_at_unix {
            Some(expires_at) if now >= expires_at => Err(IdentityError::ExpiredToken),
            _ => Ok(self),
        }
    }
}

/// Looks a token up by hash, requiring the membership it was scoped to still
/// exist. Says nothing about expiry; see [`Credential::still_valid`].
fn lookup_credential(conn: &Connection, token: &str) -> Result<Credential, IdentityError> {
    conn.query_row(
        "SELECT token.principal_id, token.organization_id, token.expires_at_unix
         FROM tokens AS token
         INNER JOIN organization_memberships AS membership
           ON membership.principal_id = token.principal_id
          AND membership.organization_id = token.organization_id
         WHERE token.token_hash = ?1",
        rusqlite::params![hash_token(token)],
        |row| {
            Ok(Credential {
                principal_id: row.get(0)?,
                organization_id: row.get(1)?,
                expires_at_unix: row.get(2)?,
            })
        },
    )
    .optional()?
    .ok_or(IdentityError::InvalidToken)
}

fn issue_token_in_transaction(
    transaction: &rusqlite::Transaction<'_>,
    principal: &PrincipalId,
    organization: &OrganizationId,
    expires_at_unix: Option<i64>,
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
        "INSERT INTO tokens (token_hash, principal_id, organization_id, created_at_unix, expires_at_unix)
         VALUES (?1, ?2, ?3, ?4, ?5)",
        rusqlite::params![
            hash_token(&token),
            principal.0,
            organization.0,
            unix_now(),
            expires_at_unix
        ],
    )?;
    Ok(token)
}

/// Brings the database up to [`IDENTITY_SCHEMA_VERSION`], or confirms it is
/// already there.
///
/// Every path through here is a read-then-write: decide what the schema is,
/// then change it. That decision is made while *holding* the write lock rather
/// than before asking for it, and the distinction is the whole correctness
/// argument. This crate is explicitly built for a long-lived `serve` process
/// and a short-lived CLI process sharing one database file, and a database in
/// the field is already in WAL mode -- the journal-mode switch in
/// [`SqliteIdentityStore::open`] put it there, so it no longer serializes
/// openers the way it does on a not-yet-WAL file. Two processes therefore
/// reach an unmigrated database at the same instant routinely. Deciding first
/// and locking second lets both act on the same stale answer, and the loser
/// fails to start: a duplicate column on either migration step, or
/// `SQLITE_BUSY` from a deferred transaction's write upgrade -- an upgrade the
/// busy handler deliberately does not retry.
///
/// The whole chain runs in that one transaction, so an upgrade is atomic
/// across every step it traverses. A database is either fully migrated or
/// untouched; it is never stranded between two versions by a failure partway.
fn initialize_schema(conn: &mut Connection) -> Result<(), IdentityError> {
    // Nothing to do is the overwhelmingly common case -- every open after the
    // first -- and it needs no write lock at all. Taking one unconditionally
    // would serialize every process start against every other for no reason.
    if schema_version(conn)? == IDENTITY_SCHEMA_VERSION {
        return validate_v3_schema(conn);
    }

    let transaction = begin_write(conn)?;
    // Re-read under the lock: another process may have migrated between the
    // check above and this `BEGIN`, in which case the version below is already
    // current and its arm does nothing.
    match schema_version(&transaction)? {
        0 => initialize_unversioned_database(&transaction)?,
        1 => {
            migrate_v1_to_v2(&transaction)?;
            migrate_v2_to_v3(&transaction)?;
        }
        2 => migrate_v2_to_v3(&transaction)?,
        3 => validate_v3_schema(&transaction)?,
        found => {
            return Err(IdentityError::UnsupportedSchemaVersion {
                found,
                supported: IDENTITY_SCHEMA_VERSION,
            })
        }
    }
    transaction.commit()?;
    Ok(())
}

fn schema_version(conn: &Connection) -> Result<i64, IdentityError> {
    Ok(conn.pragma_query_value(None, "user_version", |row| row.get(0))?)
}

/// Begins a transaction that takes the write lock at `BEGIN` rather than on
/// its first write.
///
/// Every write path in this store reads before it writes -- resolve a
/// membership, check a principal exists, look a credential up -- and then
/// writes based on what it read. A deferred transaction acquires only a read
/// lock for that first step and tries to upgrade later, and in WAL mode that
/// upgrade fails with `SQLITE_BUSY` the moment another writer has committed in
/// between. The busy handler does not retry a write upgrade, so `busy_timeout`
/// does not save it: the operation simply fails.
///
/// Taking the lock up front turns that failure into a wait. Contention becomes
/// a slower call instead of an error a caller has to know how to retry, and
/// the value each path read is guaranteed still to hold when it writes.
fn begin_write(conn: &mut Connection) -> Result<rusqlite::Transaction<'_>, IdentityError> {
    Ok(conn.transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?)
}

fn initialize_unversioned_database(conn: &Connection) -> Result<(), IdentityError> {
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
        (false, false, false, false) => create_schema_v3(conn),
        (true, true, false, false) => {
            migrate_v1_to_v2(conn)?;
            migrate_v2_to_v3(conn)
        }
        _ => Err(IdentityError::MigrationIntegrity(
            "unversioned database does not match the supported legacy identity schema".into(),
        )),
    }
}

fn create_schema_v3(conn: &Connection) -> Result<(), IdentityError> {
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
            -- NULL means the token never expires. Nullable rather than a
            -- sentinel far-future instant so that having no expiry is a state
            -- the schema can express, not a date that eventually arrives.
            expires_at_unix INTEGER,
            FOREIGN KEY (organization_id, principal_id)
                REFERENCES organization_memberships(organization_id, principal_id)
                ON DELETE CASCADE
        );
        CREATE INDEX IF NOT EXISTS idx_tokens_principal ON tokens(principal_id);
        CREATE INDEX IF NOT EXISTS idx_memberships_principal ON organization_memberships(principal_id);
        PRAGMA user_version = 3;",
    )?;
    Ok(())
}

/// Adds the token expiry column. Every existing row keeps `NULL`, which is
/// "never expires" — the behavior those tokens were issued under. A migration
/// must not silently start expiring credentials that are in use.
///
/// A step inside the caller's write transaction; see [`initialize_schema`].
fn migrate_v2_to_v3(conn: &Connection) -> Result<(), IdentityError> {
    conn.execute_batch(
        "ALTER TABLE tokens ADD COLUMN expires_at_unix INTEGER;
         PRAGMA user_version = 3;",
    )?;
    Ok(())
}

/// Gives every legacy principal a personal organization and rewrites its
/// tokens to be scoped to it.
///
/// A step inside the caller's write transaction; see [`initialize_schema`].
fn migrate_v1_to_v2(transaction: &Connection) -> Result<(), IdentityError> {
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

fn validate_v3_schema(conn: &Connection) -> Result<(), IdentityError> {
    for query in [
        "SELECT principal_id, name, created_at_unix FROM users LIMIT 0",
        "SELECT organization_id, name, created_at_unix FROM organizations LIMIT 0",
        "SELECT organization_id, principal_id, created_at_unix FROM organization_memberships LIMIT 0",
        "SELECT token_hash, principal_id, organization_id, created_at_unix, expires_at_unix FROM tokens LIMIT 0",
    ] {
        conn.prepare(query)?;
    }
    // That the column exists is not the claim this schema makes; that it is
    // nullable is. `NULL` is how a non-expiring token is stored, so a `NOT
    // NULL` version of this column would force every issuance to invent an
    // expiry date -- precisely the design this rejected.
    let expiry_is_nullable: Option<bool> = conn
        .query_row(
            "SELECT \"notnull\" = 0 FROM pragma_table_info('tokens') WHERE name = 'expires_at_unix'",
            [],
            |row| row.get(0),
        )
        .optional()?;
    match expiry_is_nullable {
        Some(true) => Ok(()),
        Some(false) => Err(IdentityError::MigrationIntegrity(
            "tokens.expires_at_unix must be nullable: NULL is how a non-expiring token is stored"
                .into(),
        )),
        None => Err(IdentityError::MigrationIntegrity(
            "tokens table has no expires_at_unix column".into(),
        )),
    }
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

/// The current time on the clock every expiry in this crate is measured
/// against.
///
/// Exposed because [`SqliteIdentityStore::issue_token_expiring_at`] takes an
/// absolute instant: a caller that computes one from some other source is
/// asserting that its clock and this one agree, and that assertion is better
/// made explicit than assumed. [`expires_in`] covers the ordinary case of a
/// lifetime starting now.
///
/// Total, deliberately. A clock set before 1970 yields a negative instant,
/// which still orders correctly against any stored expiry, and one set
/// absurdly far forward saturates. Neither is a reason for a function on an
/// authentication path to abort the process.
pub fn unix_now() -> i64 {
    match std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH) {
        Ok(since_epoch) => i64::try_from(since_epoch.as_secs()).unwrap_or(i64::MAX),
        Err(before_epoch) => i64::try_from(before_epoch.duration().as_secs())
            .map(|seconds| -seconds)
            .unwrap_or(i64::MIN),
    }
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

        let mut expected_organizations = vec![first.clone(), second.clone()];
        expected_organizations.sort_by(|left, right| left.as_str().cmp(right.as_str()));
        assert_eq!(
            store.organizations_for_principal(&principal).unwrap(),
            expected_organizations
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

    /// Reads a token's stored expiry directly, so the tests assert what was
    /// persisted rather than what an accessor chose to report.
    fn stored_expiry(store: &SqliteIdentityStore, token: &str) -> Option<i64> {
        store
            .locked_connection()
            .query_row(
                "SELECT expires_at_unix FROM tokens WHERE token_hash = ?1",
                rusqlite::params![hash_token(token)],
                |row| row.get(0),
            )
            .unwrap()
    }

    fn token_count(store: &SqliteIdentityStore) -> i64 {
        store
            .locked_connection()
            .query_row("SELECT COUNT(*) FROM tokens", [], |row| row.get(0))
            .unwrap()
    }

    #[test]
    fn a_token_issued_with_a_future_expiry_still_resolves() {
        let store = SqliteIdentityStore::open_in_memory().unwrap();
        let principal = store.create_user("ada").unwrap();
        let expires_at = expires_in(NonZeroU64::new(3600).unwrap());
        let token = store
            .issue_token_expiring_at(&principal, Some(expires_at))
            .unwrap();

        assert_eq!(
            store.resolve_context(&token).unwrap().principal_id(),
            &principal
        );
        assert_eq!(stored_expiry(&store, &token), Some(expires_at));
    }

    #[test]
    fn a_token_issued_without_an_expiry_stays_non_expiring() {
        let store = SqliteIdentityStore::open_in_memory().unwrap();
        let principal = store.create_user("ada").unwrap();
        let token = store.issue_token(&principal).unwrap();

        assert_eq!(stored_expiry(&store, &token), None);
        assert_eq!(store.resolve(&token).unwrap(), principal);
    }

    #[test]
    fn an_expired_token_is_refused_by_every_resolution_path() {
        let store = SqliteIdentityStore::open_in_memory().unwrap();
        let principal = store.create_user("ada").unwrap();
        let token = store
            .issue_token_expiring_at(&principal, Some(unix_now() - 1))
            .unwrap();

        assert!(matches!(
            store.resolve_context(&token),
            Err(IdentityError::ExpiredToken)
        ));
        // The principal-only compatibility contract must not be a way around
        // the expiry the organization-aware path enforces.
        assert!(matches!(
            store.resolve(&token),
            Err(IdentityError::ExpiredToken)
        ));
    }

    #[test]
    fn a_token_is_refused_at_the_exact_instant_it_expires() {
        let store = SqliteIdentityStore::open_in_memory().unwrap();
        let principal = store.create_user("ada").unwrap();
        let token = store
            .issue_token_expiring_at(&principal, Some(unix_now()))
            .unwrap();

        assert!(matches!(
            store.resolve_context(&token),
            Err(IdentityError::ExpiredToken)
        ));
    }

    #[test]
    fn expiry_is_refused_without_disclosing_it_to_an_unissued_token() {
        let store = SqliteIdentityStore::open_in_memory().unwrap();
        let principal = store.create_user("ada").unwrap();
        store
            .issue_token_expiring_at(&principal, Some(unix_now() - 1))
            .unwrap();

        // A token that was never issued is still indistinguishable from a
        // revoked one. Expiry is reported only to whoever holds a real token.
        assert!(matches!(
            store.resolve_context("cme_never_issued"),
            Err(IdentityError::InvalidToken)
        ));
    }

    #[test]
    fn rotation_issues_a_working_token_and_retires_the_presented_one() {
        let store = SqliteIdentityStore::open_in_memory().unwrap();
        let organization = store.create_organization("Acme").unwrap();
        let principal = store
            .create_user_in_organization("ada", &organization)
            .unwrap();
        let original = store
            .issue_token_for_organization(&principal, &organization)
            .unwrap();

        let rotated = store.rotate_token(&original, None).unwrap();

        assert_ne!(rotated, original);
        let context = store.resolve_context(&rotated).unwrap();
        assert_eq!(context.principal_id(), &principal);
        assert_eq!(context.organization_id(), &organization);
        assert!(matches!(
            store.resolve_context(&original),
            Err(IdentityError::InvalidToken)
        ));
        // Exactly one live credential: rotation is a replacement, not an
        // additional issuance that happens to invalidate something.
        assert_eq!(token_count(&store), 1);
    }

    #[test]
    fn rotation_sets_the_replacement_lifetime_independently_of_the_old_one() {
        let store = SqliteIdentityStore::open_in_memory().unwrap();
        let principal = store.create_user("ada").unwrap();
        let expires_at = expires_in(NonZeroU64::new(600).unwrap());

        let non_expiring = store.issue_token(&principal).unwrap();
        let now_expiring = store.rotate_token(&non_expiring, Some(expires_at)).unwrap();
        assert_eq!(stored_expiry(&store, &now_expiring), Some(expires_at));

        let back_to_permanent = store.rotate_token(&now_expiring, None).unwrap();
        assert_eq!(stored_expiry(&store, &back_to_permanent), None);
    }

    #[test]
    fn an_expired_token_cannot_buy_a_live_one() {
        let store = SqliteIdentityStore::open_in_memory().unwrap();
        let principal = store.create_user("ada").unwrap();
        // Captured once: reading the clock again below could land in the next
        // second and compare against an instant that was never stored.
        let expired_at = unix_now() - 1;
        let expired = store
            .issue_token_expiring_at(&principal, Some(expired_at))
            .unwrap();

        assert!(matches!(
            store.rotate_token(&expired, None),
            Err(IdentityError::ExpiredToken)
        ));
        // The refusal issued nothing and consumed nothing.
        assert_eq!(token_count(&store), 1);
        assert_eq!(stored_expiry(&store, &expired), Some(expired_at));
    }

    #[test]
    fn rotating_an_unknown_token_issues_nothing() {
        let store = SqliteIdentityStore::open_in_memory().unwrap();
        let principal = store.create_user("ada").unwrap();
        let live = store.issue_token(&principal).unwrap();

        assert!(matches!(
            store.rotate_token("cme_never_issued", None),
            Err(IdentityError::InvalidToken)
        ));
        assert_eq!(token_count(&store), 1);
        assert_eq!(
            store.resolve_context(&live).unwrap().principal_id(),
            &principal
        );
    }

    #[test]
    fn rotation_keeps_a_multi_organization_principal_in_the_same_organization() {
        let store = SqliteIdentityStore::open_in_memory().unwrap();
        let first = store.create_organization("Acme").unwrap();
        let second = store.create_organization("Example").unwrap();
        let principal = store.create_user_in_organization("ada", &first).unwrap();
        store
            .add_principal_to_organization(&principal, &second)
            .unwrap();
        let second_token = store
            .issue_token_for_organization(&principal, &second)
            .unwrap();

        let rotated = store.rotate_token(&second_token, None).unwrap();

        // `issue_token` would have refused this principal as ambiguous, so
        // rotation must carry the scope rather than re-derive it.
        assert_eq!(
            store.resolve_context(&rotated).unwrap().organization_id(),
            &second
        );
    }

    /// Writes a schema-v2 identity database holding one user, one
    /// organization, its membership, and one token.
    ///
    /// Left in WAL mode, because that is the state a real one is in: the
    /// shipped `open` switches every database it touches to WAL, so any v2
    /// file an upgrade encounters has already been through it. Building the
    /// fixture in rollback-journal mode would make `open`'s journal-mode
    /// switch take an exclusive lock and quietly serialize concurrent openers,
    /// hiding the very contention these tests exist to exercise.
    fn write_v2_database(
        path: &std::path::Path,
        principal: &PrincipalId,
        organization: &OrganizationId,
        token: &str,
    ) {
        let connection = Connection::open(path).unwrap();
        connection
            .pragma_update(None, "journal_mode", "WAL")
            .unwrap();
        connection
            .execute_batch(
                "PRAGMA foreign_keys = ON;
                 CREATE TABLE users (
                    principal_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    created_at_unix INTEGER NOT NULL
                 );
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
                 );
                 CREATE TABLE tokens (
                    token_hash TEXT PRIMARY KEY,
                    principal_id TEXT NOT NULL,
                    organization_id TEXT NOT NULL,
                    created_at_unix INTEGER NOT NULL,
                    FOREIGN KEY (organization_id, principal_id)
                        REFERENCES organization_memberships(organization_id, principal_id)
                        ON DELETE CASCADE
                 );",
            )
            .unwrap();
        connection
            .execute(
                "INSERT INTO users (principal_id, name, created_at_unix) VALUES (?1, 'ada', 1)",
                rusqlite::params![principal.as_str()],
            )
            .unwrap();
        connection
            .execute(
                "INSERT INTO organizations (organization_id, name, created_at_unix) VALUES (?1, 'Acme', 1)",
                rusqlite::params![organization.as_str()],
            )
            .unwrap();
        connection
            .execute(
                "INSERT INTO organization_memberships (organization_id, principal_id, created_at_unix) VALUES (?1, ?2, 1)",
                rusqlite::params![organization.as_str(), principal.as_str()],
            )
            .unwrap();
        connection
            .execute(
                "INSERT INTO tokens (token_hash, principal_id, organization_id, created_at_unix) VALUES (?1, ?2, ?3, 1)",
                rusqlite::params![hash_token(token), principal.as_str(), organization.as_str()],
            )
            .unwrap();
        connection
            .pragma_update(None, "user_version", 2_i64)
            .unwrap();
    }

    #[test]
    fn migrating_a_v2_database_leaves_its_tokens_non_expiring() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("identity.sqlite");
        let principal = PrincipalId::new("usr_v2_ada".to_string());
        let organization = OrganizationId::new("org_v2_acme".to_string());
        let token = "cme_v2_ada";
        write_v2_database(&path, &principal, &organization, token);

        let store = SqliteIdentityStore::open(&path).unwrap();

        // A migration must not start expiring credentials that are in use.
        assert_eq!(stored_expiry(&store, token), None);
        let context = store.resolve_context(token).unwrap();
        assert_eq!(context.principal_id(), &principal);
        assert_eq!(context.organization_id(), &organization);
        assert_eq!(
            store
                .locked_connection()
                .pragma_query_value(None, "user_version", |row| row.get::<_, i64>(0))
                .unwrap(),
            IDENTITY_SCHEMA_VERSION
        );
    }

    /// Writes an unversioned (`user_version` 0) legacy identity database:
    /// users and principal-scoped tokens, no organizations.
    ///
    /// WAL for the same reason [`write_v2_database`] is.
    fn write_legacy_database(path: &std::path::Path, principal: &PrincipalId, token: &str) {
        let connection = Connection::open(path).unwrap();
        connection
            .pragma_update(None, "journal_mode", "WAL")
            .unwrap();
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
                "INSERT INTO users (principal_id, name, created_at_unix) VALUES (?1, 'ada', 1)",
                rusqlite::params![principal.as_str()],
            )
            .unwrap();
        connection
            .execute(
                "INSERT INTO tokens (token_hash, principal_id, created_at_unix) VALUES (?1, ?2, 1)",
                rusqlite::params![hash_token(token), principal.as_str()],
            )
            .unwrap();
    }

    /// Opens `path` from 16 threads released together and returns every
    /// outcome. Schema initialization is the one moment all processes contend
    /// for the same write, so it is the one worth driving from more than one.
    fn open_concurrently(
        path: &std::path::Path,
    ) -> Vec<Result<SqliteIdentityStore, IdentityError>> {
        const OPENERS: usize = 16;
        let barrier = std::sync::Arc::new(std::sync::Barrier::new(OPENERS));
        let mut threads = Vec::new();
        for _ in 0..OPENERS {
            let path = path.to_path_buf();
            let barrier = std::sync::Arc::clone(&barrier);
            threads.push(std::thread::spawn(move || {
                barrier.wait();
                SqliteIdentityStore::open(&path)
            }));
        }
        threads
            .into_iter()
            .map(|thread| thread.join().unwrap())
            .collect()
    }

    /// Requires every opener to have succeeded and to see a fully migrated
    /// database, reporting the first failure's error rather than a bare panic.
    ///
    /// All of them, not merely one: schema initialization happens during
    /// startup, so an opener turned away here is a process that failed to
    /// boot. A weaker assertion would pass while most of a deployment is
    /// crash-looping.
    fn assert_all_opened(
        outcomes: Vec<Result<SqliteIdentityStore, IdentityError>>,
    ) -> Vec<SqliteIdentityStore> {
        let total = outcomes.len();
        let mut stores = Vec::with_capacity(total);
        let mut failures = Vec::new();
        for outcome in outcomes {
            match outcome {
                Ok(store) => stores.push(store),
                Err(error) => failures.push(error.to_string()),
            }
        }
        assert!(
            failures.is_empty(),
            "{} of {total} openers failed; first: {}",
            failures.len(),
            failures[0]
        );
        for store in &stores {
            assert_eq!(
                store
                    .locked_connection()
                    .pragma_query_value(None, "user_version", |row| row.get::<_, i64>(0))
                    .unwrap(),
                IDENTITY_SCHEMA_VERSION
            );
        }
        stores
    }

    #[test]
    fn concurrent_openers_all_migrate_a_v2_database() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("identity.sqlite");
        let principal = PrincipalId::new("usr_v2_ada".to_string());
        let organization = OrganizationId::new("org_v2_acme".to_string());
        let token = "cme_v2_ada";
        write_v2_database(&path, &principal, &organization, token);

        for store in assert_all_opened(open_concurrently(&path)) {
            assert_eq!(stored_expiry(&store, token), None);
            assert_eq!(
                store.resolve_context(token).unwrap().organization_id(),
                &organization
            );
        }
    }

    #[test]
    fn concurrent_openers_all_migrate_a_legacy_database() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("identity.sqlite");
        let principal = PrincipalId::new("usr_legacy_ada".to_string());
        let token = "cme_legacy_ada";
        write_legacy_database(&path, &principal, token);

        // The same contention over the longer chain: an unversioned database
        // runs v1 -> v2 -> v3 in one go. Covering only the v2 -> v3 step would
        // leave the reachable legacy path untested, and it is the path doing
        // the most work while holding the write lock.
        for store in assert_all_opened(open_concurrently(&path)) {
            assert_eq!(stored_expiry(&store, token), None);
            assert_eq!(
                store.resolve_context(token).unwrap().principal_id(),
                &principal
            );
        }
    }

    #[test]
    fn concurrent_openers_all_share_an_established_database() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("identity.sqlite");
        // Establish the file first, which is what an operator does by creating
        // the first user. Concurrent creation of a database that does not yet
        // exist is a separate, pre-existing contention on the journal-mode
        // switch in `open` and is documented rather than asserted here.
        let principal = {
            let store = SqliteIdentityStore::open(&path).unwrap();
            store.create_user("ada").unwrap()
        };
        let token = {
            let store = SqliteIdentityStore::open(&path).unwrap();
            store.issue_token(&principal).unwrap()
        };

        let stores = assert_all_opened(open_concurrently(&path));
        for store in &stores {
            assert_eq!(store.resolve(&token).unwrap(), principal);
        }
    }

    #[test]
    fn concurrent_rotations_of_distinct_tokens_all_succeed() {
        const ROTATORS: usize = 16;
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("identity.sqlite");
        let store = std::sync::Arc::new(SqliteIdentityStore::open(&path).unwrap());
        let organization = store.create_organization("Acme").unwrap();
        let principal = store
            .create_user_in_organization("ada", &organization)
            .unwrap();
        let tokens: Vec<String> = (0..ROTATORS)
            .map(|_| {
                store
                    .issue_token_for_organization(&principal, &organization)
                    .unwrap()
            })
            .collect();

        // Sixteen holders refreshing their own credentials at once. Nothing
        // here conflicts semantically -- every rotation touches a different
        // row -- so every one of them must succeed. Contention between
        // unrelated writers is the store's problem to absorb, not something a
        // caller should have to retry.
        let barrier = std::sync::Arc::new(std::sync::Barrier::new(ROTATORS));
        let mut threads = Vec::new();
        for token in tokens {
            let store = std::sync::Arc::clone(&store);
            let barrier = std::sync::Arc::clone(&barrier);
            threads.push(std::thread::spawn(move || {
                barrier.wait();
                store.rotate_token(&token, None)
            }));
        }
        let outcomes: Vec<_> = threads
            .into_iter()
            .map(|thread| thread.join().unwrap())
            .collect();

        let failures: Vec<String> = outcomes
            .iter()
            .filter_map(|outcome| outcome.as_ref().err().map(|error| error.to_string()))
            .collect();
        assert!(
            failures.is_empty(),
            "{} of {ROTATORS} rotations failed; first: {}",
            failures.len(),
            failures.first().map_or("", String::as_str)
        );
        for replacement in outcomes.into_iter().flatten() {
            assert_eq!(
                store.resolve_context(&replacement).unwrap().principal_id(),
                &principal
            );
        }
        assert_eq!(token_count(&store), ROTATORS as i64);
    }

    #[test]
    fn concurrent_rotations_of_one_token_leave_a_single_winner() {
        const ROTATORS: usize = 16;
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("identity.sqlite");
        let store = std::sync::Arc::new(SqliteIdentityStore::open(&path).unwrap());
        let principal = store.create_user("ada").unwrap();
        let token = store.issue_token(&principal).unwrap();

        // The same credential presented by sixteen racers. Exactly one may
        // win: rotation consumes what it replaces, so a second success would
        // mean the retired token bought a live one twice.
        let barrier = std::sync::Arc::new(std::sync::Barrier::new(ROTATORS));
        let mut threads = Vec::new();
        for _ in 0..ROTATORS {
            let store = std::sync::Arc::clone(&store);
            let barrier = std::sync::Arc::clone(&barrier);
            let token = token.clone();
            threads.push(std::thread::spawn(move || {
                barrier.wait();
                store.rotate_token(&token, None)
            }));
        }
        let outcomes: Vec<_> = threads
            .into_iter()
            .map(|thread| thread.join().unwrap())
            .collect();

        let winners: Vec<&String> = outcomes.iter().filter_map(|o| o.as_ref().ok()).collect();
        assert_eq!(winners.len(), 1, "exactly one rotation may consume a token");
        for outcome in &outcomes {
            if let Err(error) = outcome {
                // The losers must be told the token is gone, not that the
                // database was busy. A retryable-looking error here would
                // invite a client to replay a credential that no longer exists.
                assert!(
                    matches!(error, IdentityError::InvalidToken),
                    "loser saw {error} instead of a retired credential"
                );
            }
        }
        assert_eq!(store.resolve(winners[0]).unwrap(), principal);
        assert_eq!(token_count(&store), 1);
    }

    #[test]
    fn expires_in_is_a_lifetime_measured_from_now() {
        let before = unix_now();
        let expires_at = expires_in(NonZeroU64::new(60).unwrap());
        let after = unix_now();

        assert!((before + 60..=after + 60).contains(&expires_at));
    }
}
