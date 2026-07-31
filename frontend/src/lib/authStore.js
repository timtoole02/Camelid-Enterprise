/* Session state for the console.
 *
 * One rule shapes this whole module: **the token is the only thing we persist,
 * and identity is always derived from it by the server.**
 *
 * The obvious alternative — storing the principal id alongside the token — is
 * what the previous revision did, and it produced two identities per session
 * that could disagree. Only one of them was real: the gateway resolves the token
 * on every single request and that resolution is what decides authorization, the
 * quota key, and every audit line. A principal id in localStorage is a cached
 * guess about that resolution, and it goes stale in exactly the cases that
 * matter most — the token was revoked, rotated, or moved to another
 * organization. So identity lives in memory, for the lifetime of a verified
 * session, and comes back from `GET /auth/whoami`.
 *
 * A consequence worth stating: a stored token is *not* a session. On boot we
 * hold a credential, not an identity, and we are signed out until the gateway
 * says otherwise. That is why `restoreSession` exists and why it can fail.
 */

const TOKEN_KEY = 'camelid.enterprise.token'
const GATEWAY_URL_KEY = 'camelid.enterprise.gatewayUrl'

const DEFAULT_GATEWAY_URL = 'http://127.0.0.1:8080'

/* Sign-in states. `verifying` is a real state rather than a boolean flag
   because the gateway round-trip is where a bad token is actually caught, and
   the UI has to be able to say "checking" rather than optimistically showing a
   signed-in shell it may have to snatch back. */
export const AuthStatus = {
  SignedOut: 'signed-out',
  Verifying: 'verifying',
  SignedIn: 'signed-in',
}

let state = {
  status: AuthStatus.SignedOut,
  /* Server-resolved. `null` whenever we are not signed in, and also when the
     gateway runs with authentication disabled — in that mode there is no
     principal, and inventing a placeholder one would be the same lie this
     module exists to remove. */
  identity: null,
  /* 'required' | 'disabled' | null — what the gateway told us about itself.
     Never assumed: a console that guesses this guesses about a deployment's
     security posture. */
  authMode: null,
  error: null,
}

const listeners = new Set()

function setState(next) {
  state = { ...state, ...next }
  for (const listener of listeners) {
    try {
      listener(state)
    } catch (error) {
      console.error('auth listener failed', error)
    }
  }
}

export function getAuthState() {
  return state
}

export function subscribeAuth(listener) {
  listeners.add(listener)
  return () => listeners.delete(listener)
}

export function getGatewayUrl() {
  if (typeof window === 'undefined') return DEFAULT_GATEWAY_URL
  return window.localStorage.getItem(GATEWAY_URL_KEY) || DEFAULT_GATEWAY_URL
}

export function setGatewayUrl(url) {
  if (typeof window === 'undefined') return
  window.localStorage.setItem(GATEWAY_URL_KEY, normalizeGatewayUrl(url))
}

function normalizeGatewayUrl(url) {
  return (url || DEFAULT_GATEWAY_URL).trim().replace(/\/+$/, '')
}

export function getToken() {
  if (typeof window === 'undefined') return null
  return window.localStorage.getItem(TOKEN_KEY) || null
}

export function isAuthenticated() {
  return state.status === AuthStatus.SignedIn
}

export function getAuthHeaders() {
  const token = getToken()
  return token ? { Authorization: `Bearer ${token}` } : {}
}

export class AuthError extends Error {
  constructor(reason, message, options) {
    super(message, options)
    this.name = 'AuthError'
    /* 'unreachable' | 'rejected' | 'expired' | 'gateway-error' | 'unsupported-gateway' */
    this.reason = reason
  }
}

async function readJson(response) {
  try {
    return await response.json()
  } catch {
    return null
  }
}

/* The one call that turns a credential into a session.
 *
 * Everything the UI displays about "who you are" originates here, in the
 * gateway's answer — never in anything the operator typed beside the token. */
async function resolveIdentity(gatewayUrl, token) {
  const headers = { Accept: 'application/json' }
  if (token) headers.Authorization = `Bearer ${token}`

  let response
  try {
    response = await fetch(`${normalizeGatewayUrl(gatewayUrl)}/auth/whoami`, { headers })
  } catch (cause) {
    /* Distinguished from a refusal on purpose. "I could not reach the gateway"
       and "the gateway rejected this credential" call for different actions
       from whoever is looking at the screen, and collapsing them into one
       "sign-in failed" sends people to re-check a token that was fine. */
    throw new AuthError('unreachable', `Could not reach the gateway at ${gatewayUrl}.`, { cause })
  }

  if (response.status === 401) {
    const body = await readJson(response)
    /* `token_expired` is the case where re-presenting this credential can never
       succeed, so it is worth its own message: the action is to obtain a new
       token from an operator, not to retry. */
    if (body?.error?.type === 'token_expired') {
      throw new AuthError('expired', 'That token has expired. Ask an operator to issue a new one.')
    }
    throw new AuthError(
      'rejected',
      token ? 'The gateway did not recognize that token.' : 'This gateway requires a token.',
    )
  }

  if (!response.ok) {
    throw new AuthError('gateway-error', `The gateway returned ${response.status}.`)
  }

  const body = await readJson(response)

  /* An older gateway, or something that is not our gateway, can answer 200 on
     this path without being able to speak to identity at all. Treating a shape
     we do not recognize as a successful sign-in would hand out a session backed
     by nothing. */
  if (body?.authentication === 'disabled') {
    return { authMode: 'disabled', identity: null }
  }
  if (body?.authentication === 'required' && body.principal_id) {
    return {
      authMode: 'required',
      identity: {
        principalId: body.principal_id,
        organizationId: body.organization_id ?? null,
      },
    }
  }

  throw new AuthError(
    'unsupported-gateway',
    'That endpoint did not answer as a Camelid Enterprise gateway.',
  )
}

/* Sign in with a token an operator minted (`camelid-enterprise-gateway
   create-user`). The token is verified before any session exists — there is no
   local-only path to a signed-in state. */
export async function signIn({ gatewayUrl, token }) {
  const url = normalizeGatewayUrl(gatewayUrl)
  setState({ status: AuthStatus.Verifying, error: null })

  try {
    const { authMode, identity } = await resolveIdentity(url, token)

    if (typeof window !== 'undefined') {
      window.localStorage.setItem(GATEWAY_URL_KEY, url)
      /* An auth-disabled gateway needs no credential, and storing one the
         caller typed anyway would leave a secret at rest that nothing uses. */
      if (authMode === 'required' && token) {
        window.localStorage.setItem(TOKEN_KEY, token)
      } else {
        window.localStorage.removeItem(TOKEN_KEY)
      }
    }

    setState({ status: AuthStatus.SignedIn, identity, authMode, error: null })
    return state
  } catch (error) {
    setState({ status: AuthStatus.SignedOut, identity: null, authMode: null, error })
    throw error
  }
}

/* Re-establish a session on boot from the stored token.
 *
 * This deliberately re-verifies rather than trusting what is in localStorage. A
 * token revoked or expired since the last visit must not present a signed-in
 * console that only falls apart on the first real request — and the identity we
 * are about to render has to be this gateway's answer, not the last one's. */
export async function restoreSession() {
  const token = getToken()
  const gatewayUrl = getGatewayUrl()

  /* We ask even with no stored credential, because the gateway may have
     authentication disabled — in which case a sign-in prompt would be asking
     for something nobody can supply. */
  setState({ status: AuthStatus.Verifying, error: null })
  try {
    const { authMode, identity } = await resolveIdentity(gatewayUrl, token)
    setState({ status: AuthStatus.SignedIn, identity, authMode, error: null })
    return state
  } catch (error) {
    if (token && error.reason !== 'unreachable') {
      /* The stored credential is known-bad. Drop it rather than leaving a dead
         secret to fail every request from here on. An unreachable gateway is
         not evidence about the token, so that case keeps it. */
      clearSession()
    }
    setState({ status: AuthStatus.SignedOut, identity: null, authMode: null, error })
    return state
  }
}

export function clearSession() {
  if (typeof window !== 'undefined') {
    window.localStorage.removeItem(TOKEN_KEY)
  }
}

export function signOut() {
  clearSession()
  setState({ status: AuthStatus.SignedOut, identity: null, authMode: null, error: null })
}

/* A 401 from any request means the credential stopped resolving mid-session —
   revoked, expired, or the gateway was repointed. The session is over at that
   moment; letting the shell stay up would show a console whose every action now
   fails. `enterpriseApi` raises this event from one place so no call site has to
   remember to handle it. */
if (typeof window !== 'undefined') {
  window.addEventListener('camelid:auth-error', () => {
    if (state.status === AuthStatus.SignedIn) signOut()
  })
}
