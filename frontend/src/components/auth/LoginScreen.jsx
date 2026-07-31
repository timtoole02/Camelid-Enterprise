import { useState } from 'react'
import { AuthStatus, getGatewayUrl, getAuthState, signIn } from '../../lib/authStore'
import { CamelidMark } from '../ui/CamelidMark'

/* Sign-in.
 *
 * Two fields, and only two: where the gateway is, and the credential. There is
 * deliberately no name or principal-id input — an earlier revision had both, and
 * whatever you typed became what the console displayed as your identity, while
 * the token beside it silently decided every actual authorization. Identity is
 * the gateway's answer to `/auth/whoami`, so there is nothing here to type it
 * into.
 *
 * There is also no dev-bypass button. It set a fixed fake token and walked
 * straight into the shell, which meant the signed-in state could be reached
 * without a gateway ever agreeing to it. Every path to a session now goes
 * through a verified round-trip.
 */
export function LoginScreen({ onAuthenticated }) {
  const [gatewayUrl, setGatewayUrl] = useState(() => getGatewayUrl())
  const [token, setToken] = useState('')
  const [error, setError] = useState(() => getAuthState().error?.message || '')
  const [status, setStatus] = useState(AuthStatus.SignedOut)

  const verifying = status === AuthStatus.Verifying

  const handleSubmit = async (event) => {
    event.preventDefault()
    setError('')
    setStatus(AuthStatus.Verifying)
    try {
      await signIn({ gatewayUrl, token: token.trim() })
      onAuthenticated?.()
    } catch (failure) {
      /* The store already classified this; showing its message keeps
         "unreachable" from reading as "your token is wrong". */
      setError(failure.message)
      setStatus(AuthStatus.SignedOut)
    }
  }

  return (
    <div className="enterprise-login-screen">
      <div className="enterprise-login-screen__backdrop" />
      <div className="enterprise-login-screen__container">
        <div className="enterprise-login-card">
          <div className="enterprise-login-card__header">
            <div className="enterprise-login-card__logo">
              <CamelidMark size={36} />
            </div>
            <h1 className="enterprise-login-card__title">Camelid Enterprise</h1>
            <p className="enterprise-login-card__subtitle">Sign in to the serving console</p>
          </div>

          {error && (
            <div className="enterprise-login-card__error" role="alert">
              {error}
            </div>
          )}

          <form onSubmit={handleSubmit} className="enterprise-login-card__form">
            <div className="form-group">
              <label className="form-label" htmlFor="gateway-url">
                Gateway endpoint
              </label>
              <input
                id="gateway-url"
                type="url"
                className="text-input"
                value={gatewayUrl}
                onChange={(event) => setGatewayUrl(event.target.value)}
                placeholder="http://127.0.0.1:8080"
                disabled={verifying}
                required
              />
            </div>

            <div className="form-group">
              <label className="form-label" htmlFor="gateway-token">
                Access token
              </label>
              <input
                id="gateway-token"
                type="password"
                className="text-input"
                value={token}
                onChange={(event) => setToken(event.target.value)}
                placeholder="Paste your token"
                autoComplete="off"
                disabled={verifying}
                autoFocus
              />
              <span className="form-hint">
                Issued for you by an operator with{' '}
                <code>camelid-enterprise-gateway create-user</code>. Leave empty if this
                deployment runs with authentication disabled.
              </span>
            </div>

            <div className="enterprise-login-card__actions">
              <button
                type="submit"
                className="button button--primary button--large"
                disabled={verifying}
              >
                {verifying ? 'Verifying…' : 'Sign in'}
              </button>
            </div>
          </form>
        </div>
      </div>
    </div>
  )
}

export default LoginScreen
