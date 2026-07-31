import { useState } from 'react'
import { Card, CardHeader, CardBody } from '../components/ui/Card'
import { Button } from '../components/ui/Button'
import { StatusDot } from '../components/ui/StatusDot'
import { Field } from '../components/ui/Field'
import { IconCopy, IconCheck, IconServer, IconMonitor, IconSun, IconMoon } from '../components/ui/icons'
import { copyText } from '../lib/markdown'
import { getConfiguredMaxTokens, setConfiguredMaxTokens } from '../lib/responseLimits'
import { ResponseLengthControl } from '../components/settings/ResponseLengthControl'
import { getGatewayUrl, setGatewayUrl } from '../lib/authStore'

/* Settings for a console that talks to a service it does not operate.
 *
 * What this replaces was the settings page of a desktop application: it started
 * and stopped a local backend process, tailed its log, and flipped runtime GPU
 * flags over a control-plane route. None of that is reachable here, and none of
 * it should be — the replica's lifecycle belongs to whoever deployed it, and the
 * routes that would change it are withheld precisely so a client cannot move the
 * ground under the identity digests the replica publishes on every response.
 *
 * So this page settles what genuinely is the client's: which gateway to talk to,
 * who we are talking to it as, how long replies may run, and the theme.
 */
export default function SettingsView({
  runtime,
  identity = null,
  authMode = null,
  onSignOut = null,
  selectedModelId = '',
  themePreference = 'system',
  onSelectTheme = null,
  showNotice = null,
}) {
  const [endpointDraft, setEndpointDraft] = useState(() => getGatewayUrl())
  const [copied, setCopied] = useState(false)
  const [maxTokens, setMaxTokens] = useState(() => getConfiguredMaxTokens(selectedModelId))

  const online = runtime?.status === 'online'

  const applyEndpoint = () => {
    const next = endpointDraft.trim()
    if (!next) return
    setGatewayUrl(next)
    /* A different endpoint is a different identity store, so the credential we
       hold may mean nothing there. Signing out sends the next request through a
       real verification instead of letting a stale session describe a service it
       was never issued for. */
    showNotice?.('Endpoint saved. Sign in again to verify against it.', 'info')
    onSignOut?.()
  }

  const handleMaxTokens = (value) => {
    setMaxTokens(value)
    setConfiguredMaxTokens(selectedModelId || '', value)
  }

  return (
    <div className="settings-view">
      <header className="settings-view__head">
        <h2>Settings</h2>
        <p>Choose which gateway this console talks to, and how replies behave.</p>
      </header>

      <Card>
        <CardHeader eyebrow="Session" title="Signed in as" />
        <CardBody>
          {identity ? (
            <dl className="settings-identity">
              <div>
                <dt>Principal</dt>
                <dd className="settings-identity__mono">{identity.principalId}</dd>
              </div>
              {identity.organizationId && (
                <div>
                  <dt>Organization</dt>
                  <dd className="settings-identity__mono">{identity.organizationId}</dd>
                </div>
              )}
            </dl>
          ) : (
            <p className="settings-note">
              {authMode === 'disabled'
                ? 'This gateway runs with authentication disabled, so requests carry no identity and nothing here is attributed to a principal.'
                : 'No verified session.'}
            </p>
          )}
          <p className="settings-note">
            These values come from the gateway resolving your token — they are not stored in this
            browser and cannot be edited here.
          </p>
          {onSignOut && (
            <Button variant="ghost" onClick={onSignOut}>
              Sign out
            </Button>
          )}
        </CardBody>
      </Card>

      <Card>
        <CardHeader eyebrow="Connection" title="Gateway endpoint" />
        <CardBody>
          <div className="settings-status-row">
            <StatusDot tone={online ? 'ready' : 'offline'} />
            <span>{online ? 'Connected' : 'Not responding'}</span>
            {runtime?.active_model_id && (
              <span className="settings-identity__mono">serving {runtime.active_model_id}</span>
            )}
          </div>
          <Field label="Endpoint URL" hint="The gateway this console sends every request to.">
            <div className="settings-endpoint">
              <input
                className="text-input"
                type="url"
                value={endpointDraft}
                onChange={(event) => setEndpointDraft(event.target.value)}
                placeholder="http://127.0.0.1:8080"
              />
              <Button onClick={applyEndpoint} icon={<IconServer size={16} />}>
                Save
              </Button>
              <Button
                variant="ghost"
                onClick={async () => {
                  await copyText(endpointDraft)
                  setCopied(true)
                  setTimeout(() => setCopied(false), 1200)
                }}
                icon={copied ? <IconCheck size={16} /> : <IconCopy size={16} />}
              >
                {copied ? 'Copied' : 'Copy'}
              </Button>
            </div>
          </Field>
        </CardBody>
      </Card>

      <Card>
        <CardHeader eyebrow="Generation" title="Response length" />
        <CardBody>
          <ResponseLengthControl value={maxTokens} onChange={handleMaxTokens} />
          <p className="settings-note">
            An upper bound on the reply. The service clamps it to whatever room the context has
            left, so a high value here is a ceiling rather than a request.
          </p>
        </CardBody>
      </Card>

      <Card>
        <CardHeader eyebrow="Appearance" title="Theme" />
        <CardBody>
          <div className="settings-theme">
            {[
              { value: 'system', label: 'System', Icon: IconMonitor },
              { value: 'light', label: 'Light', Icon: IconSun },
              { value: 'dark', label: 'Dark', Icon: IconMoon },
            ].map(({ value, label, Icon }) => (
              <button
                key={value}
                type="button"
                className={`settings-theme__opt ${themePreference === value ? 'is-active' : ''}`}
                aria-pressed={themePreference === value}
                onClick={() => onSelectTheme?.(value)}
              >
                <Icon size={18} />
                <span>{label}</span>
              </button>
            ))}
          </div>
        </CardBody>
      </Card>
    </div>
  )
}
