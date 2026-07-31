import { memo, useEffect, useState } from 'react'
import { clampText } from '../lib/formatters'
import { getServiceReadiness } from '../lib/serviceReadiness'
import { IconMenu } from './ui/icons'
import { StatusDot } from './ui/StatusDot'
import { CamelidMark } from './ui/CamelidMark'

const TITLES = {
  chat: 'Chat',
  history: 'Chat history',
  settings: 'Settings',
}

/* Slim top bar. Chat tab shows the conversation title and service readiness;
   the right side shows who the gateway resolved this session to.

   `identity` is passed in rather than read from the store here, so there is
   exactly one place (the app root) that decides what a signed-in session is. */
function TopBar({
  tab,
  setTab,
  selectedConversationTitle,
  runtime,
  selectedModelId,
  models = [],
  identity = null,
  authMode = null,
  onSignOut = null,
  onToggleSidebar = null,
}) {

  const rawTitle = selectedConversationTitle?.trim()
  const hasCustomTitle = Boolean(rawTitle && rawTitle.toLowerCase() !== 'new conversation')
  const heading = tab === 'chat'
    ? (hasCustomTitle ? clampText(rawTitle, 64) : 'New chat')
    : (TITLES[tab] || 'Camelid Enterprise')

  const selectedModel = models.find((m) => m.id === selectedModelId)
    || models.find((m) => m.id === runtime?.active_model_id)
  const readiness = getServiceReadiness(runtime, selectedModelId)
  const apiUnavailable = runtime?.status === 'offline'
  const tone = readiness.canSend ? 'ready' : apiUnavailable ? 'offline' : runtime?.loaded_now ? 'warn' : 'neutral'
  const modelName = selectedModel?.name || runtime?.active_model_id || 'No model selected'
  /* With authentication disabled there is no principal to show, and inventing a
     placeholder name would misrepresent an unauthenticated deployment as an
     identified session. */
  const sessionLabel = identity?.principalId || (authMode === 'disabled' ? 'No authentication' : 'Signed out')

  return (
    <header className="topbar">
      {onToggleSidebar && (
        <button type="button" className="topbar__menu" aria-label="Toggle sidebar" onClick={onToggleSidebar}>
          <IconMenu size={22} />
        </button>
      )}
      <CamelidMark size={18} className="topbar__mark" />
      <h1 className="topbar__title" title={tab === 'chat' && hasCustomTitle ? rawTitle : heading}>{heading}</h1>
      <div className="topbar__spacer" />
      {(
        <div className="topbar__gate">
          <button
            type="button"
            className="button button--ghost topbar__session"
            onClick={onSignOut}
            title={
              identity
                ? `${identity.principalId}${identity.organizationId ? ` · ${identity.organizationId}` : ''} — sign out`
                : 'Sign out'
            }
          >
            <StatusDot tone={identity ? 'ready' : 'neutral'} />
            <span className="topbar__session-name">{sessionLabel}</span>
          </button>
          {/* Not a control. A replica serves one model for its process
              lifetime, so this reports what is being served rather than
              offering to change it. */}
          <span className="topbar__model" title={readiness.copy}>
            <StatusDot tone={tone} pulse={readiness.canSend} />
            <span className="topbar__model-name">{clampText(modelName, 32)}</span>
          </span>
        </div>
      )}
    </header>
  )
}

export default memo(TopBar)

