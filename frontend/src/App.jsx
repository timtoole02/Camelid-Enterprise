import { lazy, Suspense, useEffect, useMemo, useState } from 'react'
import SidebarRail from './components/layout/SidebarRail'
import TopBar from './components/TopBar'
import { Notice } from './components/ui/Notice'
import { ConfirmDialog } from './components/ui/ConfirmDialog'
import { formatPreview } from './lib/formatters'
import { useConsole } from './hooks/useConsole'
import { useNotice } from './hooks/useNotice'
import { useTheme } from './hooks/useTheme'
import ChatWorkspace from './views/ChatWorkspace'
import { CommandPalette } from './components/CommandPalette'
import { ShortcutsOverlay } from './components/ShortcutsOverlay'
import { LoginScreen } from './components/auth/LoginScreen'
import { AuthStatus, getAuthState, restoreSession, signOut, subscribeAuth } from './lib/authStore'

const HistoryView = lazy(() => import('./views/HistoryView'))
const SettingsView = lazy(() => import('./views/SettingsView'))

const HASH_TABS = new Set(['chat', 'history', 'settings'])

export default function App() {
  const { notice, noticeTone, showNotice, clearNotice } = useNotice()
  const { preference, resolved, cyclePreference, setPreference } = useTheme()

  const [auth, setAuth] = useState(getAuthState)

  /* One verification on boot. A stored token is a credential, not a session:
     it may have been revoked or expired since the last visit, and the identity
     we are about to render has to be this gateway's answer rather than a
     remembered one. */
  useEffect(() => {
    const unsubscribe = subscribeAuth(setAuth)
    restoreSession()
    return unsubscribe
  }, [])

  const [sidebarCollapsed, setSidebarCollapsed] = useState(
    () => typeof window !== 'undefined' && window.localStorage.getItem('camelid.sidebarCollapsed') === 'true',
  )
  const [mobileNavOpen, setMobileNavOpen] = useState(false)
  const [isMobile, setIsMobile] = useState(
    () => typeof window !== 'undefined' && window.matchMedia('(max-width: 860px)').matches,
  )
  const [pendingDeleteConversationId, setPendingDeleteConversationId] = useState(null)
  const [deleteBusy, setDeleteBusy] = useState(false)
  const [paletteOpen, setPaletteOpen] = useState(false)
  const [shortcutsOpen, setShortcutsOpen] = useState(false)

  useEffect(() => {
    if (typeof window === 'undefined' || !window.matchMedia) return undefined
    const media = window.matchMedia('(max-width: 860px)')
    const sync = () => {
      setIsMobile(media.matches)
      if (!media.matches) setMobileNavOpen(false)
    }
    media.addEventListener('change', sync)
    return () => media.removeEventListener('change', sync)
  }, [])

  useEffect(() => {
    if (typeof window === 'undefined') return
    window.localStorage.setItem('camelid.sidebarCollapsed', String(sidebarCollapsed))
  }, [sidebarCollapsed])

  const consoleState = useConsole({ showNotice, clearNotice })
  const {
    runtime, models, conversations, filteredConversations,
    selectedConversation, setSelectedConversationId,
    selectedModel, selectedModelId, setSelectedModelId,
    composer, setComposer, search, setSearch, sending, stoppingGeneration,
    pendingConversation, tab, setTab, sendMessage, stopGeneration,
    showNewChatLanding, renameConversation, deleteConversation, deleteAllConversations,
  } = consoleState

  useEffect(() => {
    if (typeof window === 'undefined') return
    const hash = window.location.hash.replace('#', '')
    if (HASH_TABS.has(hash)) setTab(hash)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  useEffect(() => {
    const onKeyDown = (event) => {
      if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === 'k') {
        event.preventDefault()
        setShortcutsOpen(false)
        setPaletteOpen((value) => !value)
        return
      }
      const typing =
        ['INPUT', 'TEXTAREA', 'SELECT'].includes(document.activeElement?.tagName) ||
        document.activeElement?.isContentEditable
      if (event.key === '?' && !typing && !event.metaKey && !event.ctrlKey) {
        event.preventDefault()
        setPaletteOpen(false)
        setShortcutsOpen((value) => !value)
      }
    }
    window.addEventListener('keydown', onKeyDown)
    return () => window.removeEventListener('keydown', onKeyDown)
  }, [])

  const closeMobileNav = () => setMobileNavOpen(false)

  const navigateTab = (next) => {
    setTab(next)
    if (typeof window !== 'undefined' && HASH_TABS.has(next)) {
      window.history.replaceState(null, '', next === 'chat' ? window.location.pathname : `#${next}`)
    }
    closeMobileNav()
  }

  const selectConversation = (id) => {
    setSelectedConversationId(id)
    setTab('chat')
    closeMobileNav()
  }

  const pendingDeleteConversation = useMemo(
    () => conversations.find((conversation) => conversation.id === pendingDeleteConversationId) || null,
    [conversations, pendingDeleteConversationId],
  )

  const pendingDeleteCopy = useMemo(() => {
    if (!pendingDeleteConversation) return ''
    const latest = [...(pendingDeleteConversation.messages || [])]
      .reverse()
      .find((message) => typeof message?.content === 'string' && message.content.trim())
    const preview = formatPreview(latest?.content, 80)
    return preview === 'No messages yet'
      ? 'This conversation will be permanently removed.'
      : `“${preview}” — this conversation will be permanently removed.`
  }, [pendingDeleteConversation])

  const handleDeleteConfirm = async () => {
    if (!pendingDeleteConversationId || deleteBusy) return
    setDeleteBusy(true)
    const ok = await deleteConversation(pendingDeleteConversationId)
    if (ok) setPendingDeleteConversationId(null)
    setDeleteBusy(false)
  }

  /* The gateway has not answered yet. Showing the sign-in form here would ask
     for a credential before we know whether this deployment wants one. */
  if (auth.status === AuthStatus.Verifying) {
    return (
      <div className="loading-shell">
        <div className="loading-shell-stack">
          <div>Connecting…</div>
        </div>
      </div>
    )
  }

  if (auth.status !== AuthStatus.SignedIn) {
    return <LoginScreen />
  }

  const shellClasses = ['camelid-app', sidebarCollapsed ? 'is-collapsed' : '', mobileNavOpen ? 'is-mobile-open' : '']
    .filter(Boolean)
    .join(' ')

  return (
    <div className={shellClasses}>
      <SidebarRail
        collapsed={!isMobile && sidebarCollapsed}
        onToggleCollapsed={() => setSidebarCollapsed((value) => !value)}
        showNewChatLanding={() => {
          showNewChatLanding()
          closeMobileNav()
        }}
        search={search}
        setSearch={setSearch}
        tab={tab}
        setTab={navigateTab}
        filteredConversations={filteredConversations}
        selectedConversationId={selectedConversation?.id || null}
        onSelectConversation={selectConversation}
        renameConversation={renameConversation}
        requestDeleteConversation={(id) => {
          setPendingDeleteConversationId(id)
          setDeleteBusy(false)
        }}
        runtime={runtime}
        themePreference={preference}
        themeResolved={resolved}
        onCycleTheme={cyclePreference}
      />

      {mobileNavOpen && (
        <button
          type="button"
          className="camelid-app__scrim"
          aria-label="Close navigation"
          onClick={closeMobileNav}
        />
      )}

      <main className="camelid-main" data-view={tab}>
        <TopBar
          tab={tab}
          setTab={navigateTab}
          selectedConversationTitle={selectedConversation?.title || ''}
          runtime={runtime}
          selectedModelId={selectedModelId}
          models={models}
          identity={auth.identity}
          authMode={auth.authMode}
          onSignOut={signOut}
          onToggleSidebar={() => setMobileNavOpen((value) => !value)}
        />

        {notice && (
          <div className="camelid-notice-slot">
            <Notice notice={notice} tone={noticeTone} onDismiss={clearNotice} />
          </div>
        )}

        <div className={`camelid-view ${tab === 'chat' ? 'camelid-view--chat' : 'camelid-view--page'}`}>
          <Suspense
            fallback={
              <div className="view-loading" role="status" aria-label="Loading view">
                Loading view…
              </div>
            }
          >
            {tab === 'chat' && (
              <ChatWorkspace
                selectedConversation={selectedConversation}
                selectedModel={selectedModel}
                selectedModelId={selectedModelId}
                setSelectedModelId={setSelectedModelId}
                models={models}
                runtime={runtime}
                pendingConversation={pendingConversation}
                composer={composer}
                setComposer={setComposer}
                sendMessage={sendMessage}
                stopGeneration={stopGeneration}
                sending={sending}
                stoppingGeneration={stoppingGeneration}
                setTab={navigateTab}
                showNewChatLanding={showNewChatLanding}
              />
            )}
            {tab === 'history' && (
              <HistoryView
                conversations={conversations}
                onSelectConversation={selectConversation}
                onRequestDelete={(id) => setPendingDeleteConversationId(id)}
                onDeleteAll={deleteAllConversations}
                search={search}
                setSearch={setSearch}
              />
            )}
            {tab === 'settings' && (
              <SettingsView
                runtime={runtime}
                identity={auth.identity}
                authMode={auth.authMode}
                onSignOut={signOut}
                selectedModelId={selectedModelId}
                themePreference={preference}
                onSelectTheme={setPreference}
                showNotice={showNotice}
              />
            )}
          </Suspense>
        </div>
      </main>

      <CommandPalette
        open={paletteOpen}
        onClose={() => setPaletteOpen(false)}
        conversations={conversations}
        onSelectConversation={selectConversation}
        setTab={navigateTab}
        onNewChat={showNewChatLanding}
      />
      <ShortcutsOverlay open={shortcutsOpen} onClose={() => setShortcutsOpen(false)} />

      <ConfirmDialog
        open={Boolean(pendingDeleteConversationId)}
        title="Delete conversation?"
        body={pendingDeleteCopy}
        confirmLabel="Delete"
        busy={deleteBusy}
        onConfirm={handleDeleteConfirm}
        onCancel={() => setPendingDeleteConversationId(null)}
      />
    </div>
  )
}
