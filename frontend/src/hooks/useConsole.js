import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { getAuthHeaders, getAuthState, getGatewayUrl, isAuthenticated, subscribeAuth } from '../lib/authStore'
import { fetchEnterpriseJson } from '../lib/enterpriseApi'
import { readStreamingChatCompletion } from '../lib/chatCompletionStream'
import { getConversationsStorageKey, normalizeStoredConversations } from '../lib/conversationStorage'
import { NEW_CHAT_SENTINEL, resolveSelectedConversation } from '../lib/chatState'
import { getConfiguredMaxTokens } from '../lib/responseLimits'
import { extractAttributionFromHeaders } from '../lib/attribution'

/* Console state, built on the ten routes a replica publishes and nothing else.
 *
 * What this replaces was a dashboard hook for a single-user desktop application:
 * it scanned a models directory, loaded and unloaded weights, drove downloads,
 * and read a capability contract — about two dozen `/api/*` calls, none of which
 * exist here. They are absent by design rather than by omission. A replica
 * publishes a model digest, a config digest and a host summary that describe the
 * weights it hashed before it bound its port; a control plane that could swap the
 * model over that same port would leave all three describing something the
 * process is no longer serving.
 *
 * So this hook reads. Health, the model list, and generation — everything else a
 * deployment does (which model, which host, how many replicas) is an operator
 * action taken elsewhere, and the console's job is to show the result honestly.
 */

const HEALTH_POLL_MS = 5000

function conversationsKey() {
  /* Per principal, so two people signing into the same browser never see each
     other's threads. Conversations are client-side only — the replica keeps no
     history, and the gateway's audit log is an operator record, not a transcript
     store. */
  return getConversationsStorageKey(getAuthState().identity?.principalId || null)
}

function loadConversations() {
  if (typeof window === 'undefined') return []
  try {
    const raw = window.localStorage.getItem(conversationsKey())
    if (!raw) return []
    /* `clearStaleStreaming` matters on reload: a thread persisted mid-stream
       would otherwise come back with an assistant turn stuck in a streaming
       state that nothing is feeding. */
    return normalizeStoredConversations(JSON.parse(raw), { clearStaleStreaming: true })
  } catch {
    return []
  }
}

function persistConversations(conversations) {
  if (typeof window === 'undefined') return
  try {
    window.localStorage.setItem(conversationsKey(), JSON.stringify(conversations))
  } catch {
    /* Quota or private-mode failures must not take the chat down; the in-memory
       thread stays usable for this session. */
  }
}

const newId = () => `c_${Math.random().toString(36).slice(2, 10)}${Date.now().toString(36)}`

export function useConsole({ showNotice, clearNotice }) {
  const [runtime, setRuntime] = useState({ status: 'unknown' })
  const [models, setModels] = useState([])
  const [conversations, setConversations] = useState(loadConversations)
  const [selectedConversationId, setSelectedConversationId] = useState(NEW_CHAT_SENTINEL)
  const [selectedModelId, setSelectedModelId] = useState('')
  const [composer, setComposer] = useState('')
  const [search, setSearch] = useState('')
  const [sending, setSending] = useState(false)
  const [stoppingGeneration, setStoppingGeneration] = useState(false)
  const [pendingConversation, setPendingConversation] = useState(null)
  const [tab, setTab] = useState('chat')

  const [signedIn, setSignedIn] = useState(isAuthenticated)

  /* Tracked as state so the poll below restarts the moment a session begins,
     rather than waiting for some other render to happen to come along. */
  useEffect(() => subscribeAuth(() => setSignedIn(isAuthenticated())), [])

  const abortRef = useRef(null)

  useEffect(() => persistConversations(conversations), [conversations])

  /* ---- Health ----
     Polled rather than streamed: there is no telemetry route in the contract,
     and a poll that says "unreachable" is better than a socket that silently
     stops delivering. */
  const readHealth = useCallback(async () => {
    try {
      const health = await fetchEnterpriseJson('/v1/health')
      /* Spread first, then stamp: `status` here means "we reached it", and the
         replica sends its own `status` field ("ok") which would otherwise
         overwrite ours and leave every reachability check reading a word it does
         not recognize. Liveness that matters is `loaded_now` /
         `generation_ready`, both of which come through untouched. */
      setRuntime({ ...health, status: 'online' })
    } catch (error) {
      /* A 401 already raised `camelid:auth-error` inside the API layer, which
         ends the session; treating it as "offline" here too would be a
         misleading second story about the same event. */
      if (error.status !== 401) setRuntime({ status: 'offline', error: error.message })
    }
  }, [])

  const readModels = useCallback(async () => {
    try {
      const listing = await fetchEnterpriseJson('/v1/models')
      const rows = Array.isArray(listing?.data) ? listing.data : []
      setModels(
        rows.map((row) => ({
          id: row.id,
          name: row.id,
          /* Descriptive shape metadata only. It is never promoted into a
             readiness signal — `/v1/health` is the only thing that decides
             whether this endpoint can generate. */
          meta: row.meta || null,
        })),
      )
    } catch (error) {
      if (error.status !== 401) setModels([])
    }
  }, [])

  /* Nothing is polled until there is a session.
   *
   * This hook is called unconditionally — hooks cannot be conditional — so
   * without this gate the health poll starts at mount and runs while the user is
   * still on the sign-in screen. Every one of those requests is a 401, and the
   * gateway audits refusals by design: the console would be writing a steady
   * stream of authentication failures into an operator's security log, caused by
   * nothing but its own polling. An operator watching for credential attacks
   * would be reading our noise. */
  useEffect(() => {
    if (!isAuthenticated()) return undefined
    readHealth()
    readModels()
    const timer = setInterval(readHealth, HEALTH_POLL_MS)
    return () => clearInterval(timer)
  }, [readHealth, readModels, signedIn])

  /* The served model is the default selection. A replica serves exactly one, so
     asking the operator to pick it would be asking about something they cannot
     change from here. */
  useEffect(() => {
    if (!selectedModelId && runtime.active_model_id) setSelectedModelId(runtime.active_model_id)
  }, [runtime.active_model_id, selectedModelId])

  const selectedConversation = useMemo(
    () => resolveSelectedConversation(conversations, selectedConversationId),
    [conversations, selectedConversationId],
  )

  const filteredConversations = useMemo(() => {
    const needle = search.trim().toLowerCase()
    if (!needle) return conversations
    return conversations.filter((conversation) => {
      if (conversation.title?.toLowerCase().includes(needle)) return true
      return (conversation.messages || []).some((message) =>
        String(message.content || '').toLowerCase().includes(needle),
      )
    })
  }, [conversations, search])

  const selectedModel = useMemo(
    () => models.find((model) => model.id === selectedModelId) || null,
    [models, selectedModelId],
  )

  const updateConversation = useCallback((id, updater) => {
    setConversations((current) =>
      current.map((conversation) => (conversation.id === id ? updater(conversation) : conversation)),
    )
  }, [])

  const showNewChatLanding = useCallback(() => {
    setSelectedConversationId(NEW_CHAT_SENTINEL)
    setComposer('')
    setTab('chat')
  }, [])

  const sendMessage = useCallback(
    async (overrideText = null) => {
      /* Only a string counts as an override. Wired straight to an `onClick` this
         receives the click event, and `String(event)` is "[object Object]" — a
         perfectly valid-looking prompt that silently replaces what the user
         typed. Ignoring non-strings makes that miswiring a no-op instead. */
      const override = typeof overrideText === 'string' ? overrideText : null
      const text = String(override ?? composer).trim()
      if (!text || sending) return

      clearNotice?.()

      /* A thread is created on first send rather than on "new chat", so
         abandoning the landing screen leaves nothing behind. */
      let conversationId = selectedConversation?.id
      if (!conversationId) {
        conversationId = newId()
        setConversations((current) => [
          {
            id: conversationId,
            title: text.slice(0, 60),
            created_at: new Date().toISOString(),
            messages: [],
          },
          ...current,
        ])
        setSelectedConversationId(conversationId)
      }

      const userMessage = { id: newId(), role: 'user', content: text }
      const assistantId = newId()
      updateConversation(conversationId, (conversation) => ({
        ...conversation,
        messages: [
          ...(conversation.messages || []),
          userMessage,
          { id: assistantId, role: 'assistant', content: '', streaming: true },
        ],
      }))

      setComposer('')
      setPendingConversation(null)
      setSending(true)

      const controller = new AbortController()
      abortRef.current = controller

      const history = [
        ...(selectedConversation?.messages || [])
          .filter((message) => !message.streaming && message.content)
          .map((message) => ({ role: message.role, content: message.content })),
        { role: 'user', content: text },
      ]

      try {
        const response = await fetch(`${getGatewayUrl()}/v1/chat/completions`, {
          method: 'POST',
          signal: controller.signal,
          headers: {
            'Content-Type': 'application/json',
            Accept: 'text/event-stream',
            ...getAuthHeaders(),
          },
          body: JSON.stringify({
            model: selectedModelId || runtime.active_model_id,
            messages: history,
            stream: true,
            max_tokens: getConfiguredMaxTokens(selectedModelId),
          }),
        })

        if (response.status === 401) {
          window.dispatchEvent(new CustomEvent('camelid:auth-error', { detail: { status: 401 } }))
          throw new Error('Your session ended. Sign in again.')
        }

        /* The replica stamps its identity on every response. Captured per turn so
           a transcript can still say which configuration and which weights
           produced it, even after the endpoint has been restarted or repointed. */
        const attribution = extractAttributionFromHeaders(response.headers)

        const result = await readStreamingChatCompletion(response, (_delta, full) => {
          updateConversation(conversationId, (conversation) => ({
            ...conversation,
            messages: (conversation.messages || []).map((message) =>
              message.id === assistantId ? { ...message, content: full } : message,
            ),
          }))
        })

        updateConversation(conversationId, (conversation) => ({
          ...conversation,
          messages: (conversation.messages || []).map((message) =>
            message.id === assistantId
              ? {
                  ...message,
                  content: result.content,
                  streaming: false,
                  finish_reason: result.finishReason,
                  usage: result.usage,
                  attribution,
                }
              : message,
          ),
        }))
      } catch (error) {
        const aborted = error.name === 'AbortError'
        updateConversation(conversationId, (conversation) => ({
          ...conversation,
          messages: (conversation.messages || []).map((message) =>
            message.id === assistantId
              ? {
                  ...message,
                  streaming: false,
                  content: message.content || (aborted ? '(generation stopped)' : ''),
                  error: aborted ? null : error.message,
                }
              : message,
          ),
        }))
        if (!aborted) showNotice?.(error.message, 'error')
      } finally {
        abortRef.current = null
        setSending(false)
        setStoppingGeneration(false)
      }
    },
    [
      clearNotice,
      composer,
      runtime.active_model_id,
      selectedConversation,
      selectedModelId,
      sending,
      showNotice,
      updateConversation,
    ],
  )

  const stopGeneration = useCallback(() => {
    if (!abortRef.current) return
    setStoppingGeneration(true)
    abortRef.current.abort()
  }, [])

  const renameConversation = useCallback(
    (id, title) => updateConversation(id, (conversation) => ({ ...conversation, title })),
    [updateConversation],
  )

  const deleteConversation = useCallback(
    async (id) => {
      setConversations((current) => current.filter((conversation) => conversation.id !== id))
      if (selectedConversationId === id) setSelectedConversationId(NEW_CHAT_SENTINEL)
      return true
    },
    [selectedConversationId],
  )

  const deleteAllConversations = useCallback(async () => {
    setConversations([])
    setSelectedConversationId(NEW_CHAT_SENTINEL)
    return true
  }, [])

  return {
    runtime,
    models,
    conversations,
    filteredConversations,
    selectedConversation,
    selectedConversationId,
    setSelectedConversationId,
    selectedModel,
    selectedModelId,
    setSelectedModelId,
    composer,
    setComposer,
    search,
    setSearch,
    sending,
    stoppingGeneration,
    pendingConversation,
    tab,
    setTab,
    sendMessage,
    stopGeneration,
    showNewChatLanding,
    renameConversation,
    deleteConversation,
    deleteAllConversations,
    refresh: () => {
      readHealth()
      readModels()
    },
  }
}
