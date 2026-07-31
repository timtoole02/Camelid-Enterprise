import { useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react'
import { getServiceReadiness, isModelServable, Readiness } from '../lib/serviceReadiness'
import { getConfiguredMaxTokens, modelContextLength, validateSendBudget } from '../lib/responseLimits'
import { CamelidMark } from '../components/ui/CamelidMark'
import { Avatar } from '../components/ui/Avatar'
import { StatusDot } from '../components/ui/StatusDot'
import { IconSend, IconStop, IconBolt, IconChart, IconChat, IconEdit } from '../components/ui/icons'
import { MessageTurn } from '../components/chat/MessageTurn'
import { PREPARING_STREAMING_LABEL, StreamingLoader } from '../components/chat/render/StreamingIndicator'

const isBootstrapMessage = (message) =>
  message?.role === 'assistant' &&
  typeof message?.content === 'string' &&
  message.content.startsWith('Conversation created.')

const isInterruptedPlaceholderMessage = (message) => {
  if (message?.role !== 'assistant') return false
  const content = String(message?.content || '').trim().toLowerCase()
  return content === '(generation interrupted)' || content === '(generation stopped)'
}

function readinessTone({ ready = false, blocked = false, offline = false, waiting = false } = {}) {
  if (ready) return 'ready'
  if (offline || blocked) return 'blocked'
  if (waiting) return 'waiting'
  return 'idle'
}

const SUGGESTIONS = [
  { title: 'Summarize this plan', body: 'Summarize this implementation plan and call out the risks', Icon: IconChart },
  { title: 'Draft a release note', body: 'Draft a concise release note from these changes', Icon: IconEdit },
  { title: 'Prioritize next steps', body: 'Turn this checklist into a prioritized next-step plan', Icon: IconBolt },
  { title: 'Tighten this answer', body: 'Review this response and tighten it into a shorter final answer', Icon: IconChat },
]

const FOLLOW_UP_PROMPTS = [
  'Continue with the exact next steps.',
  'Tighten that into a shorter final answer.',
  'Turn this into a checklist I can execute.',
]

export default function ChatWorkspace({
  selectedConversation,
  selectedModel,
  selectedModelId,
  setSelectedModelId,
  models,
  runtime,
  pendingConversation,
  composer,
  setComposer,
  sendMessage,
  stopGeneration,
  sending,
  stoppingGeneration = false,
  setTab,
  showNewChatLanding = null,
}) {
  const [generationElapsedSeconds, setGenerationElapsedSeconds] = useState(0)
  const [showAllMessages, setShowAllMessages] = useState(false)
  const [userScrolledAway, setUserScrolledAway] = useState(false)
  const chatBottomRef = useRef(null)
  const composerRef = useRef(null)
  const autoFollowGenerationRef = useRef(true)
  const composerReadinessId = 'camelid-chat-readiness-note'

  const rawVisibleMessages = useMemo(
    () => (selectedConversation?.messages || []).filter((message) => !isBootstrapMessage(message)),
    [selectedConversation?.messages],
  )
  const hasStreamingAssistant = rawVisibleMessages.some((m) => m.role === 'assistant' && m.streaming)
  const hasStreamingAssistantContent = rawVisibleMessages.some((m) => m.role === 'assistant' && m.streaming && String(m.content || '').trim())
  const generationActive = Boolean(sending || hasStreamingAssistant)
  const visibleMessages = useMemo(() => {
    if (!generationActive) return rawVisibleMessages
    return rawVisibleMessages.filter((message, index, messages) => {
      const isTrailingInterruptedPlaceholder = index === messages.length - 1 && isInterruptedPlaceholderMessage(message)
      return !isTrailingInterruptedPlaceholder
    })
  }, [generationActive, rawVisibleMessages])
  const pendingPrompt = (pendingConversation?.content || (sending ? composer.trim() : '')).trim()
  const pendingPromptAlreadyVisible = Boolean(
    pendingPrompt && [...visibleMessages].reverse().some((m) => m.role === 'user' && m.content === pendingPrompt),
  )
  const pendingUserPrompt = pendingPromptAlreadyVisible ? '' : pendingPrompt
  const lastVisibleMessage = visibleMessages.at(-1)
  const lastVisibleMessageIsUser = lastVisibleMessage?.role === 'user'
  const awaitingAssistant = Boolean(generationActive && !hasStreamingAssistantContent && !hasStreamingAssistant && (pendingPrompt || lastVisibleMessageIsUser || sending))
  const streamingScrollSignature = useMemo(() => (
    visibleMessages.map((m) => `${m.id}:${m.streaming ? 'streaming' : 'done'}:${String(m.content || '').length}`).join('|')
    + `|awaiting:${awaitingAssistant ? '1' : '0'}|active:${generationActive ? '1' : '0'}`
  ), [awaitingAssistant, generationActive, visibleMessages])
  const isFreshThread = selectedConversation
    ? (visibleMessages.length === 0 && !pendingPrompt && !awaitingAssistant && !hasStreamingAssistant)
    : (!pendingPrompt && !awaitingAssistant && !hasStreamingAssistant)

  /* ----- Readiness, derived from /v1/health alone -----
     Sending is unlocked by the service's own readiness and nothing else. The
     previous revision also required the selected model to match a row in a
     published compatibility contract, which this deployment has no route to
     serve — so that condition could never become true and send would never
     unlock. */
  const readiness = getServiceReadiness(runtime, selectedModelId)
  const canChat = readiness.canSend
  const apiUnavailable = readiness.state === Readiness.Offline
  const selectedRuntimeReady = readiness.canSend
  const selectedRuntimeMatchesLoadedModel = readiness.state !== Readiness.OtherModel
  const selectedModelName = selectedModel?.name || selectedModelId || runtime?.active_model_id || 'No model selected'
  const selectedModelIssue = selectedModel?.load_error || ''

  const runtimeStatusLabel = readiness.label
  const runtimeStatusCopy = readiness.copy
  const readinessFinePrint = apiUnavailable
    ? 'Drafts stay editable while the gateway reconnects.'
    : readiness.copy
  const servingOtherModel = readiness.state === Readiness.OtherModel
  const selectedModelReadinessCopy = selectedModelIssue || readiness.copy
  const selectedModelGateSummary = selectedModelReadinessCopy

  const productHeroTitle = selectedRuntimeReady ? 'How can I help?' : "Hi there, let's get into it"
  const productHeroSummary = selectedRuntimeReady
    ? 'Ask anything — responses come from the model this endpoint serves.'
    : apiUnavailable
      ? 'Keep writing here. Send unlocks again once the gateway responds.'
      : readiness.copy

  const readinessState = selectedRuntimeReady
    ? 'ready'
    : apiUnavailable
      ? 'offline'
      : servingOtherModel
        ? 'blocked'
        : 'waiting'
  const runtimeTone = readinessTone({
    ready: selectedRuntimeReady,
    offline: apiUnavailable,
    blocked: servingOtherModel,
    waiting: Boolean(runtime?.loaded_now),
  })
  const statusTone = selectedRuntimeReady
    ? 'ready'
    : apiUnavailable
      ? 'offline'
      : servingOtherModel
        ? 'warn'
        : runtime?.loaded_now
          ? 'warn'
          : 'neutral'

  const selectionSummaryCopy = selectedModelIssue || readiness.copy

  const canSubmit = Boolean(composer.trim()) && canChat && !generationActive
  const sendDisabledReason = canChat
    ? ''
    : generationActive
      ? 'Wait for the current reply to finish or stop it before sending again.'
      : readiness.copy
  const promptHintCopy = canChat
    ? 'Enter sends · Shift+Enter for a new line'
    : apiUnavailable
      ? 'Draft now · send unlocks after the gateway reconnects'
      : 'Draft now · send unlocks when the service is ready'
  const composerHintCopy = canSubmit ? promptHintCopy : sendDisabledReason || promptHintCopy

  const composerDraftUnlocked = Boolean(selectedModel || apiUnavailable)
  const composerDisabled = !composerDraftUnlocked
  const composerPlaceholder = canChat
    ? 'Message Camelid…'
    : apiUnavailable
      ? 'Draft a prompt while the service comes back'
      : composerDraftUnlocked
        ? 'Draft a prompt while Camelid finishes getting ready'
        : isFreshThread
          ? 'Load a model first'
          : 'Choose a ready model first'
  const composerStopLabel = stoppingGeneration ? 'Stopping…' : 'Stop'

  // ----- Effects -----
  useEffect(() => {
    if (!generationActive) {
      setGenerationElapsedSeconds(0)
      return undefined
    }
    setGenerationElapsedSeconds(0)
    const startedAt = Date.now()
    const interval = window.setInterval(() => {
      setGenerationElapsedSeconds(Math.max(1, Math.floor((Date.now() - startedAt) / 1000)))
    }, 1000)
    return () => window.clearInterval(interval)
  }, [generationActive])

  useEffect(() => {
    if (!generationActive) return undefined
    autoFollowGenerationRef.current = true
    setUserScrolledAway(false)
    const updateAutoFollow = () => {
      const el = document.querySelector('.cxchat__scroll')
      if (!el) return
      const distanceFromBottom = el.scrollHeight - (el.scrollTop + el.clientHeight)
      const follow = distanceFromBottom < 260
      if (follow !== autoFollowGenerationRef.current) setUserScrolledAway(!follow)
      autoFollowGenerationRef.current = follow
    }
    const el = document.querySelector('.cxchat__scroll')
    el?.addEventListener('scroll', updateAutoFollow, { passive: true })
    return () => el?.removeEventListener('scroll', updateAutoFollow)
  }, [generationActive, selectedConversation?.id])

  useLayoutEffect(() => {
    if (!generationActive || !autoFollowGenerationRef.current) return undefined
    const frame = window.requestAnimationFrame(() => {
      chatBottomRef.current?.scrollIntoView({ block: 'end', behavior: 'auto' })
    })
    return () => window.cancelAnimationFrame(frame)
  }, [generationActive, streamingScrollSignature])

  useLayoutEffect(() => {
    const input = composerRef.current
    if (!input) return
    input.style.height = 'auto'
    input.style.height = `${Math.min(input.scrollHeight, 220)}px`
  }, [composer, isFreshThread, selectedConversation?.id])

  useEffect(() => {
    if (generationActive || !composerDraftUnlocked) return
    const input = composerRef.current
    if (!input) return
    const activeElement = document.activeElement
    if (activeElement && activeElement !== document.body && activeElement !== input) return
    const frame = window.requestAnimationFrame(() => input.focus())
    return () => window.cancelAnimationFrame(frame)
  }, [composerDraftUnlocked, generationActive, isFreshThread, selectedConversation?.id])

  const handleComposerKeyDown = async (event) => {
    if (event.key === 'Escape' && generationActive) {
      event.preventDefault()
      stopGeneration?.()
      return
    }
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault()
      if (canSubmit) await sendMessage()
    }
  }

  const handleSuggestion = (prompt) => {
    if (!composerDraftUnlocked) return
    setComposer(prompt)
  }

  /* Model picker. A replica serves one model per process, so at most one entry
     here is servable; the rest are listed because the endpoint advertises them,
     not because this console can switch to them. */
  const runnableModels = models.filter((model) => isModelServable(model, runtime))
  const waitingModels = models.filter((model) => !isModelServable(model, runtime))
  const selectedPickerModelId = models.some((model) => model.id === selectedModel?.id) ? selectedModel.id : ''
  const modelOptionLabel = (model) => {
    if (isModelServable(model, runtime)) return `${model.name} · Ready`
    if (apiUnavailable) return `${model.name} · Service unavailable`
    if (runtime?.active_model_id) return `${model.name} · Not served here`
    return `${model.name} · Not loaded`
  }

  /* Send-time budget check: the response limit is an upper bound the backend
     clamps to the context's remaining room, so an overshoot is a non-blocking
     notice — only a prompt that fills the whole context is a hard error. Prompt
     size is a client estimate, labeled as such. */
  const estimatedPromptTokens = useMemo(() => {
    const history = visibleMessages.map((m) => String(m.content || '')).join(' ')
    const text = `${history} ${composer}`
    const pieces = text.match(/[\p{L}\p{N}_]+|[^\s\p{L}\p{N}_]/gu) || []
    return Math.max(1, Math.round(Math.max(pieces.length, text.length / 4)))
  }, [visibleMessages, composer])
  const sendBudget = validateSendBudget({
    promptTokens: estimatedPromptTokens,
    maxTokens: getConfiguredMaxTokens(selectedModelId),
    contextLength: modelContextLength(selectedModel),
  })

  const detailCopy = selectedRuntimeReady ? selectionSummaryCopy : (selectedModelIssue || readinessFinePrint)

  const renderComposer = () => (
    <div className={`cxcomposer is-${readinessState}`}>
      <div className="cxcomposer__box">
        <textarea
          ref={composerRef}
          className="cxcomposer__input"
          aria-label="Message Camelid"
          aria-describedby={composerReadinessId}
          value={composer}
          onChange={(e) => setComposer(e.target.value)}
          onKeyDown={handleComposerKeyDown}
          rows={1}
          placeholder={composerPlaceholder}
          disabled={composerDisabled}
        />
        <div className="cxcomposer__toolbar">
          <div className="cxcomposer__tools">
            {models.length ? (
              <label className="cxcomposer__model" title={readiness.copy}>
                <span className="sr-only">Choose model for chat</span>
                <select
                  className="cxcomposer__model-select"
                  aria-label="Choose model for chat"
                  value={selectedPickerModelId}
                  onChange={(e) => {
                    const id = e.target.value
                    if (!id) return
                    /* Selection only. Choosing a model here cannot load one —
                       there is no route for that, and the endpoint is already
                       serving whatever it hashed at startup. */
                    setSelectedModelId(id)
                  }}
                  disabled={generationActive}
                >
                  {!selectedModel && <option value="">Choose model</option>}
                  {runnableModels.length > 0 && (
                    <optgroup label="Ready">
                      {runnableModels.map((model) => <option key={model.id} value={model.id}>{modelOptionLabel(model)}</option>)}
                    </optgroup>
                  )}
                  {waitingModels.length > 0 && (
                    <optgroup label="Needs readiness">
                      {waitingModels.map((model) => <option key={model.id} value={model.id}>{modelOptionLabel(model)}</option>)}
                    </optgroup>
                  )}
                </select>
              </label>
            ) : null}
          </div>
          <div className="cxcomposer__actions">
            {generationActive && (
              <button type="button" className="cxcomposer__stop" aria-label="Stop Camelid generation" onClick={stopGeneration} disabled={stoppingGeneration}>
                <IconStop size={16} /> {composerStopLabel}
              </button>
            )}
            <button
              type="button"
              className="cxcomposer__send"
              aria-label="Send message"
              data-send-ready={canSubmit ? 'true' : 'false'}
              title={!canSubmit ? sendDisabledReason : 'Send message to Camelid'}
              onClick={() => sendMessage()}
              disabled={!canSubmit || sendBudget.level === 'error'}
            >
              <IconSend size={20} />
            </button>
          </div>
        </div>
      </div>

      {sendBudget.level === 'error' && (
        <p className="cxcomposer__budget-error" role="alert">
          <span aria-hidden="true">✕</span> {sendBudget.message}
        </p>
      )}
      {sendBudget.level === 'notice' && (
        <p className="cxcomposer__budget-notice">
          <span aria-hidden="true">ⓘ</span> {sendBudget.message}
        </p>
      )}
      <div className={`cxcomposer__status is-${statusTone}`} role="status" aria-live="polite" title={`${runtimeStatusCopy} ${readinessFinePrint}`}>
        <StatusDot tone={statusTone} pulse={selectedRuntimeReady} />
        <strong className="cxcomposer__status-label">{runtimeStatusLabel}</strong>
        <span className="cxcomposer__status-sep" aria-hidden="true">·</span>
        <span className="cxcomposer__status-model">{selectedModelName}</span>
      </div>
      {detailCopy !== composerHintCopy && (
        <p id={composerReadinessId} className="cxcomposer__detail">{detailCopy}</p>
      )}
      <p className="cxcomposer__hint">{composerHintCopy}</p>
    </div>
  )

  return (
    <section className={`cxchat is-${readinessState} ${userScrolledAway ? 'is-user-scrolled' : ''} ${isFreshThread ? 'cxchat--empty' : ''}`} data-view="chat">
      <div className="cxchat__scroll">
        <div className="cxchat__column">
          {servingOtherModel && (
            <div className="cxchat__experimental-banner" role="note">
              <span>{readiness.copy}</span>
            </div>
          )}
          {isFreshThread ? (
            <div className="cxchat__empty">
              <div className="cxchat-hero">
                <CamelidMark size={52} className="cxchat-hero__mark" />
                <h2 className="cxchat-hero__title">{productHeroTitle}</h2>
                <p className="cxchat-hero__summary">{productHeroSummary}</p>
              </div>
              {composerDraftUnlocked && (
                <div className="cxchat__suggestions" aria-label="Prompt starters">
                  {SUGGESTIONS.map(({ title, body, Icon }) => (
                    <button key={body} type="button" className="cxchat__suggestion" onClick={() => handleSuggestion(body)} disabled={!composerDraftUnlocked}>
                      <span className="cxchat__suggestion-text">{body}</span>
                      <span className="cxchat__suggestion-icon"><Icon size={18} /></span>
                    </button>
                  ))}
                </div>
              )}
            </div>
          ) : (
            <div className="cxchat__thread">
              {visibleMessages.length > 0 && !generationActive && selectedRuntimeReady && (
                <div className="cxchat__followups" aria-label="Follow-up prompts">
                  {FOLLOW_UP_PROMPTS.map((prompt) => (
                    <button key={prompt} type="button" className="cxchat__followup" onClick={() => handleSuggestion(prompt)}>{prompt}</button>
                  ))}
                </div>
              )}
              {/* Long-thread windowing: render the latest 60 turns;
                  earlier turns mount on demand. Keeps streaming smooth without
                  a virtualization dependency. */}
              {!showAllMessages && visibleMessages.length > 60 && (
                <button type="button" className="cxchat__show-earlier" onClick={() => setShowAllMessages(true)}>
                  Show {visibleMessages.length - 60} earlier messages
                </button>
              )}
              {(showAllMessages ? visibleMessages : visibleMessages.slice(-60)).map((message) => {
                const index = visibleMessages.indexOf(message)
                const priorUserMessage = message.role === 'assistant'
                  ? [...visibleMessages.slice(0, index)].reverse().find((item) => item.role === 'user')
                  : null
                const priorUserPrompt = priorUserMessage?.content || null
                const canResend = false
                return (
                  <MessageTurn
                    key={message.id}
                    message={message}
                    generationElapsedSeconds={generationElapsedSeconds}
                    priorUserPrompt={priorUserPrompt}
                    onReusePrompt={setComposer}
                    onRegenerate={null}
                    onEditResend={null}
                  />
                )
              })}
              {generationActive && (
                <button
                  type="button"
                  className="cxchat__jump-latest"
                  data-autofollow-affordance
                  onClick={() => { autoFollowGenerationRef.current = true; setUserScrolledAway(false); chatBottomRef.current?.scrollIntoView({ block: 'end' }) }}
                >
                  ↓ jump to latest
                </button>
              )}
              {awaitingAssistant && (
                <>
                  {pendingUserPrompt && (
                    <article className="cxturn cxturn--user"><div className="cxturn__user-chip"><p>{pendingUserPrompt}</p></div></article>
                  )}
                  <article className="cxturn cxturn--assistant is-streaming" aria-busy="true" data-streaming-state="active">
                    <div className="cxturn__avatar"><Avatar size={30} state="awaiting" /></div>
                    <div className="cxturn__body"><StreamingLoader elapsedSeconds={generationElapsedSeconds} label={PREPARING_STREAMING_LABEL} /></div>
                  </article>
                </>
              )}
              <div className="cxchat__anchor" ref={chatBottomRef} aria-hidden="true" />
            </div>
          )}
        </div>
      </div>

      <div className="cxchat__dock">
        <div className="cxchat__column">
          {renderComposer()}
          <p className="cxchat__disclaimer">Replies are generated by the model this endpoint serves. Verify important output.</p>
        </div>
      </div>
    </section>
  )
}
