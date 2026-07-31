import { memo, useEffect, useRef, useState } from 'react'
import { Avatar } from '../ui/Avatar'
import { IconCopy, IconCheck, IconRefresh, IconEdit } from '../ui/icons'
import { AssistantMarkdown, copyText, hasOpenCodeFence } from '../../lib/markdown'
import { cleanLegacyDemoCapCopy } from '../../lib/conversationStorage'
import {
  LiveGenerationBadge,
  StreamingLoader,
  streamingStatusLabel,
} from './render/StreamingIndicator'

const formatMs = (value) => {
  const ms = Number(value)
  if (!Number.isFinite(ms) || ms <= 0) return null
  return ms >= 1000 ? `${(ms / 1000).toFixed(1)}s` : `${Math.round(ms)}ms`
}

const formatRate = (value) => {
  const rate = Number(value)
  if (!Number.isFinite(rate) || rate <= 0) return null
  return `${rate >= 10 ? Math.round(rate) : rate.toFixed(1)} tok/s`
}

import { formatShaShort } from '../../lib/attribution'

function EnterpriseAttributionBadge({ attribution }) {
  if (!attribution) return null
  const { lane, configSha256, modelSha256, host, workerThreads, requestId } = attribution

  return (
    <div
      className="cxturn__meta-attribution"
      style={{
        display: 'inline-flex',
        alignItems: 'center',
        gap: '6px',
        fontSize: '0.75rem',
        padding: '2px 8px',
        borderRadius: '12px',
        background: 'rgba(34, 197, 94, 0.08)',
        border: '1px solid rgba(34, 197, 94, 0.25)',
        color: 'var(--color-text-main)',
        marginTop: '4px',
      }}
      title={`Host: ${host || 'unknown'} | Worker threads: ${workerThreads || 'pool default'} | Request ID: ${requestId || 'none'}`}
    >
      <span style={{ color: '#22c55e', fontWeight: 600 }}>✓ {lane || 'deterministic'}</span>
      {configSha256 && <span style={{ opacity: 0.8 }}>cfg: <code>{formatShaShort(configSha256)}</code></span>}
      {modelSha256 && <span style={{ opacity: 0.8 }}>mdl: <code>{formatShaShort(modelSha256)}</code></span>}
      {requestId && <span style={{ opacity: 0.7 }}>id: <code>{formatShaShort(requestId)}</code></span>}
    </div>
  )
}

/* Per-message metadata footer with Enterprise Replica Attribution */
function MessageMetaFooter({ message }) {
  const usage = message.usage
  const ttft = formatMs(message.first_content_ms)
  const rate = formatRate(message.tokens_out_per_sec)
  const duration = formatMs(message.elapsed_ms)
  const usageLabel = message.usage_source === 'backend' ? 'usage' : 'usage est.'
  const attribution = message.attribution

  if (!usage && !ttft && !rate && !message.model_id && !attribution) return null
  return (
    <footer className="cxturn__meta" aria-label="Generation details and replica attribution">
      {attribution && <EnterpriseAttributionBadge attribution={attribution} />}
      {message.model_id && <span className="cxturn__meta-item cxturn__meta-model">{message.model_id}</span>}
      {usage && Number.isFinite(Number(usage.prompt_tokens)) && (
        <span className="cxturn__meta-item" title={message.usage_source === 'backend' ? 'Token counts reported by the backend' : 'Token counts estimated client-side (backend sent no usage)'}>
          {usageLabel} {usage.prompt_tokens}→{usage.completion_tokens}
        </span>
      )}
      {ttft && <span className="cxturn__meta-item" title="Time to first content, measured in this browser">TTFT {ttft}</span>}
      {rate && <span className="cxturn__meta-item" title="Decode rate, measured in this browser">{rate}</span>}
      {duration && <span className="cxturn__meta-item" title="Total request duration, measured in this browser">{duration}</span>}
      {/* No blanket "verified" stamp here. The badge above carries what the
          replica actually asserted about this turn — its lane and the digests of
          the configuration and weights that produced it. A fixed label claiming
          verification would say the same thing on a turn that carried no
          attribution at all. */}
    </footer>
  )
}

/* User rows: copy + inline edit-and-resend. Editing truncates the thread at
   this message and resends through the normal gate-checked send path. */
function UserTurn({ message, messageContent, onEditResend }) {
  const [editing, setEditing] = useState(false)
  const [draft, setDraft] = useState(messageContent)
  const submitEdit = () => {
    const next = draft.trim()
    setEditing(false)
    if (next && next !== messageContent) onEditResend?.(message.id, next)
  }
  return (
    <article className="cxturn cxturn--user">
      <div className="cxturn__user-chip">
        {editing ? (
          <div className="cxturn__edit">
            <textarea
              className="cxturn__edit-input"
              value={draft}
              rows={Math.min(8, Math.max(2, draft.split('\n').length))}
              onChange={(event) => setDraft(event.target.value)}
              onKeyDown={(event) => {
                if (event.key === 'Enter' && !event.shiftKey) {
                  event.preventDefault()
                  submitEdit()
                }
                if (event.key === 'Escape') {
                  event.stopPropagation()
                  setEditing(false)
                  setDraft(messageContent)
                }
              }}
              aria-label="Edit message and resend"
              autoFocus
            />
            <div className="cxturn__edit-actions">
              <button type="button" className="cxturn__action" onClick={submitEdit}>Resend</button>
              <button type="button" className="cxturn__action" onClick={() => { setEditing(false); setDraft(messageContent) }}>Cancel</button>
            </div>
          </div>
        ) : (
          <p>{messageContent}</p>
        )}
      </div>
      {!editing && onEditResend && (
        <div className="cxturn__actions cxturn__actions--user" aria-label="Message actions">
          <button type="button" className="cxturn__action" onClick={() => copyText(messageContent)} title="Copy message">
            <IconCopy size={14} /> <span>Copy</span>
          </button>
          <button type="button" className="cxturn__action" onClick={() => { setDraft(messageContent); setEditing(true) }} title="Edit this message and resend — replaces the replies after it">
            <IconEdit size={14} /> <span>Edit &amp; resend</span>
          </button>
        </div>
      )}
    </article>
  )
}

export const MessageTurn = memo(function MessageTurn({ message, generationElapsedSeconds, priorUserPrompt, onReusePrompt, onRegenerate, onEditResend }) {
  const [copied, setCopied] = useState(false)
  const copiedResetRef = useRef(null)
  const messageContent = cleanLegacyDemoCapCopy(message.content)
  const isUser = message.role === 'user'
  const assistantStreaming = message.role === 'assistant' && Boolean(message.streaming)
  const isOpenStreamingCode = assistantStreaming && hasOpenCodeFence(messageContent)
  const streamingPhase = message.streaming_phase || (messageContent ? 'streaming' : 'generating')
  const liveStatusLabel = streamingStatusLabel(streamingPhase, generationElapsedSeconds, isOpenStreamingCode)
  const showStreamingStatus = assistantStreaming && !messageContent
  const showLiveGenerationBadge = assistantStreaming && Boolean(messageContent)
  const showLengthWarning = message.role === 'assistant' && !assistantStreaming && message.finish_reason === 'length'
  const showErrorWarning = message.role === 'assistant' && !assistantStreaming && message.finish_reason === 'error'
  const showInterruptedWarning = message.role === 'assistant' && !assistantStreaming && message.finish_reason === 'interrupted'
  const showReusePromptAction = Boolean(priorUserPrompt) && (showErrorWarning || showInterruptedWarning)
  const showMessageActions = message.role === 'assistant' && Boolean(String(messageContent || '').trim())

  useEffect(() => () => {
    if (copiedResetRef.current) window.clearTimeout(copiedResetRef.current)
  }, [])

  const handleCopyMessage = async () => {
    await copyText(messageContent)
    setCopied(true)
    if (copiedResetRef.current) window.clearTimeout(copiedResetRef.current)
    copiedResetRef.current = window.setTimeout(() => setCopied(false), 1600)
  }

  if (isUser) {
    return (
      <UserTurn
        message={message}
        messageContent={messageContent}
        onEditResend={onEditResend}
      />
    )
  }

  return (
    <article
      className={`cxturn cxturn--assistant ${assistantStreaming ? 'is-streaming' : ''}`}
      aria-busy={assistantStreaming ? 'true' : undefined}
      data-streaming-state={assistantStreaming ? 'active' : undefined}
      data-streaming-code-state={isOpenStreamingCode ? 'open' : undefined}
    >
      <div className="cxturn__avatar">
        <Avatar
          size={30}
          state={assistantStreaming ? (messageContent ? 'streaming' : 'awaiting') : 'idle'}
          pulse={assistantStreaming ? String(messageContent || '').length : 0}
        />
      </div>
      <div className="cxturn__body">
        {showStreamingStatus && <StreamingLoader elapsedSeconds={generationElapsedSeconds} label={liveStatusLabel} compact />}
        {(messageContent || !assistantStreaming) && <AssistantMarkdown content={messageContent} streaming={assistantStreaming} />}
        {showLiveGenerationBadge && <LiveGenerationBadge elapsedSeconds={generationElapsedSeconds} label={liveStatusLabel} tokensPerSec={message.tokens_out_per_sec} />}

        {showLengthWarning && (
          <div className="cxturn__warning" role="status">Stopped before completing. Ask “continue” for a complete file.</div>
        )}
        {showErrorWarning && (
          <div className="cxturn__warning cxturn__warning--error" role="status">Generation stopped before Camelid returned a complete reply.</div>
        )}
        {showInterruptedWarning && (
          <div className="cxturn__warning cxturn__warning--interrupted" role="status">Generation was interrupted before the reply finished.</div>
        )}

        {(showMessageActions || showReusePromptAction) && (
          <div className="cxturn__actions" aria-label="Message actions">
            {showMessageActions && (
              <button type="button" className="cxturn__action" onClick={handleCopyMessage}>
                {copied ? <IconCheck size={16} /> : <IconCopy size={16} />}
                <span>{copied ? 'Copied' : 'Copy'}</span>
              </button>
            )}
            {showMessageActions && onRegenerate && (
              <button type="button" className="cxturn__action" onClick={() => onRegenerate()} title="Resend the prompt that produced this reply, with the same parameters">
                <IconRefresh size={16} /> <span>Regenerate</span>
              </button>
            )}
            {showReusePromptAction && (
              <button type="button" className="cxturn__action" onClick={() => onReusePrompt?.(priorUserPrompt)}>
                <IconRefresh size={16} /> <span>Use prompt again</span>
              </button>
            )}
          </div>
        )}

        {/* Rendered during streaming too: tokens_out_per_sec is live-patched per
           frame (backed token-for-token by window.__tpsTrace), so the footer
           doubles as the live tok/s readout while decoding. Absent fields stay
           hidden until the stream completes; the footer itself reserves the
           layout space the old placeholder div held. */}
        {message.role === 'assistant' && <MessageMetaFooter message={message} />}

      </div>
    </article>
  )
})

export default MessageTurn
