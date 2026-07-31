/* Streaming status indicators — extracted verbatim from ChatWorkspace.
   The class names (streaming-loader-track, streaming-loader-dot-N,
   message-live-generation-badge) are asserted by the CI smokes. */

export const PREPARING_STREAMING_LABEL = 'Preparing local response'
export const FIRST_TOKEN_STREAMING_LABEL = 'Backend is generating'
export const LONG_FIRST_TOKEN_STREAMING_LABEL = 'Local response is taking a while'
export const ACTIVE_STREAMING_LABEL = 'Streaming response'
export const OPEN_CODE_STREAMING_LABEL = 'Streaming code response'

export const streamingStatusLabel = (phase, elapsedSeconds, isOpenCode = false) => {
  if (phase === 'preparing') return PREPARING_STREAMING_LABEL
  if (phase === 'streaming') return isOpenCode ? OPEN_CODE_STREAMING_LABEL : ACTIVE_STREAMING_LABEL
  if (elapsedSeconds >= 20) return LONG_FIRST_TOKEN_STREAMING_LABEL
  return FIRST_TOKEN_STREAMING_LABEL
}

export function StreamingLoader({ elapsedSeconds, label = ACTIVE_STREAMING_LABEL, compact = false }) {
  return (
    <div className={`streaming-loader ${compact ? 'streaming-loader-compact' : ''}`} role="status" aria-live="polite" aria-label={`${label}. ${elapsedSeconds} seconds elapsed.`}>
      <div className="streaming-loader-track" aria-hidden="true">
        <span className="streaming-loader-dot streaming-loader-dot-1" />
        <span className="streaming-loader-dot streaming-loader-dot-2" />
        <span className="streaming-loader-dot streaming-loader-dot-3" />
      </div>
    </div>
  )
}

export function LiveGenerationBadge({ elapsedSeconds, label = ACTIVE_STREAMING_LABEL }) {
  return (
    <div className="message-live-generation-badge" role="status" aria-live="polite" data-live-status="active">
      <span className="message-live-dot" aria-hidden="true" />
      <span>{label}</span>
      <span>{elapsedSeconds}s</span>
    </div>
  )
}
