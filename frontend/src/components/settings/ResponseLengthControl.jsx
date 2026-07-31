import { useMemo, useState } from 'react'
import {
  DETENTS,
  MAX_RESPONSE_TOKENS,
  MIN_RESPONSE_TOKENS,
  modelContextLength,
  sliderToTokens,
  tokensToSlider,
  validateResponseLength,
  
} from '../../lib/responseLimits'

/* Response-length slider + numeric input. Log-scale track with
   detents; threshold markers render ONLY from real data: the verified bound
   (Evidence-Chip treatment — the one marker allowed evidence styling), the
   model context length (metadata, explicitly not a support claim), and a
   memory ceiling that is ABSENT until the backend reports system memory and
   KV cost — no client-side guessing, no fake gauge. */

const fmt = (n) => n.toLocaleString()

export function ResponseLengthControl({ value, onChange, model = null }) {
  /* While dragging, the thumb renders from RAW local position — a controlled
     rewrite that snaps the thumb under the pointer breaks the drag in some
     engines (the thumb stuck at the 256k detent and 1M was unreachable).
     Tokens still update live; the snap applies to the committed value only. */
  const [dragPos, setDragPos] = useState(null)
  const contextLength = modelContextLength(model)
  const verdict = validateResponseLength({ value, contextLength, modelName: model?.name || 'the loaded model' })

  const setValue = (next) => {
    const clamped = Math.min(Math.max(Math.round(next), MIN_RESPONSE_TOKENS), MAX_RESPONSE_TOKENS)
    if (Number.isFinite(clamped)) onChange(clamped)
  }

  return (
    <div className={`rlc rlc--${verdict.level}`} data-validation={verdict.level}>
      <div className="rlc__row">
        <div className="rlc__track-wrap">
          <input
            type="range"
            className="rlc__slider"
            min="0"
            max="1000"
            value={dragPos ?? Math.round(tokensToSlider(value) * 1000)}
            onChange={(event) => {
              const raw = Number(event.target.value)
              setDragPos(raw)
              setValue(sliderToTokens(raw / 1000))
            }}
            onPointerUp={() => setDragPos(null)}
            onTouchEnd={() => setDragPos(null)}
            onKeyUp={() => setDragPos(null)}
            onBlur={() => setDragPos(null)}
            aria-label="Response length in tokens (logarithmic scale)"
            aria-invalid={verdict.level === 'error'}
          />
          <div className="rlc__detents" aria-hidden="true">
            {DETENTS.map((detent) => (
              <span key={detent} className="rlc__detent" style={{ left: `${tokensToSlider(detent) * 100}%` }}>
                <i />{detent >= 1000000 ? '1M' : detent >= 1000 ? `${detent / 1000}k` : detent}
              </span>
            ))}
          </div>
          <div className="rlc__markers" aria-hidden="true">
            {contextLength !== null && (
              <span
                className={`rlc__marker rlc__marker--context ${value > contextLength ? 'is-violated' : ''}`}
                data-edge={tokensToSlider(contextLength) > 0.6 ? 'right' : undefined}
                style={{ left: `${tokensToSlider(contextLength) * 100}%` }}
              >
                <span className="rlc__marker-label">model max {fmt(contextLength)} · from model metadata, not a support claim</span>
              </span>
            )}
          </div>
        </div>
        <input
          type="number"
          className="rlc__number"
          min={MIN_RESPONSE_TOKENS}
          max={MAX_RESPONSE_TOKENS}
          value={value}
          aria-label="Response length in tokens"
          aria-invalid={verdict.level === 'error'}
          onChange={(event) => setValue(Number(event.target.value) || MIN_RESPONSE_TOKENS)}
          onKeyDown={(event) => {
            if (event.key === 'ArrowUp' || event.key === 'ArrowDown') {
              event.preventDefault()
              const direction = event.key === 'ArrowUp' ? 1 : -1
              setValue(value + direction * (event.shiftKey ? 10 : 1))
            }
          }}
        />
      </div>

      {verdict.level !== 'ok' && (
        <p className={`rlc__message rlc__message--${verdict.level}`} role="status">
          <span className="rlc__message-icon" aria-hidden="true">{verdict.level === 'error' ? '✕' : '◷'}</span>
          {verdict.message}
        </p>
      )}
      {contextLength === null && (
        <p className="rlc__absent">model context length unavailable — no loaded-model metadata to validate against</p>
      )}

      {/* Memory estimate: ABSENT until the backend reports the inputs. When it
          does, the readout renders here labeled "estimated" with its formula in
          the popover; it never renders on invented numbers. */}
      <p className="rlc__absent">memory estimate unavailable — the service does not report system memory or KV-cache cost per token, and this is deliberately not estimated client-side</p>
    </div>
  )
}

export default ResponseLengthControl
