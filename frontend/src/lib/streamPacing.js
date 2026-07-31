/* Display pacing: smooths bursty token arrival into a steady visual
   cadence under hard honesty bounds — the displayed text may never lag the
   truly received stream by more than MAX_LAG_MS, drains instantly on stream
   end/abort, and the final rendered text is byte-identical to the received
   text. Metrics (TTFT/tok-s tiles, telemetry, Flow Bench) always use real
   arrival data, never the paced view (I4). Pure functions; smoke-tested. */

export const MAX_LAG_MS = 150

export function createPacerState() {
  return { shownChars: 0, arrivals: [] }
}

/* Record what has truly arrived; returns the text that may be shown at `now`:
   at least everything that arrived MAX_LAG_MS ago (the lag bound), advanced
   smoothly toward the freshest text. */
export function paceStep(state, receivedText, nowMs) {
  const received = receivedText.length
  const last = state.arrivals[state.arrivals.length - 1]
  if (!last || last.chars < received) state.arrivals.push({ chars: received, at: nowMs })
  while (state.arrivals.length > 2 && state.arrivals[1].at <= nowMs - MAX_LAG_MS) state.arrivals.shift()
  // lag bound: everything that arrived at or before now - MAX_LAG_MS must show
  let mustShow = 0
  for (const arrival of state.arrivals) {
    if (arrival.at <= nowMs - MAX_LAG_MS) mustShow = arrival.chars
  }
  // smooth advance: close 60% of the remaining gap per step (min 8 chars) so
  // the tail converges well inside the lag bound
  const gap = received - state.shownChars
  const eased = state.shownChars + Math.max(8, Math.ceil(gap * 0.6))
  state.shownChars = Math.min(received, Math.max(mustShow, eased, state.shownChars))
  return receivedText.slice(0, state.shownChars)
}

/* Stream ended or aborted: drain instantly, byte-identical. */
export function paceDrain(state, receivedText) {
  state.shownChars = receivedText.length
  state.arrivals = []
  return receivedText
}
