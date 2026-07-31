/* The Camelid mark: an original, instrument-grade llama glyph —
   upright neck, forward muzzle, the signature splayed ear pair — drawn
   stroke-only on a 24px grid so it stays legible from favicon to wordmark.
   Colors bind to currentColor (neutral ink by default); copper is never spent
   on decoration.

   It is also the chat streaming indicator. States:
   - idle:      static
   - awaiting:  slow breathing (working, not frozen)
   - streaming: ears flick per rAF-coalesced token batch (`pulse` prop) — the
                rhythm IS the real generation cadence
   - settle:    motion stops with one restrained transition (error/abort)
   All motion is CSS transform/opacity on SVG sub-elements; reduced-motion
   renders every state static (state is also conveyed by text affordances). */

export function CamelidMark({ size = 24, state = 'idle', pulse = 0, className = '', title }) {
  return (
    <svg
      className={`camelid-mark ${className}`.trim()}
      data-state={state}
      data-step={state === 'streaming' ? pulse % 2 : undefined}
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2.1"
      strokeLinecap="round"
      strokeLinejoin="round"
      role={title ? 'img' : undefined}
      aria-hidden={title ? undefined : 'true'}
      xmlns="http://www.w3.org/2000/svg"
    >
      {title ? <title>{title}</title> : null}
      <path className="camelid-mark__body" d="M7.4 21V11.2Q7.4 8.7 9.8 8.7H14.2V11" />
      <path className="camelid-mark__ear camelid-mark__ear--l" d="M9.6 8.4L8.9 4.2" />
      <path className="camelid-mark__ear camelid-mark__ear--r" d="M12.6 8.4L13.5 4.2" />
    </svg>
  )
}

export default CamelidMark
