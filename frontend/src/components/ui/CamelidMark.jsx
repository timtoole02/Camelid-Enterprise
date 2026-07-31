/* The Camelid mark: the app icon's camelid head, as a filled glyph on a 24px
   grid, so one silhouette carries from the favicon up to the wordmark.

   The geometry is traced from the shipped app icon rather than redrawn by eye —
   the raster is the only master that exists, so a hand-approximation would have
   been a second, slightly-different mark rather than the same one. Contours were
   extracted from the 512px source at a luminance threshold, simplified, and
   normalized into this viewBox; the eye is emitted as a circle from the hole's
   own centroid and area, because at 16px a traced 12-point contour and a circle
   are the same handful of pixels and the circle stays round under scaling.

   It is split into three paths for one reason: the mark doubles as the chat
   streaming indicator, and the ears are what moves. A single merged path would
   have been fewer bytes and no longer animatable.
   - idle:      static
   - awaiting:  slow breathing (working, not frozen)
   - streaming: ears flick per rAF-coalesced token batch (`pulse` prop) — the
                rhythm IS the real generation cadence
   - settle:    motion stops with one restrained transition (error/abort)
   The ear paths overlap the body slightly so the seam never opens during a
   flick. All motion is CSS transform/opacity on SVG sub-elements; reduced-motion
   renders every state static (state is also conveyed by text affordances).

   Fill binds to currentColor, so the glyph inherits whatever ink its context
   uses and stays correct in both themes. `eyeFill` exists because the eye is a
   hole in the source: it has to be painted in the surface behind the mark, and
   only the caller knows what that is. It defaults to the page background. */

export function CamelidMark({
  size = 24,
  state = 'idle',
  pulse = 0,
  className = '',
  title,
  eyeFill = 'var(--camelid-mark-eye, var(--color-bg, #0e1216))',
}) {
  return (
    <svg
      className={`camelid-mark ${className}`.trim()}
      data-state={state}
      data-step={state === 'streaming' ? pulse % 2 : undefined}
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="currentColor"
      role={title ? 'img' : undefined}
      aria-hidden={title ? undefined : 'true'}
      xmlns="http://www.w3.org/2000/svg"
    >
      {title ? <title>{title}</title> : null}
      <path
        className="camelid-mark__body"
        d="M10.17,6.69L11.43,6.69L11.54,7.14L12.29,6.69L13.54,6.69L13.54,6.91L14.91,7.37L16.23,8.17L16.91,9.03L17.03,9.37L17.03,10.06L16.74,10.63L16.17,11.03L15.26,11.09L14.91,10.91L13.77,11.37L13.26,11.71L12.40,12.69L12.11,13.54L11.94,14.91L11.83,22.00L6.97,22.00L6.97,16.40L7.14,13.89L7.37,12.46L7.77,11.09L8.40,9.77L9.14,8.74L10.40,7.54L10.23,6.74Z"
      />
      <path
        className="camelid-mark__ear camelid-mark__ear--l"
        d="M10.06,2.40L10.86,4.57L11.49,6.91L11.49,7.09L10.29,7.09L10.23,6.97L10.00,5.77L9.89,4.63L9.89,3.54L10.06,2.46Z"
      />
      <path
        className="camelid-mark__ear camelid-mark__ear--r"
        d="M13.09,2.00L13.31,2.74L13.54,4.29L13.60,5.66L13.54,5.71L13.49,6.86L14.17,7.09L11.66,7.09L12.29,6.69L12.29,5.66L12.51,3.89L12.69,3.14L13.09,2.06Z"
      />
      <circle className="camelid-mark__eye" cx="13" cy="9.53" r="0.57" fill={eyeFill} />
    </svg>
  )
}

export default CamelidMark
