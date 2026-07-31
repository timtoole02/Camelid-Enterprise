/* StatusDot — small state indicator. tone: ready | warn | error | info | neutral | offline */
export function StatusDot({ tone = 'neutral', pulse = false, label = null, className = '' }) {
  const classes = ['cx-statusdot', `cx-statusdot--${tone}`, pulse ? 'is-pulsing' : '', className]
    .filter(Boolean).join(' ')
  if (label) {
    return (
      <span className="cx-statusdot-wrap">
        <span className={classes} aria-hidden="true" />
        <span className="cx-statusdot-label">{label}</span>
      </span>
    )
  }
  return <span className={classes} aria-hidden="true" />
}

export default StatusDot
