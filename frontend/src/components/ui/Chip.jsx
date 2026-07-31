/* Chip — compact status/label pill. tone: neutral | accent | ready | warn | error | info */
export function Chip({ tone = 'neutral', dot = false, icon = null, className = '', children, ...rest }) {
  const classes = ['cx-chip', `cx-chip--${tone}`, className].filter(Boolean).join(' ')
  return (
    <span className={classes} {...rest}>
      {dot && <span className="cx-chip__dot" aria-hidden="true" />}
      {icon && <span className="cx-chip__icon">{icon}</span>}
      {children}
    </span>
  )
}

export default Chip
