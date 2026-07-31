/* Card — elevated surface. Optional title/eyebrow/actions header. tone: default | accent | muted */
export function Card({ tone = 'default', interactive = false, className = '', children, as: Tag = 'section', ...rest }) {
  const classes = [
    'cx-card',
    `cx-card--${tone}`,
    interactive ? 'cx-card--interactive' : '',
    className,
  ].filter(Boolean).join(' ')
  return <Tag className={classes} {...rest}>{children}</Tag>
}

export function CardHeader({ eyebrow, title, icon = null, actions = null, className = '' }) {
  return (
    <header className={`cx-card__header ${className}`.trim()}>
      {icon && <span className="cx-card__header-icon">{icon}</span>}
      <div className="cx-card__header-copy">
        {eyebrow && <span className="cx-card__eyebrow">{eyebrow}</span>}
        {title && <h3 className="cx-card__title">{title}</h3>}
      </div>
      {actions && <div className="cx-card__header-actions">{actions}</div>}
    </header>
  )
}

export function CardBody({ className = '', children }) {
  return <div className={`cx-card__body ${className}`.trim()}>{children}</div>
}

export default Card
