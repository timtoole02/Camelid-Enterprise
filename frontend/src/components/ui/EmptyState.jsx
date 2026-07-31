/* EmptyState — centered icon + title + copy + optional action. */
export function EmptyState({ icon = null, title, description = '', action = null, className = '' }) {
  return (
    <div className={`cx-empty ${className}`.trim()}>
      {icon && <div className="cx-empty__icon">{icon}</div>}
      {title && <h3 className="cx-empty__title">{title}</h3>}
      {description && <p className="cx-empty__desc">{description}</p>}
      {action && <div className="cx-empty__action">{action}</div>}
    </div>
  )
}

export default EmptyState
