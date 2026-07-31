/* Tooltip — lightweight CSS hover/focus tooltip. Wraps a single trigger.
   placement: top | bottom | right | left */
export function Tooltip({ content, placement = 'top', className = '', children }) {
  if (!content) return children
  return (
    <span className={`cx-tooltip cx-tooltip--${placement} ${className}`.trim()}>
      {children}
      <span role="tooltip" className="cx-tooltip__bubble">{content}</span>
    </span>
  )
}

export default Tooltip
