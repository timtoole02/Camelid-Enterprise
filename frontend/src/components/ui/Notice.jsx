import { IconCheckCircle, IconClose, IconError, IconInfo, IconWarning } from './icons'

const ICONS = { success: IconCheckCircle, error: IconError, warning: IconWarning, info: IconInfo }

/* Notice — transient toast. Replaces GlobalNotice. tone: success | error | warning | info */
export function Notice({ notice, tone = 'info', onDismiss = null }) {
  if (!notice) return null
  const Icon = ICONS[tone] || IconInfo
  const role = tone === 'error' ? 'alert' : 'status'
  return (
    <div className={`cx-notice cx-notice--${tone}`} role={role} aria-live={tone === 'error' ? 'assertive' : 'polite'}>
      <span className="cx-notice__icon"><Icon size={18} /></span>
      <span className="cx-notice__text">{notice}</span>
      {onDismiss && (
        <button type="button" className="cx-notice__close" aria-label="Dismiss" onClick={onDismiss}>
          <IconClose size={16} />
        </button>
      )}
    </div>
  )
}

export default Notice
