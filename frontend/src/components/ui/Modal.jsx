import { useEffect, useRef } from 'react'
import { createPortal } from 'react-dom'
import { IconClose } from './icons'
import { IconButton } from './IconButton'

/* Modal — accessible dialog with backdrop, Esc-to-close, scroll lock, focus capture. */
export function Modal({ open, onClose, title, children, footer = null, labelledById, size = 'md', className = '' }) {
  const panelRef = useRef(null)

  useEffect(() => {
    if (!open) return undefined
    const previouslyFocused = typeof document !== 'undefined' ? document.activeElement : null
    const onKey = (event) => {
      if (event.key === 'Escape') {
        event.stopPropagation()
        onClose?.()
      }
    }
    window.addEventListener('keydown', onKey)
    const { body } = document
    const prevOverflow = body.style.overflow
    body.style.overflow = 'hidden'
    const frame = window.requestAnimationFrame(() => panelRef.current?.focus())
    return () => {
      window.removeEventListener('keydown', onKey)
      body.style.overflow = prevOverflow
      window.cancelAnimationFrame(frame)
      if (previouslyFocused && typeof previouslyFocused.focus === 'function') previouslyFocused.focus()
    }
  }, [open, onClose])

  if (!open || typeof document === 'undefined') return null

  return createPortal(
    <div className="cx-modal-overlay" onMouseDown={(e) => { if (e.target === e.currentTarget) onClose?.() }}>
      <div
        ref={panelRef}
        className={`cx-modal cx-modal--${size} ${className}`.trim()}
        role="dialog"
        aria-modal="true"
        aria-labelledby={labelledById}
        tabIndex={-1}
      >
        {title && (
          <header className="cx-modal__header">
            <h2 id={labelledById} className="cx-modal__title">{title}</h2>
            <IconButton label="Close" size="sm" onClick={onClose}><IconClose size={18} /></IconButton>
          </header>
        )}
        <div className="cx-modal__body">{children}</div>
        {footer && <footer className="cx-modal__footer">{footer}</footer>}
      </div>
    </div>,
    document.body,
  )
}

export default Modal
