import { memo, useEffect, useRef, useState } from 'react'
import { clampText } from '../../lib/formatters'
import { IconDots, IconEdit, IconTrash } from '../ui/icons'

function ConversationListItemInner({
  conversation,
  selected,
  collapsed,
  onSelect,
  onRename,
  onDelete,
}) {
  const [menuOpen, setMenuOpen] = useState(false)
  const [editing, setEditing] = useState(false)
  const [draftTitle, setDraftTitle] = useState('')
  const rootRef = useRef(null)

  const rawTitle = conversation.title || 'Untitled conversation'
  const title = clampText(rawTitle, 52) || 'Untitled conversation'

  useEffect(() => {
    if (!menuOpen) return undefined
    const onDocDown = (event) => {
      if (!rootRef.current?.contains(event.target)) setMenuOpen(false)
    }
    const onKey = (event) => { if (event.key === 'Escape') setMenuOpen(false) }
    window.addEventListener('pointerdown', onDocDown)
    window.addEventListener('keydown', onKey)
    return () => {
      window.removeEventListener('pointerdown', onDocDown)
      window.removeEventListener('keydown', onKey)
    }
  }, [menuOpen])

  const beginRename = () => {
    setMenuOpen(false)
    setDraftTitle(rawTitle)
    setEditing(true)
  }
  const commitRename = async () => {
    const ok = await onRename(conversation.id, draftTitle)
    if (ok !== false) setEditing(false)
  }

  if (editing) {
    return (
      <div className="rail-convo rail-convo--editing" ref={rootRef}>
        <input
          className="rail-convo__rename"
          value={draftTitle}
          autoFocus
          aria-label={`Rename ${rawTitle}`}
          onChange={(e) => setDraftTitle(e.target.value)}
          onBlur={commitRename}
          onKeyDown={(e) => {
            if (e.key === 'Enter') { e.preventDefault(); void commitRename() }
            if (e.key === 'Escape') { e.preventDefault(); setEditing(false) }
          }}
        />
      </div>
    )
  }

  return (
    <div className={`rail-convo ${selected ? 'is-selected' : ''} ${menuOpen ? 'has-menu' : ''}`} ref={rootRef}>
      <button
        type="button"
        className="rail-convo__main"
        aria-current={selected ? 'true' : undefined}
        title={rawTitle}
        onClick={() => onSelect(conversation.id)}
      >
        <span className="rail-convo__title">{collapsed ? rawTitle.slice(0, 1).toUpperCase() : title}</span>
      </button>
      {!collapsed && (
        <div className="rail-convo__actions">
          <button
            type="button"
            className="rail-convo__menu-btn"
            aria-label={`Options for ${title}`}
            aria-haspopup="menu"
            aria-expanded={menuOpen}
            onClick={(e) => { e.stopPropagation(); setMenuOpen((v) => !v) }}
          >
            <IconDots size={18} />
          </button>
          {menuOpen && (
            <div className="rail-menu" role="menu">
              <button type="button" role="menuitem" className="rail-menu__item" onClick={beginRename}>
                <IconEdit size={16} /> <span>Rename</span>
              </button>
              <button
                type="button"
                role="menuitem"
                className="rail-menu__item rail-menu__item--danger"
                onClick={() => { setMenuOpen(false); onDelete(conversation.id) }}
              >
                <IconTrash size={16} /> <span>Delete</span>
              </button>
            </div>
          )}
        </div>
      )}
    </div>
  )
}

export const ConversationListItem = memo(ConversationListItemInner)
export default ConversationListItem
