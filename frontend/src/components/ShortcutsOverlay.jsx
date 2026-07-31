/* Keyboard shortcut map. Opened with “?” outside text inputs. */

const SHORTCUTS = [
  { keys: '⌘/Ctrl + K', what: 'Command palette — navigate, switch model, jump to a ledger row' },
  { keys: 'Enter', what: 'Send the drafted message (chat composer)' },
  { keys: 'Shift + Enter', what: 'New line in the composer' },
  { keys: 'Esc', what: 'Stop a running generation · close palette/overlays · cancel an inline edit' },
  { keys: '?', what: 'This shortcut map (outside text fields)' },
]

export function ShortcutsOverlay({ open, onClose }) {
  if (!open) return null
  return (
    <div className="palette-overlay" onMouseDown={(event) => { if (event.target === event.currentTarget) onClose() }}>
      <div className="shortcuts" role="dialog" aria-modal="true" aria-label="Keyboard shortcuts">
        <header className="shortcuts__head">
          <h2>Keyboard shortcuts</h2>
          <button type="button" className="cxturn__action" onClick={onClose} autoFocus>Close</button>
        </header>
        <dl className="shortcuts__list">
          {SHORTCUTS.map((shortcut) => (
            <div key={shortcut.keys} className="shortcuts__row">
              <dt><kbd>{shortcut.keys}</kbd></dt>
              <dd>{shortcut.what}</dd>
            </div>
          ))}
        </dl>
      </div>
    </div>
  )
}

export default ShortcutsOverlay
