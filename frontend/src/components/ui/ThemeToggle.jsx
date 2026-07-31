import { IconMonitor, IconMoon, IconSun } from './icons'

const NEXT_LABEL = { system: 'light', light: 'dark', dark: 'system' }

/* Cycles theme preference system → light → dark. Shows the active preference. */
export function ThemeToggle({ preference, resolved, onCycle, compact = false }) {
  const Icon = preference === 'system' ? IconMonitor : preference === 'light' ? IconSun : IconMoon
  const labelFor = { system: 'System', light: 'Light', dark: 'Dark' }
  const aria = `Theme: ${labelFor[preference]} (${resolved}). Switch to ${labelFor[NEXT_LABEL[preference]]}.`
  return (
    <button
      type="button"
      className={`cx-theme-toggle ${compact ? 'is-compact' : ''}`.trim()}
      onClick={onCycle}
      aria-label={aria}
      title={aria}
    >
      <span className="cx-theme-toggle__icon"><Icon size={18} /></span>
      {!compact && <span className="cx-theme-toggle__label">{labelFor[preference]}</span>}
    </button>
  )
}

export default ThemeToggle
