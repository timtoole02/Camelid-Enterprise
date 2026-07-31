import { useCallback, useEffect, useState } from 'react'

const STORAGE_KEY = 'camelid-theme'
const VALID = new Set(['system', 'light', 'dark'])
const ORDER = ['dark', 'light', 'system']

/* Dark is the design's canonical palette, so it is the default preference;
   'system' and 'light' remain one toggle away. */
function readPreference() {
  if (typeof window === 'undefined') return 'dark'
  const saved = window.localStorage.getItem(STORAGE_KEY)
  return saved && VALID.has(saved) ? saved : 'dark'
}

function systemPrefersDark() {
  if (typeof window === 'undefined' || !window.matchMedia) return false
  return window.matchMedia('(prefers-color-scheme: dark)').matches
}

function applyPreference(preference) {
  if (typeof document === 'undefined') return
  const root = document.documentElement
  if (preference === 'system') {
    // Remove the attribute so the prefers-color-scheme media query drives the palette.
    delete root.dataset.theme
  } else {
    root.dataset.theme = preference
  }
}

/**
 * Dual light/dark theme, system-following.
 *  - preference ∈ { 'system', 'light', 'dark' } (persisted)
 *  - 'system' removes [data-theme] so CSS prefers-color-scheme wins, and a live
 *    matchMedia listener keeps `resolved` in sync for the toggle UI.
 *  - 'light' / 'dark' set [data-theme] explicitly.
 */
export function useTheme() {
  const [preference, setPreferenceState] = useState(readPreference)
  const [resolved, setResolved] = useState(() =>
    preference === 'system' ? (systemPrefersDark() ? 'dark' : 'light') : preference,
  )

  useEffect(() => {
    applyPreference(preference)
    if (typeof window !== 'undefined') {
      window.localStorage.setItem(STORAGE_KEY, preference)
    }
    if (preference !== 'system') {
      setResolved(preference)
      return undefined
    }
    if (typeof window === 'undefined' || !window.matchMedia) {
      setResolved('light')
      return undefined
    }
    const media = window.matchMedia('(prefers-color-scheme: dark)')
    const sync = () => setResolved(media.matches ? 'dark' : 'light')
    sync()
    media.addEventListener('change', sync)
    return () => media.removeEventListener('change', sync)
  }, [preference])

  const setPreference = useCallback((next) => {
    setPreferenceState(VALID.has(next) ? next : 'system')
  }, [])

  const cyclePreference = useCallback(() => {
    setPreferenceState((current) => {
      const index = ORDER.indexOf(current)
      return ORDER[(index + 1) % ORDER.length]
    })
  }, [])

  return { preference, setPreference, cyclePreference, resolved }
}
