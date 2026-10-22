import { afterEach, describe, expect, it, vi } from 'vitest'
import { fireEvent, render, screen } from '@testing-library/react'
import { LocaleProvider, useLocale } from './LocaleContext'
import { ThemeProvider, useTheme } from './ThemeContext'

function Preferences() {
  const { locale, setLocale } = useLocale()
  const { theme, toggleTheme } = useTheme()
  return <>
    <button onClick={() => setLocale(locale === 'en' ? 'zh' : 'en')}>{locale}</button>
    <button onClick={toggleTheme}>{theme}</button>
  </>
}

afterEach(() => { vi.restoreAllMocks() })

describe('preferences without browser persistence', () => {
  it.each(['denied', 'quota'] as const)('keeps rendering and switching preferences with %s storage', (failure) => {
    if (failure === 'denied') {
      vi.spyOn(window, 'localStorage', 'get').mockImplementation(() => {
        throw new DOMException('Blocked', 'SecurityError')
      })
    } else {
      vi.spyOn(Storage.prototype, 'setItem').mockImplementation(() => {
        throw new DOMException('Full', 'QuotaExceededError')
      })
    }
    render(<ThemeProvider><LocaleProvider><Preferences /></LocaleProvider></ThemeProvider>)
    fireEvent.click(screen.getByRole('button', { name: 'en' }))
    expect(screen.getByRole('button', { name: 'zh' })).toBeInTheDocument()
    const initial = document.documentElement.getAttribute('data-theme')
    fireEvent.click(screen.getByRole('button', { name: initial! }))
    expect(document.documentElement.getAttribute('data-theme')).toBe(initial === 'dark' ? 'light' : 'dark')
  })
})
