import { useLocale } from '@/contexts/LocaleContext'
import type { Locale } from '@/i18n/translations'
import { Moon, Sun } from 'lucide-react'
import { useTheme } from '@/contexts/ThemeContext'

export default function LocaleToggle() {
  const { locale, setLocale } = useLocale()
  const { theme, toggleTheme } = useTheme()

  return (
    <div className="flex items-center gap-2">
      <button
        onClick={toggleTheme}
        className="p-1.5 rounded-md hover:bg-[var(--color-surface-hover)] transition-colors"
        title={theme === 'dark' ? 'Light mode' : 'Dark mode'}
      >
        {theme === 'dark' ? <Sun size={16} /> : <Moon size={16} />}
      </button>
      <button
        onClick={() => setLocale(locale === 'en' ? 'zh' : ('en' as Locale))}
        className="px-2 py-1 text-xs rounded-md bg-[var(--color-surface)] border border-[var(--color-border)] hover:bg-[var(--color-surface-hover)] transition-colors"
      >
        {locale === 'en' ? '中文' : 'EN'}
      </button>
    </div>
  )
}
