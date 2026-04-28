import { useLocale } from '@/contexts/LocaleContext'
import type { Locale } from '@/i18n/translations'

export default function LocaleToggle() {
  const { locale, setLocale } = useLocale()

  return (
    <button
      onClick={() => setLocale(locale === 'en' ? 'zh' : ('en' as Locale))}
      className="px-2 py-1 text-xs rounded-md bg-[var(--bg-card)] border border-[var(--border)] hover:bg-[var(--bg-card2)] text-[var(--text-muted)] hover:text-[var(--text)] transition-colors cursor-pointer"
    >
      {locale === 'en' ? '中文' : 'EN'}
    </button>
  )
}
