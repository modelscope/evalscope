import { useEffect, useState } from 'react'
import { useLocale } from '@/contexts/LocaleContext'
import { listBenchmarks } from '@/api/eval'
import type { BenchmarkEntry } from '@/api/types'
import LoadingSpinner from '@/components/common/LoadingSpinner'
import { BookOpen } from 'lucide-react'

type Tab = 'text' | 'multimodal'

export default function BenchmarksPage() {
  const { t, locale } = useLocale()
  const [tab, setTab] = useState<Tab>('text')
  const [loading, setLoading] = useState(true)
  const [textBenchmarks, setTextBenchmarks] = useState<BenchmarkEntry[]>([])
  const [mmBenchmarks, setMmBenchmarks] = useState<BenchmarkEntry[]>([])

  useEffect(() => {
    setLoading(true)
    listBenchmarks()
      .then((res) => {
        setTextBenchmarks(res.text ?? [])
        setMmBenchmarks(res.multimodal ?? [])
      })
      .catch(() => {})
      .finally(() => setLoading(false))
  }, [])

  const items = tab === 'text' ? textBenchmarks : mmBenchmarks

  const getDescription = (entry: BenchmarkEntry) => {
    const desc = locale === 'zh' ? entry.description?.zh : entry.description?.en
    return desc?.full ?? t('benchmarks.noDescription')
  }

  if (loading) return <LoadingSpinner />

  return (
    <div className="space-y-4 max-w-5xl">
      <h1 className="text-xl font-semibold">{t('benchmarks.title')}</h1>

      {/* Tabs */}
      <div className="flex gap-1 border-b border-[var(--color-border)]">
        <button
          onClick={() => setTab('text')}
          className={`px-4 py-2 text-sm font-medium border-b-2 transition-colors ${
            tab === 'text'
              ? 'border-[var(--color-primary)] text-[var(--color-primary)]'
              : 'border-transparent text-[var(--color-ink-muted)] hover:text-[var(--color-ink)]'
          }`}
        >
          {t('benchmarks.text')} ({textBenchmarks.length})
        </button>
        <button
          onClick={() => setTab('multimodal')}
          className={`px-4 py-2 text-sm font-medium border-b-2 transition-colors ${
            tab === 'multimodal'
              ? 'border-[var(--color-primary)] text-[var(--color-primary)]'
              : 'border-transparent text-[var(--color-ink-muted)] hover:text-[var(--color-ink)]'
          }`}
        >
          {t('benchmarks.multimodal')} ({mmBenchmarks.length})
        </button>
      </div>

      {/* Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
        {items.map((entry) => (
          <div
            key={entry.name}
            className="rounded-lg border border-[var(--color-border)] bg-[var(--color-surface)] p-4 hover:border-[var(--color-primary)] transition-colors"
          >
            <div className="flex items-start gap-3">
              <BookOpen size={18} className="text-[var(--color-primary)] mt-0.5 shrink-0" />
              <div className="min-w-0">
                <h3 className="text-sm font-semibold">{entry.name}</h3>
                <p className="text-xs text-[var(--color-ink-muted)] mt-1 line-clamp-3">
                  {getDescription(entry)}
                </p>
                {entry.metrics.length > 0 && (
                  <div className="flex flex-wrap gap-1 mt-2">
                    {entry.metrics.map((m) => (
                      <span
                        key={m}
                        className="text-[10px] px-1.5 py-0.5 rounded bg-[var(--color-surface-hover)] text-[var(--color-ink-muted)]"
                      >
                        {m}
                      </span>
                    ))}
                  </div>
                )}
              </div>
            </div>
          </div>
        ))}
        {items.length === 0 && (
          <p className="text-sm text-[var(--color-ink-muted)] col-span-2 text-center py-8">
            {t('common.noData')}
          </p>
        )}
      </div>
    </div>
  )
}
