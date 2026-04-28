import { useEffect, useMemo, useRef, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { useLocale } from '@/contexts/LocaleContext'
import { listBenchmarks } from '@/api/eval'
import type { BenchmarkEntry } from '@/api/types'
import LoadingSpinner from '@/components/common/LoadingSpinner'
import SearchInput from '@/components/ui/SearchInput'
import Tabs from '@/components/ui/Tabs'
import Badge from '@/components/ui/Badge'
import Button from '@/components/ui/Button'
import { BookOpen, ArrowRight } from 'lucide-react'

type TabKey = 'text' | 'multimodal'

export default function BenchmarksPage() {
  const { t, locale } = useLocale()
  const navigate = useNavigate()
  const [tab, setTab] = useState<TabKey>('text')
  const [loading, setLoading] = useState(true)
  const [textBenchmarks, setTextBenchmarks] = useState<BenchmarkEntry[]>([])
  const [mmBenchmarks, setMmBenchmarks] = useState<BenchmarkEntry[]>([])
  const [search, setSearch] = useState('')
  const [debouncedSearch, setDebouncedSearch] = useState('')
  const timerRef = useRef<ReturnType<typeof setTimeout>>(undefined)

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

  // Debounce search
  useEffect(() => {
    timerRef.current = setTimeout(() => setDebouncedSearch(search), 300)
    return () => clearTimeout(timerRef.current)
  }, [search])

  const allItems = tab === 'text' ? textBenchmarks : mmBenchmarks

  const getDescription = (entry: BenchmarkEntry) => {
    const desc = locale === 'zh' ? entry.description?.zh : entry.description?.en
    return desc?.full ?? t('benchmarks.noDescription')
  }

  const items = useMemo(() => {
    if (!debouncedSearch) return allItems
    const q = debouncedSearch.toLowerCase()
    return allItems.filter(
      (e) =>
        e.name.toLowerCase().includes(q) ||
        getDescription(e).toLowerCase().includes(q),
    )
  }, [allItems, debouncedSearch, locale])

  const handleUse = (name: string) => {
    navigate(`/eval?dataset=${encodeURIComponent(name)}`)
  }

  const tabItems = [
    { key: 'text', label: `${t('benchmarks.text')} (${textBenchmarks.length})` },
    { key: 'multimodal', label: `${t('benchmarks.multimodal')} (${mmBenchmarks.length})` },
  ]

  if (loading) return <LoadingSpinner />

  return (
    <div className="page-enter space-y-5">
      <h1 className="text-xl font-semibold">{t('benchmarks.title')}</h1>

      {/* Controls row */}
      <div className="flex flex-col sm:flex-row items-start sm:items-center gap-3">
        <Tabs tabs={tabItems} activeKey={tab} onChange={(k) => setTab(k as TabKey)} />
        <SearchInput
          value={search}
          onChange={setSearch}
          placeholder={t('benchmarks.search')}
          className="sm:ml-auto w-full sm:w-72"
        />
      </div>

      {/* Cards grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
        {items.map((entry) => (
          <div
            key={entry.name}
            className="card-hover rounded-[var(--radius)] border border-[var(--border)] bg-[var(--bg-card)] p-5 flex flex-col"
          >
            <div className="flex items-start gap-3 flex-1">
              <BookOpen size={18} className="text-[var(--accent)] mt-0.5 shrink-0" />
              <div className="min-w-0 flex-1">
                <h3 className="text-sm font-semibold text-[var(--text)]">{entry.name}</h3>
                <p className="text-xs text-[var(--text-muted)] mt-1 line-clamp-3">
                  {getDescription(entry)}
                </p>
                {entry.metrics.length > 0 && (
                  <div className="flex flex-wrap gap-1 mt-2">
                    {entry.metrics.map((m) => (
                      <Badge key={m} variant="default" className="text-[10px]">
                        {m}
                      </Badge>
                    ))}
                  </div>
                )}
              </div>
            </div>
            <div className="mt-4 pt-3 border-t border-[var(--border)]">
              <Button
                variant="ghost"
                size="sm"
                onClick={() => handleUse(entry.name)}
                className="w-full justify-center text-[var(--accent)] hover:text-white hover:bg-[var(--accent)]"
              >
                {t('benchmarks.use')} <ArrowRight size={14} />
              </Button>
            </div>
          </div>
        ))}
        {items.length === 0 && (
          <p className="text-sm text-[var(--text-muted)] col-span-full text-center py-8">
            {t('common.noData')}
          </p>
        )}
      </div>
    </div>
  )
}
