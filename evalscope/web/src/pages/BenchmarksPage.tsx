import { useEffect, useMemo, useRef, useState, useCallback } from 'react'
import { createPortal } from 'react-dom'
import { useLocale } from '@/contexts/LocaleContext'
import { listBenchmarks } from '@/api/eval'
import type { BenchmarkEntry } from '@/api/types'
import LoadingSpinner from '@/components/common/LoadingSpinner'
import MarkdownRenderer from '@/components/common/MarkdownRenderer'
import SearchInput from '@/components/ui/SearchInput'
import Tabs from '@/components/ui/Tabs'
import Badge from '@/components/ui/Badge'
import { BookOpen, X, Database, Layers, FlaskConical, Tag, ExternalLink } from 'lucide-react'

type TabKey = 'all' | 'text' | 'multimodal'

/** Strip markdown formatting for a plain-text preview */
function stripMarkdown(md: string): string {
  return md
    .replace(/^#{1,6}\s+/gm, '')   // headings
    .replace(/\*\*(.+?)\*\*/g, '$1')  // bold
    .replace(/\*(.+?)\*/g, '$1')      // italic
    .replace(/`([^`]+)`/g, '$1')     // inline code
    .replace(/\[([^\]]+)\]\([^)]+\)/g, '$1') // links
    .replace(/^[-*]\s+/gm, '')       // unordered list markers
    .replace(/^\d+\.\s+/gm, '')      // ordered list markers
    .trim()
}

export default function BenchmarksPage() {
  const { t, locale } = useLocale()
  const [tab, setTab] = useState<TabKey>('all')
  const [loading, setLoading] = useState(true)
  const [allBenchmarks, setAllBenchmarks] = useState<BenchmarkEntry[]>([])
  const [search, setSearch] = useState('')
  const [debouncedSearch, setDebouncedSearch] = useState('')
  const [selectedTags, setSelectedTags] = useState<string[]>([])
  const [selectedEntry, setSelectedEntry] = useState<BenchmarkEntry | null>(null)
  const timerRef = useRef<ReturnType<typeof setTimeout>>(undefined)

  // Normalise a raw API entry so all new fields have safe defaults
  // inferCategory: when the backend hasn't been restarted (no `category` field),
  // infer from which list the item came from.
  const normalize = (e: BenchmarkEntry, inferCategory?: 'llm' | 'vlm'): BenchmarkEntry => ({
    ...e,
    pretty_name: e.pretty_name ?? e.meta?.pretty_name ?? e.name,
    tags: Array.isArray(e.tags) ? e.tags : Array.isArray((e.meta as Record<string, unknown>)?.tags) ? (e.meta as Record<string, unknown>).tags as string[] : [],
    category: e.category ?? ((e.meta as Record<string, unknown>)?.category as string | undefined) ?? inferCategory ?? 'llm',
    subset_list: Array.isArray(e.subset_list) ? e.subset_list : Array.isArray((e.meta as Record<string, unknown>)?.subset_list) ? (e.meta as Record<string, unknown>).subset_list as string[] : [],
    total_samples: e.total_samples ?? (e.meta as Record<string, unknown>)?.total_samples as number ?? 0,
    few_shot_num: e.few_shot_num ?? (e.meta as Record<string, unknown>)?.few_shot_num as number ?? 0,
    dataset_id: e.dataset_id ?? (e.meta as Record<string, unknown>)?.dataset_id as string ?? '',
    paper_url: e.paper_url ?? (e.meta as Record<string, unknown>)?.paper_url as string | null ?? null,
  })

  useEffect(() => {
    setLoading(true)
    listBenchmarks(undefined, true)
      .then((res) => {
        const textList = (res.text ?? []).map((e) => normalize(e, 'llm'))
        const mmList = (res.multimodal ?? []).map((e) => normalize(e, 'vlm'))
        setAllBenchmarks([...textList, ...mmList])
      })
      .catch(() => {})
      .finally(() => setLoading(false))
  }, [])

  // Debounce search
  useEffect(() => {
    timerRef.current = setTimeout(() => setDebouncedSearch(search), 300)
    return () => clearTimeout(timerRef.current)
  }, [search])

  // Extract unique tags from all benchmarks
  const allTags = useMemo(() => {
    const tagSet = new Set<string>()
    for (const b of allBenchmarks) {
      for (const tag of b.tags ?? []) {
        tagSet.add(tag)
      }
    }
    return Array.from(tagSet).sort()
  }, [allBenchmarks])

  // Count by category
  const textCount = useMemo(() => allBenchmarks.filter((b) => b.category === 'llm').length, [allBenchmarks])
  const mmCount = useMemo(() => allBenchmarks.filter((b) => b.category === 'vlm').length, [allBenchmarks])

  // Filter by tab
  const tabFiltered = useMemo(() => {
    if (tab === 'text') return allBenchmarks.filter((b) => b.category === 'llm')
    if (tab === 'multimodal') return allBenchmarks.filter((b) => b.category === 'vlm')
    return allBenchmarks
  }, [allBenchmarks, tab])

  const getDescription = useCallback(
    (entry: BenchmarkEntry) => {
      const desc = locale === 'zh' ? entry.description?.zh : entry.description?.en
      const full = desc?.full
      if (!full) return t('benchmarks.noDescription')
      return stripMarkdown(full)
    },
    [locale, t],
  )

  // Filter by search + tags
  const items = useMemo(() => {
    let result = tabFiltered
    if (selectedTags.length > 0) {
      result = result.filter((e) => selectedTags.some((tag) => (e.tags ?? []).includes(tag)))
    }
    if (debouncedSearch) {
      const q = debouncedSearch.toLowerCase()
      result = result.filter(
        (e) =>
          e.name.toLowerCase().includes(q) ||
          (e.pretty_name ?? '').toLowerCase().includes(q) ||
          getDescription(e).toLowerCase().includes(q) ||
          (e.tags ?? []).some((tag) => tag.toLowerCase().includes(q)),
      )
    }
    return result
  }, [tabFiltered, debouncedSearch, selectedTags, getDescription])

  const handleCardClick = useCallback((entry: BenchmarkEntry) => {
    setSelectedEntry(entry)
  }, [])

  const closeDetail = useCallback(() => {
    setSelectedEntry(null)
  }, [])

  const toggleTag = (tag: string) => {
    setSelectedTags((prev) => (prev.includes(tag) ? prev.filter((t) => t !== tag) : [...prev, tag]))
  }

  const clearFilters = () => {
    setSelectedTags([])
    setSearch('')
  }

  const hasActiveFilters = selectedTags.length > 0 || debouncedSearch.length > 0

  const tabItems = [
    { key: 'all', label: `${t('benchmarks.all')} (${allBenchmarks.length})` },
    { key: 'text', label: `${t('benchmarks.text')} (${textCount})` },
    { key: 'multimodal', label: `${t('benchmarks.multimodal')} (${mmCount})` },
  ]

  if (loading) return <LoadingSpinner />

  return (
    <div className="page-enter space-y-5">
      {/* Header */}
      <div className="flex items-center justify-between">
        <h1 className="text-xl font-semibold">{t('benchmarks.title')}</h1>
        <span className="text-sm text-[var(--text-muted)]">
          {t('benchmarks.showing')
            .replace('${n}', String(items.length))
            .replace('${total}', String(allBenchmarks.length))}
        </span>
      </div>

      {/* Controls row */}
      <div className="flex flex-col gap-3">
        <div className="flex flex-col sm:flex-row items-start sm:items-center gap-3">
          <Tabs tabs={tabItems} activeKey={tab} onChange={(k) => setTab(k as TabKey)} />
          <SearchInput
            value={search}
            onChange={setSearch}
            placeholder={t('benchmarks.search')}
            className="sm:ml-auto w-full sm:w-72"
          />
        </div>

        {/* Tag filters */}
        {allTags.length > 0 && (
          <div className="space-y-1.5">
            <div className="flex items-center gap-2 text-xs text-[var(--text-muted)]">
              <Tag size={12} />
              <span>{t('benchmarks.filterByTag')}</span>
            </div>
            <div className="flex flex-wrap gap-1.5">
              {allTags.map((tag) => (
                <button
                  key={tag}
                  onClick={() => toggleTag(tag)}
                  className={[
                    'px-2.5 py-1 rounded-full text-xs font-medium transition-all duration-[var(--transition)] cursor-pointer border',
                    selectedTags.includes(tag)
                      ? 'bg-[var(--accent)] text-[var(--text-on-filled)] border-[var(--accent)] shadow-[var(--shadow-glow-soft)]'
                      : 'bg-[var(--bg-card)] text-[var(--text-muted)] border-[var(--border)] hover:border-[var(--accent-dim)] hover:text-[var(--accent)]',
                  ].join(' ')}
                >
                  {tag}
                </button>
              ))}
            </div>
          </div>
        )}

        {/* Active filters + clear */}
        {hasActiveFilters && (
          <div className="flex items-center gap-2">
            {selectedTags.map((tag) => (
              <span
                key={tag}
                className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-medium bg-[var(--accent-dim)] text-[var(--accent)]"
              >
                {tag}
                <button onClick={() => toggleTag(tag)} className="cursor-pointer hover:text-white">
                  <X size={10} />
                </button>
              </span>
            ))}
            {debouncedSearch && (
              <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-medium bg-[var(--accent-dim)] text-[var(--accent)]">
                "{debouncedSearch}"
                <button onClick={() => setSearch('')} className="cursor-pointer hover:text-white">
                  <X size={10} />
                </button>
              </span>
            )}
            <button
              onClick={clearFilters}
              className="px-3 py-1.5 text-xs font-medium rounded-[var(--radius-sm)] text-[var(--text-muted)] hover:text-[var(--text)] hover:bg-[var(--bg-card2)] transition-colors cursor-pointer"
            >
              {t('benchmarks.clearFilters')}
            </button>
          </div>
        )}
      </div>

      {/* Cards grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
        {items.map((entry) => (
          <div
            key={entry.name}
            onClick={() => handleCardClick(entry)}
            className="card-hover rounded-[var(--radius)] border border-[var(--border)] bg-[var(--bg-card)] p-5 flex flex-col cursor-pointer"
          >
            {/* Card header */}
            <div className="flex items-start gap-3 flex-1">
              <div className="mt-0.5 shrink-0 w-8 h-8 rounded-[var(--radius-sm)] bg-[var(--accent-dim)] flex items-center justify-center">
                <BookOpen size={16} className="text-[var(--accent)]" />
              </div>
              <div className="min-w-0 flex-1">
                <div className="flex items-center gap-2">
                  <h3 className="text-sm font-semibold text-[var(--text)] truncate">
                    {entry.pretty_name}
                  </h3>
                  {entry.category === 'vlm' && (
                    <Badge variant="warning" className="text-[9px] shrink-0">VLM</Badge>
                  )}
                </div>
                <p className="text-[11px] text-[var(--text-muted)] font-mono mt-0.5">{entry.name}</p>
              </div>
            </div>

            {/* Description */}
            <p className="text-xs text-[var(--text-muted)] mt-3 line-clamp-2 leading-relaxed">
              {getDescription(entry)}
            </p>

            {/* Meta info row */}
            <div className="flex items-center gap-3 mt-3 text-[11px] text-[var(--text-muted)]">
              {entry.total_samples > 0 && (
                <span className="inline-flex items-center gap-1">
                  <Database size={10} />
                  {entry.total_samples.toLocaleString()}
                </span>
              )}
              {(entry.subset_list ?? []).length > 0 && (
                <span className="inline-flex items-center gap-1">
                  <Layers size={10} />
                  {(entry.subset_list ?? []).length} {t('benchmarks.subsets')}
                </span>
              )}
              <span className="inline-flex items-center gap-1">
                <FlaskConical size={10} />
                {t('benchmarks.shots').replace('${n}', String(entry.few_shot_num))}
              </span>
            </div>

            {/* Tags + Metrics */}
            <div className="flex flex-wrap gap-1 mt-2.5">
              {(entry.tags ?? []).map((tag) => (
                <Badge key={tag} variant="default" className="text-[9px]">
                  {tag}
                </Badge>
              ))}
              {(entry.metrics ?? []).map((m) => (
                <Badge key={m} variant="success" className="text-[9px]">
                  {m}
                </Badge>
              ))}
            </div>
          </div>
        ))}
      </div>

      {/* Detail modal */}
      {selectedEntry != null && createPortal(
        <div
          className="fixed inset-0 z-[100] flex items-center justify-center p-4"
          style={{ background: 'rgba(0,0,0,0.6)', backdropFilter: 'blur(4px)' }}
          onClick={closeDetail}
        >
          <div
            className="relative w-full max-w-3xl max-h-[85vh] rounded-[var(--radius-lg)] bg-[var(--bg-card)] border border-[var(--border)] shadow-2xl flex flex-col overflow-hidden"
            onClick={(e) => e.stopPropagation()}
          >
            {/* Modal header */}
            <div className="flex items-start gap-3 p-5 pb-3 border-b border-[var(--border)]">
              <div className="shrink-0 w-10 h-10 rounded-[var(--radius-sm)] bg-[var(--accent-dim)] flex items-center justify-center">
                <BookOpen size={20} className="text-[var(--accent)]" />
              </div>
              <div className="min-w-0 flex-1">
                <div className="flex items-center gap-2">
                  <h2 className="text-lg font-semibold text-[var(--text)]">{selectedEntry.pretty_name}</h2>
                  {selectedEntry.category === 'vlm' && (
                    <Badge variant="warning" className="text-[10px] shrink-0">VLM</Badge>
                  )}
                </div>
                <p className="text-xs text-[var(--text-muted)] font-mono mt-0.5">{selectedEntry.name}</p>
                {/* Meta row */}
                <div className="flex items-center gap-3 mt-2 text-xs text-[var(--text-muted)]">
                  {selectedEntry.total_samples > 0 && (
                    <span className="inline-flex items-center gap-1">
                      <Database size={11} />
                      {selectedEntry.total_samples.toLocaleString()} {t('benchmarks.samples')}
                    </span>
                  )}
                  {(selectedEntry.subset_list ?? []).length > 0 && (
                    <span className="inline-flex items-center gap-1">
                      <Layers size={11} />
                      {(selectedEntry.subset_list ?? []).length} {t('benchmarks.subsets')}
                    </span>
                  )}
                  <span className="inline-flex items-center gap-1">
                    <FlaskConical size={11} />
                    {t('benchmarks.shots').replace('${n}', String(selectedEntry.few_shot_num))}
                  </span>
                  {selectedEntry.paper_url && (
                    <a
                      href={selectedEntry.paper_url}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="inline-flex items-center gap-1 text-[var(--accent)] hover:underline"
                    >
                      <ExternalLink size={11} />
                      Paper
                    </a>
                  )}
                </div>
                {/* Tags + Metrics */}
                <div className="flex flex-wrap gap-1 mt-2">
                  {(selectedEntry.tags ?? []).map((tag) => (
                    <Badge key={tag} variant="default" className="text-[10px]">{tag}</Badge>
                  ))}
                  {(selectedEntry.metrics ?? []).map((m) => (
                    <Badge key={m} variant="success" className="text-[10px]">{m}</Badge>
                  ))}
                </div>
              </div>
              <button
                onClick={closeDetail}
                className="shrink-0 p-1.5 rounded-[var(--radius-sm)] text-[var(--text-muted)] hover:text-[var(--text)] hover:bg-[var(--bg-card2)] transition-colors cursor-pointer"
              >
                <X size={18} />
              </button>
            </div>

            {/* Modal body — full markdown */}
            <div className="flex-1 overflow-y-auto p-5">
              <MarkdownRenderer
                content={
                  (locale === 'zh'
                    ? selectedEntry.description?.zh?.full
                    : selectedEntry.description?.en?.full
                  ) ?? ''
                }
              />
            </div>
          </div>
        </div>,
        document.body,
      )}

      {/* Empty state */}
      {items.length === 0 && !loading && (
        <div className="text-center py-16">
          <div className="inline-flex items-center justify-center w-16 h-16 rounded-full bg-[var(--bg-deep)] mb-4">
            {/* text-dim allowed: empty-state icon (DESIGN.md §Text) */}
            <BookOpen size={24} className="text-[var(--text-dim)]" />
          </div>
          <p className="text-sm text-[var(--text-muted)]">{t('benchmarks.noResults')}</p>
          {hasActiveFilters && (
            <button
              onClick={clearFilters}
              className="mt-4 px-3 py-1.5 text-xs font-medium rounded-[var(--radius-sm)] border border-[var(--border-md)] text-[var(--text)] hover:border-[var(--accent)] hover:text-[var(--accent)] transition-colors cursor-pointer"
            >
              {t('benchmarks.clearFilters')}
            </button>
          )}
        </div>
      )}
    </div>
  )
}
