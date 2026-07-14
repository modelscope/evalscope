import { useEffect, useMemo, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { Gauge, Inbox, ChevronRight, Check, GitCompareArrows } from 'lucide-react'
import { useLocale } from '@/contexts/LocaleContext'
import { useReports } from '@/contexts/ReportsContext'
import { listPerfRuns } from '@/api/perf'
import type { PerfRunSummary } from '@/api/types'
import Breadcrumb from '@/components/ui/Breadcrumb'
import Badge from '@/components/ui/Badge'
import Skeleton from '@/components/ui/Skeleton'
import EmptyState from '@/components/common/EmptyState'
import SearchInput from '@/components/ui/SearchInput'

type SortKey = 'time' | 'rps' | 'latency'

function formatFull(ts: string): string {
  return ts ? ts.replace('T', ' ').slice(0, 19) : ''
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex flex-col">
      <span className="type-caption-mono text-[var(--text)] tabular-nums">{value}</span>
      <span className="type-table-xs uppercase tracking-wider text-[var(--text-muted)]">{label}</span>
    </div>
  )
}

function Checkbox({ checked }: { checked: boolean }) {
  return (
    <span
      role="checkbox"
      aria-checked={checked}
      className={[
        'flex items-center justify-center w-4 h-4 rounded-[4px] border transition-colors',
        checked
          ? 'bg-[var(--accent)] border-[var(--accent)] text-[var(--text-on-filled)]'
          : 'border-[var(--border-md)] text-transparent',
      ].join(' ')}
    >
      <Check size={12} />
    </span>
  )
}

function PerfRunCard({
  run,
  selected,
  onToggle,
  onClick,
}: {
  run: PerfRunSummary
  selected: boolean
  onToggle: () => void
  onClick: () => void
}) {
  return (
    <div
      className={[
        'flex items-center gap-3 p-4 rounded-[var(--radius)] border bg-[var(--bg-card)] transition-colors',
        selected ? 'border-[var(--accent)]' : 'border-[var(--border)] hover:border-[var(--border-strong)]',
      ].join(' ')}
    >
      <button onClick={onToggle} aria-label="Select for compare" className="shrink-0 p-0.5 cursor-pointer">
        <Checkbox checked={selected} />
      </button>
      <button onClick={onClick} className="flex items-center gap-4 flex-1 min-w-0 text-left">
        <span className="text-[var(--accent)] shrink-0">
          <Gauge size={20} />
        </span>
        <div className="flex flex-col min-w-0 flex-1 gap-0.5">
          <div className="flex items-center gap-2 min-w-0">
            <span className="type-title-md text-[var(--text)] truncate">{run.model}</span>
            {run.api_type && <Badge>{run.api_type}</Badge>}
          </div>
          <span className="type-caption-mono text-[var(--text-muted)] truncate">
            {(run.dataset || '—')} · {formatFull(run.timestamp)}
          </span>
        </div>
        <div className="hidden sm:flex items-center gap-6 shrink-0">
          <Stat label="Runs" value={String(run.num_runs)} />
          <Stat label="Best RPS" value={run.best_rps.toFixed(2)} />
          <Stat label="Min Lat" value={`${run.best_latency.toFixed(2)}s`} />
          <Stat label="Success" value={`${run.success_rate.toFixed(0)}%`} />
        </div>
        <ChevronRight size={16} className="text-[var(--text-dim)] shrink-0" />
      </button>
    </div>
  )
}

export default function PerfReportsPage() {
  const { t } = useLocale()
  const navigate = useNavigate()
  const { rootPath, scanToken } = useReports()

  const [runs, setRuns] = useState<PerfRunSummary[]>([])
  const [loading, setLoading] = useState(false)
  const [hasLoaded, setHasLoaded] = useState(false)
  const [error, setError] = useState<string | null>(null)

  // List controls (symmetric with the Evaluations page).
  const [query, setQuery] = useState('')
  const [sortBy, setSortBy] = useState<SortKey>('time')

  // Multi-select for cross-run comparison (page-local; independent of eval Compare).
  const [selected, setSelected] = useState<string[]>([])

  const toggleSelect = (path: string) =>
    setSelected((prev) => (prev.includes(path) ? prev.filter((p) => p !== path) : [...prev, path]))

  const compareSelected = () => {
    if (selected.length < 2) return
    navigate(
      `/perf-compare?paths=${encodeURIComponent(selected.join(';'))}&root_path=${encodeURIComponent(rootPath)}`,
    )
  }

  useEffect(() => {
    if (!rootPath) return
    let cancelled = false
    const load = async () => {
      setLoading(true)
      setError(null)
      setSelected([])
      try {
        const res = await listPerfRuns(rootPath)
        if (!cancelled) setRuns(res.runs)
      } catch (err) {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : 'Failed to load perf runs')
          setRuns([])
        }
      } finally {
        if (!cancelled) {
          setLoading(false)
          setHasLoaded(true)
        }
      }
    }
    load()
    return () => {
      cancelled = true
    }
  }, [rootPath, scanToken])

  const openRun = (run: PerfRunSummary) => {
    navigate(`/perf-report?path=${encodeURIComponent(run.path)}&root_path=${encodeURIComponent(rootPath)}`)
  }

  // Apply keyword search + sort.
  const visibleRuns = useMemo(() => {
    const q = query.trim().toLowerCase()
    const filtered = q
      ? runs.filter(
          (r) =>
            (r.model || '').toLowerCase().includes(q) ||
            (r.dataset || '').toLowerCase().includes(q) ||
            (r.api_type || '').toLowerCase().includes(q),
        )
      : runs
    const sorted = [...filtered]
    if (sortBy === 'rps') sorted.sort((a, b) => b.best_rps - a.best_rps)
    else if (sortBy === 'latency') sorted.sort((a, b) => a.best_latency - b.best_latency)
    else sorted.sort((a, b) => (b.timestamp || '').localeCompare(a.timestamp || ''))
    return sorted
  }, [runs, query, sortBy])

  return (
    <div className="page-enter flex flex-col gap-5">
      <Breadcrumb items={[{ label: t('nav.performance') }]} />

      {error && (
        <div className="px-4 py-3 rounded-[var(--radius)] bg-[var(--danger-bg)] border border-[var(--danger-border)] text-sm text-[var(--danger)]">
          {error}
        </div>
      )}

      {loading ? (
        <div className="flex flex-col gap-2">
          {Array.from({ length: 5 }).map((_, i) => (
            <Skeleton key={i} height={72} className="rounded-[var(--radius)]" />
          ))}
        </div>
      ) : runs.length === 0 ? (
        <div className="rounded-[var(--radius)] border border-[var(--border)] bg-[var(--bg-card)]">
          <EmptyState
            icon={<Inbox size={28} strokeWidth={1.5} />}
            title={t('performance.noRuns')}
            hint={hasLoaded ? t('performance.noRunsHint') : ''}
          />
        </div>
      ) : (
        <>
          {/* Controls */}
          <div className="flex flex-col sm:flex-row sm:items-center gap-2">
            <div className="flex items-center gap-1 p-0.5 rounded-[var(--radius-sm)] bg-[var(--bg-deep)] border border-[var(--border)] w-fit">
              {(['time', 'rps', 'latency'] as const).map((k) => (
                <button
                  key={k}
                  onClick={() => setSortBy(k)}
                  className={[
                    'px-3 py-1 rounded-[var(--radius-sm)] type-body-xs transition-colors',
                    sortBy === k
                      ? 'bg-[var(--accent)] text-[var(--text-on-filled)]'
                      : 'text-[var(--text-muted)] hover:text-[var(--text)]',
                  ].join(' ')}
                >
                  {t(`performance.sort_${k}`)}
                </button>
              ))}
            </div>
          <div className="flex items-center gap-2 sm:ml-auto w-full sm:w-auto">
            <button
              onClick={compareSelected}
              disabled={selected.length < 2}
              className={[
                'inline-flex items-center gap-1.5 px-3 py-1.5 rounded-[var(--radius-sm)] type-body-sm border transition-colors shrink-0',
                selected.length >= 2
                  ? 'border-[var(--accent)] text-[var(--accent)] hover:bg-[var(--bg-card2)] cursor-pointer'
                  : 'border-[var(--border)] text-[var(--text-dim)] cursor-not-allowed',
              ].join(' ')}
            >
              <GitCompareArrows size={14} />
              {t('performance.compareN').replace('${n}', String(selected.length))}
            </button>
            <SearchInput
              value={query}
              onChange={setQuery}
              placeholder={t('performance.searchPlaceholder')}
              className="w-full sm:w-64"
            />
          </div>
          </div>

          {visibleRuns.length > 0 ? (
            <div className="flex flex-col gap-2">
              {visibleRuns.map((run) => (
                <PerfRunCard
                  key={run.path}
                  run={run}
                  selected={selected.includes(run.path)}
                  onToggle={() => toggleSelect(run.path)}
                  onClick={() => openRun(run)}
                />
              ))}
            </div>
          ) : (
            <div className="py-10 text-center type-body-sm text-[var(--text-muted)]">{t('performance.noMatch')}</div>
          )}
        </>
      )}
    </div>
  )
}
