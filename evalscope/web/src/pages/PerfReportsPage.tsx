import { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { Gauge, Inbox, ChevronRight } from 'lucide-react'
import { useLocale } from '@/contexts/LocaleContext'
import { useReports } from '@/contexts/ReportsContext'
import { listPerfRuns } from '@/api/perf'
import type { PerfRunSummary } from '@/api/types'
import Breadcrumb from '@/components/ui/Breadcrumb'
import Badge from '@/components/ui/Badge'
import Skeleton from '@/components/ui/Skeleton'
import EmptyState from '@/components/common/EmptyState'

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

function PerfRunCard({ run, onClick }: { run: PerfRunSummary; onClick: () => void }) {
  return (
    <button
      onClick={onClick}
      className="flex items-center gap-4 p-4 rounded-[var(--radius)] border border-[var(--border)] bg-[var(--bg-card)] hover:bg-[var(--bg-card2)] hover:border-[var(--border-strong)] transition-colors text-left w-full"
    >
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

  useEffect(() => {
    if (!rootPath) return
    let cancelled = false
    const load = async () => {
      setLoading(true)
      setError(null)
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
        <div className="flex flex-col gap-2">
          {runs.map((run) => (
            <PerfRunCard key={run.path} run={run} onClick={() => openRun(run)} />
          ))}
        </div>
      )}
    </div>
  )
}
