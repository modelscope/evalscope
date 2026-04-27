import { useCallback, useEffect, useMemo, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { useReports } from '@/contexts/ReportsContext'
import { useLocale } from '@/contexts/LocaleContext'
import { listReports } from '@/api/reports'
import type { ReportSummary } from '@/api/types'
import Card from '@/components/ui/Card'
import Badge from '@/components/ui/Badge'
import Button from '@/components/ui/Button'
import Skeleton from '@/components/ui/Skeleton'
import {
  FileText,
  Cpu,
  Database,
  Clock,
  Search,
  Inbox,
  FolderInput,
  FolderOpen,
  ChevronDown,
  ChevronRight,
} from 'lucide-react'

// ------------------------------------------------------------------ //
// Helpers                                                              //
// ------------------------------------------------------------------ //
function scoreColor(score: number): string {
  const hue = Math.round(score * 120)
  return `hsl(${hue}, 70%, 45%)`
}

function scoreBg(score: number): string {
  const hue = Math.round(score * 120)
  return `hsla(${hue}, 70%, 45%, 0.15)`
}

// ------------------------------------------------------------------ //
// KPI Card                                                            //
// ------------------------------------------------------------------ //
interface KpiCardProps {
  icon: React.ReactNode
  value: string
  label: string
  gradient: string
  delay?: number
}

function KpiCard({ icon, value, label, gradient, delay = 0 }: KpiCardProps) {
  return (
    <div
      className="kpi-card"
      style={{
        animationDelay: `${delay}ms`,
        background: 'linear-gradient(135deg, rgba(129,109,248,0.10) 0%, rgba(129,109,248,0.03) 100%)',
        border: '1px solid var(--border)',
        borderRadius: 'var(--radius)',
        padding: '1.25rem',
        backdropFilter: 'blur(12px)',
      }}
    >
      <div className="flex items-start justify-between mb-3">
        <div
          className="w-10 h-10 rounded-xl flex items-center justify-center text-white"
          style={{ background: gradient }}
        >
          {icon}
        </div>
      </div>
      <div className="text-2xl font-bold text-[var(--text)] tracking-tight">{value}</div>
      <div className="text-xs text-[var(--text-muted)] mt-0.5 font-medium">{label}</div>
    </div>
  )
}

// ------------------------------------------------------------------ //
// Dataset Score Chip                                                   //
// ------------------------------------------------------------------ //
function DatasetChip({ name, score }: { name: string; score: number }) {
  const fg = scoreColor(score)
  const bg = scoreBg(score)
  return (
    <span
      className="px-2 py-0.5 rounded-full text-[10px] font-mono whitespace-nowrap"
      style={{ background: bg, color: fg }}
    >
      {name} {(score * 100).toFixed(1)}
    </span>
  )
}

// ------------------------------------------------------------------ //
// EvalRunCard (Timeline view)                                          //
// ------------------------------------------------------------------ //
interface EvalRunCardProps {
  report: ReportSummary
  onClick: () => void
}

function EvalRunCard({ report, onClick }: EvalRunCardProps) {
  const score = report.score
  const fg = scoreColor(score)
  const bg = scoreBg(score)
  const dsScores = report.dataset_scores

  return (
    <button
      onClick={onClick}
      className="rounded-[var(--radius)] border border-[var(--border)] bg-[var(--bg-card)] hover:border-[var(--border-md)] transition-colors p-4 cursor-pointer w-full text-left"
    >
      {/* Top row */}
      <div className="flex items-center gap-2">
        <span className="text-sm font-semibold text-[var(--text)] truncate flex-1 min-w-0">
          {report.model_name}
        </span>
        {report.timestamp && (
          <span className="text-[10px] font-mono text-[var(--text-dim)] shrink-0">
            {report.timestamp.replace('T', ' ')}
          </span>
        )}
        <span
          className="shrink-0 px-2.5 py-0.5 rounded-full text-xs font-bold tabular-nums"
          style={{ background: bg, color: fg }}
        >
          {(score * 100).toFixed(1)}%
        </span>
      </div>

      {/* Dataset chips */}
      <div className="flex flex-wrap gap-1.5 mt-2">
        {dsScores && Object.keys(dsScores).length > 0 ? (
          Object.entries(dsScores).map(([ds, s]) => (
            <DatasetChip key={ds} name={ds} score={s} />
          ))
        ) : (
          <span className="text-[10px] text-[var(--text-dim)] font-mono">{report.dataset_name}</span>
        )}
      </div>
    </button>
  )
}

// ------------------------------------------------------------------ //
// CompactRunRow (Grouped view)                                         //
// ------------------------------------------------------------------ //
interface CompactRunRowProps {
  report: ReportSummary
  onClick: () => void
}

function CompactRunRow({ report, onClick }: CompactRunRowProps) {
  const score = report.score
  const fg = scoreColor(score)
  const bg = scoreBg(score)
  const dsScores = report.dataset_scores

  return (
    <button
      onClick={onClick}
      className="flex items-center gap-3 py-2.5 px-3 rounded-[var(--radius-sm)] hover:bg-[var(--bg-card2)] transition-colors w-full text-left"
    >
      {report.timestamp && (
        <span className="text-[10px] font-mono text-[var(--text-dim)] shrink-0 w-[110px]">
          {report.timestamp.replace('T', ' ').slice(5)}
        </span>
      )}
      <div className="flex flex-wrap gap-1 flex-1 min-w-0">
        {dsScores && Object.keys(dsScores).length > 0 ? (
          Object.entries(dsScores).map(([ds, s]) => (
            <DatasetChip key={ds} name={ds} score={s} />
          ))
        ) : (
          <span className="text-[10px] text-[var(--text-dim)] font-mono">{report.dataset_name}</span>
        )}
      </div>
      <span
        className="shrink-0 px-2 py-0.5 rounded-full text-[10px] font-bold tabular-nums"
        style={{ background: bg, color: fg }}
      >
        {(score * 100).toFixed(1)}%
      </span>
    </button>
  )
}

// ------------------------------------------------------------------ //
// KPI Skeleton                                                        //
// ------------------------------------------------------------------ //
function KpiSkeleton() {
  return (
    <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
      {Array.from({ length: 4 }).map((_, i) => (
        <div
          key={i}
          className="rounded-[var(--radius)] border border-[var(--border)] bg-[var(--bg-card)] p-5"
        >
          <Skeleton width={40} height={40} className="mb-3" />
          <Skeleton width={60} height={28} className="mb-1" />
          <Skeleton width={100} height={14} />
        </div>
      ))}
    </div>
  )
}

// ------------------------------------------------------------------ //
// Dashboard Page                                                      //
// ------------------------------------------------------------------ //
export default function DashboardPage() {
  const { t } = useLocale()
  const { rootPath, setRootPath } = useReports()
  const navigate = useNavigate()

  const [pathInput, setPathInput] = useState(rootPath || './outputs')
  const [scanning, setScanning] = useState(false)
  const [reports, setReports] = useState<ReportSummary[]>([])
  const [scanned, setScanned] = useState(false)

  // Evaluation list state
  const [view, setView] = useState<'timeline' | 'grouped'>('timeline')
  const [search, setSearch] = useState('')
  const [sortBy, setSortBy] = useState('time')
  const [expandedGroups, setExpandedGroups] = useState<Set<string>>(new Set())

  // Scan for reports using listReports API
  const handleScan = useCallback(async () => {
    const trimmed = pathInput.trim()
    if (!trimmed) return
    setRootPath(trimmed)
    setScanning(true)
    try {
      const res = await listReports({ rootPath: trimmed, pageSize: 1000, sortBy: 'time', sortOrder: 'desc' })
      setReports(res.reports)
      setScanned(true)
    } catch {
      setReports([])
      setScanned(true)
    } finally {
      setScanning(false)
    }
  }, [pathInput, setRootPath])

  // Auto-scan if rootPath is already set on mount
  useEffect(() => {
    if (rootPath && !scanned) {
      setPathInput(rootPath)
      handleScan()
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  // Initialize expanded groups when reports change
  useEffect(() => {
    const models = new Set(reports.map(r => r.model_name))
    setExpandedGroups(models)
  }, [reports])

  // KPI stats
  const kpiStats = useMemo(() => {
    const totalEvals = reports.length
    const models = new Set(reports.map((r) => r.model_name))
    const datasets = new Set(reports.map((r) => r.dataset_name))
    const latest = reports.length > 0
      ? (reports[0].timestamp || reports[0].name).replace('T', ' ')
      : t('dashboard.neverText')
    return { totalEvals, models: models.size, datasets: datasets.size, latest }
  }, [reports, t])

  // Filtered & sorted reports
  const sortedReports = useMemo(() => {
    let filtered = reports
    if (search) {
      const q = search.toLowerCase()
      filtered = reports.filter(r =>
        r.model_name.toLowerCase().includes(q) ||
        r.dataset_name.toLowerCase().includes(q)
      )
    }
    return [...filtered].sort((a, b) => {
      if (sortBy === 'time') return (b.timestamp || '').localeCompare(a.timestamp || '')
      if (sortBy === 'score') return (b.score ?? 0) - (a.score ?? 0)
      if (sortBy === 'model') return a.model_name.localeCompare(b.model_name)
      return 0
    })
  }, [reports, search, sortBy])

  // Grouped by model
  const grouped = useMemo(() => {
    const map = new Map<string, ReportSummary[]>()
    for (const r of sortedReports) {
      const list = map.get(r.model_name) || []
      list.push(r)
      map.set(r.model_name, list)
    }
    return Array.from(map.entries())
  }, [sortedReports])

  const toggleGroup = (model: string) => {
    setExpandedGroups(prev => {
      const next = new Set(prev)
      if (next.has(model)) next.delete(model)
      else next.add(model)
      return next
    })
  }

  const navigateToReport = (report: ReportSummary) => {
    navigate(`/reports/${encodeURIComponent(report.name)}?root_path=${encodeURIComponent(rootPath)}`)
  }

  const hasData = scanned && reports.length > 0

  return (
    <div className="flex flex-col gap-5 min-h-0">
      {/* ── Path Bar ── */}
      <div
        className="flex items-center gap-3 p-3 rounded-[var(--radius)] border border-[var(--border)] bg-[var(--bg-card)]"
        style={{ boxShadow: 'var(--shadow-sm)' }}
      >
        <FolderInput size={18} className="text-[var(--accent)] shrink-0" />
        <input
          type="text"
          value={pathInput}
          onChange={(e) => setPathInput(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && handleScan()}
          placeholder={t('dashboard.pathPlaceholder')}
          className="flex-1 min-w-0 px-3 py-2 text-sm rounded-[var(--radius-sm)] bg-[var(--bg-deep)] border border-[var(--border)] text-[var(--text)] placeholder:text-[var(--text-dim)] focus:outline-none focus:border-[var(--accent)] focus:ring-1 focus:ring-[var(--accent-dim)] transition-all duration-150"
        />
        <Button
          onClick={handleScan}
          disabled={scanning || !pathInput.trim()}
          size="md"
        >
          {scanning ? (
            <span className="flex items-center gap-1.5">
              <svg className="animate-spin w-3.5 h-3.5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={3}>
                <circle cx="12" cy="12" r="9" strokeOpacity={0.25} />
                <path d="M21 12a9 9 0 11-9-9" />
              </svg>
              {t('dashboard.scanning')}
            </span>
          ) : (
            <span className="flex items-center gap-1.5">
              <Search size={14} />
              {t('dashboard.scanBtn')}
            </span>
          )}
        </Button>
      </div>

      {/* ── KPI Cards ── */}
      {scanning ? (
        <KpiSkeleton />
      ) : (
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
          <KpiCard
            icon={<FileText size={18} strokeWidth={2} />}
            value={String(kpiStats.totalEvals)}
            label={t('dashboard.totalEvaluations')}
            gradient="linear-gradient(135deg, #6366f1, #8b5cf6)"
            delay={0}
          />
          <KpiCard
            icon={<Cpu size={18} strokeWidth={2} />}
            value={String(kpiStats.models)}
            label={t('dashboard.modelsEvaluated')}
            gradient="linear-gradient(135deg, #10b981, #06b6d4)"
            delay={60}
          />
          <KpiCard
            icon={<Database size={18} strokeWidth={2} />}
            value={String(kpiStats.datasets)}
            label={t('dashboard.datasetsUsed')}
            gradient="linear-gradient(135deg, #f59e0b, #f97316)"
            delay={120}
          />
          <KpiCard
            icon={<Clock size={18} strokeWidth={2} />}
            value={kpiStats.latest.length > 20 ? kpiStats.latest.slice(0, 20) + '…' : kpiStats.latest}
            label={t('dashboard.latestEval')}
            gradient="linear-gradient(135deg, #ec4899, #8b5cf6)"
            delay={180}
          />
        </div>
      )}

      {/* ── Loading skeleton for content ── */}
      {scanning && (
        <Card title={t('dashboard.evaluations')}>
          <Skeleton lines={8} height={14} />
        </Card>
      )}

      {/* ── Unified Evaluation List ── */}
      {hasData && !scanning && (
        <Card
          title={t('dashboard.evaluations')}
          badge={<Badge>{sortedReports.length}</Badge>}
        >
          {/* Controls bar */}
          <div className="flex items-center gap-3 flex-wrap mb-4">
            {/* View toggle */}
            <div
              style={{
                display: 'inline-flex',
                borderRadius: 'var(--radius-sm)',
                border: '1px solid var(--border-md)',
                overflow: 'hidden',
              }}
            >
              <button
                onClick={() => setView('timeline')}
                style={{
                  padding: '6px 14px',
                  fontSize: '12px',
                  background: view === 'timeline' ? 'var(--accent)' : 'var(--bg-card2)',
                  color: view === 'timeline' ? '#fff' : 'var(--text-muted)',
                  border: 'none',
                  cursor: 'pointer',
                  transition: 'all 0.15s',
                }}
              >
                {t('dashboard.timelineView')}
              </button>
              <button
                onClick={() => setView('grouped')}
                style={{
                  padding: '6px 14px',
                  fontSize: '12px',
                  background: view === 'grouped' ? 'var(--accent)' : 'var(--bg-card2)',
                  color: view === 'grouped' ? '#fff' : 'var(--text-muted)',
                  border: 'none',
                  cursor: 'pointer',
                  transition: 'all 0.15s',
                }}
              >
                {t('dashboard.groupedView')}
              </button>
            </div>

            {/* Search */}
            <input
              type="text"
              placeholder={t('dashboard.searchPlaceholder')}
              value={search}
              onChange={e => setSearch(e.target.value)}
              className="flex-1 min-w-[160px] max-w-[300px] px-3 py-1.5 text-xs rounded-[var(--radius-sm)] bg-[var(--bg-deep)] border border-[var(--border)] text-[var(--text)] placeholder-[var(--text-dim)]"
            />

            {/* Sort */}
            <select
              value={sortBy}
              onChange={e => setSortBy(e.target.value)}
              className="px-3 py-1.5 text-xs rounded-[var(--radius-sm)] bg-[var(--bg-deep)] border border-[var(--border)] text-[var(--text)]"
            >
              <option value="time">{t('dashboard.sortTime')}</option>
              <option value="score">{t('dashboard.sortScore')}</option>
              <option value="model">{t('dashboard.sortModel')}</option>
            </select>
          </div>

          {/* List content */}
          <div style={{ maxHeight: 'calc(100vh - 300px)', overflowY: 'auto' }}>
            {sortedReports.length === 0 ? (
              <div className="text-center py-8 text-sm text-[var(--text-dim)]">
                {t('dashboard.noEvals')}
              </div>
            ) : view === 'timeline' ? (
              /* ── Timeline view ── */
              <div className="flex flex-col gap-3">
                {sortedReports.map((report) => (
                  <EvalRunCard
                    key={`${report.name}-${report.dataset_name}`}
                    report={report}
                    onClick={() => navigateToReport(report)}
                  />
                ))}
              </div>
            ) : (
              /* ── Grouped view ── */
              <div className="flex flex-col gap-2">
                {grouped.map(([model, runs]) => {
                  const expanded = expandedGroups.has(model)
                  const bestScore = Math.max(...runs.map(r => r.score))
                  const bestFg = scoreColor(bestScore)

                  return (
                    <div key={model} className="rounded-[var(--radius)] border border-[var(--border)] bg-[var(--bg-card)] overflow-hidden">
                      {/* Group header */}
                      <button
                        onClick={() => toggleGroup(model)}
                        className="flex items-center gap-2 w-full px-4 py-3 hover:bg-[var(--bg-card2)] transition-colors text-left"
                      >
                        {expanded ? (
                          <ChevronDown size={14} className="text-[var(--text-dim)] shrink-0" />
                        ) : (
                          <ChevronRight size={14} className="text-[var(--text-dim)] shrink-0" />
                        )}
                        <span className="text-sm font-semibold text-[var(--text)]">{model}</span>
                        <span className="text-[10px] text-[var(--text-dim)] font-mono">
                          ({runs.length} {t('dashboard.runs')})
                        </span>
                        <span className="ml-auto text-[10px] font-mono" style={{ color: bestFg }}>
                          {t('dashboard.bestScore')}: {(bestScore * 100).toFixed(1)}%
                        </span>
                      </button>

                      {/* Group entries */}
                      {expanded && (
                        <div className="border-t border-[var(--border)]">
                          {runs.map((report) => (
                            <CompactRunRow
                              key={`${report.name}-${report.dataset_name}`}
                              report={report}
                              onClick={() => navigateToReport(report)}
                            />
                          ))}
                        </div>
                      )}
                    </div>
                  )
                })}
              </div>
            )}
          </div>
        </Card>
      )}

      {/* ── Empty state after scan ── */}
      {scanned && !hasData && !scanning && (
        <div className="flex flex-col items-center justify-center py-16 gap-4 rounded-[var(--radius)] border border-[var(--border)] bg-[var(--bg-card)]">
          <div className="w-16 h-16 rounded-2xl bg-[var(--bg-deep)] border border-[var(--border)] flex items-center justify-center">
            <Inbox size={28} className="text-[var(--text-dim)]" />
          </div>
          <div className="text-center">
            <p className="text-sm font-medium text-[var(--text-muted)]">{t('dashboard.noReportsYet')}</p>
            <p className="text-xs text-[var(--text-dim)] mt-1">{t('dashboard.noReportsHint')}</p>
          </div>
        </div>
      )}

      {/* ── Welcome state (before any scan) ── */}
      {!scanned && !scanning && (
        <div className="flex flex-col items-center justify-center py-12 gap-4 rounded-[var(--radius)] border border-[var(--border)] bg-[var(--bg-card)]">
          <div className="w-16 h-16 rounded-2xl bg-[var(--bg-deep)] border border-[var(--border)] flex items-center justify-center">
            <FolderOpen size={28} className="text-[var(--accent)]" />
          </div>
          <div className="text-center">
            <p className="text-sm font-semibold text-[var(--text)]">{t('dashboard.welcomeTitle')}</p>
            <p className="text-xs text-[var(--text-muted)] mt-1">{t('dashboard.welcomeDesc')}</p>
          </div>
        </div>
      )}
    </div>
  )
}
