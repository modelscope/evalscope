import { useCallback, useEffect, useMemo, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { useReports } from '@/contexts/ReportsContext'
import { useLocale } from '@/contexts/LocaleContext'
import { listReports } from '@/api/reports'
import type { ReportSummary } from '@/api/types'
import ScoreHeatmap from '@/components/charts/ScoreHeatmap'
import Card from '@/components/ui/Card'
import Badge from '@/components/ui/Badge'
import Button from '@/components/ui/Button'
import Skeleton from '@/components/ui/Skeleton'
import {
  FileText,
  Cpu,
  Database,
  Clock,
  PlayCircle,
  Gauge,
  FolderOpen,
  BookOpen,
  ArrowRight,
  Search,
  Inbox,
  FolderInput,
} from 'lucide-react'

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
// Quick Action Card                                                   //
// ------------------------------------------------------------------ //
interface QuickActionProps {
  icon: React.ReactNode
  title: string
  desc: string
  onClick: () => void
  gradient: string
  delay?: number
}

function QuickActionCard({ icon, title, desc, onClick, gradient, delay = 0 }: QuickActionProps) {
  return (
    <button
      onClick={onClick}
      className="group text-left p-5 rounded-[var(--radius)] border border-[var(--border)] bg-[var(--bg-card)] flex items-start gap-4 transition-all duration-200 hover:border-[var(--accent)] hover:-translate-y-0.5 hover:shadow-[var(--shadow-sm)]"
      style={{ animationDelay: `${delay}ms` }}
    >
      <div
        className="w-11 h-11 rounded-xl flex items-center justify-center text-white shrink-0 transition-transform duration-200 group-hover:scale-110"
        style={{ background: gradient }}
      >
        {icon}
      </div>
      <div className="flex-1 min-w-0">
        <div className="flex items-center justify-between gap-2">
          <span className="text-sm font-semibold text-[var(--text)]">{title}</span>
          <ArrowRight
            size={14}
            className="text-[var(--text-dim)] group-hover:text-[var(--accent)] group-hover:translate-x-0.5 transition-all duration-200 shrink-0"
          />
        </div>
        <p className="text-xs text-[var(--text-muted)] mt-0.5 leading-relaxed">{desc}</p>
      </div>
    </button>
  )
}

// ------------------------------------------------------------------ //
// Timeline Entry                                                      //
// ------------------------------------------------------------------ //
interface TimelineEntryProps {
  report: ReportSummary
  index: number
  onClick: () => void
}

function TimelineEntry({ report, index, onClick }: TimelineEntryProps) {
  const score = report.score
  const hue = Math.round(score * 120)
  const bg = `hsla(${hue}, 70%, 45%, 0.15)`
  const fg = `hsl(${hue}, 70%, 45%)`

  return (
    <button
      onClick={onClick}
      className="flex items-center gap-3 py-3 border-b border-[var(--border)] last:border-0 group hover:bg-[var(--bg-card2)] -mx-3 px-3 rounded-[var(--radius-sm)] transition-colors duration-150 w-full text-left"
    >
      {/* Index badge */}
      <div className="w-6 h-6 rounded-full bg-[var(--bg-deep)] border border-[var(--border)] flex items-center justify-center text-[10px] font-bold text-[var(--text-dim)] shrink-0">
        {index + 1}
      </div>
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2 flex-wrap">
          <span className="text-xs font-semibold text-[var(--text)] truncate max-w-[180px]" title={report.model_name}>
            {report.model_name}
          </span>
          <span className="text-[10px] text-[var(--text-dim)]">·</span>
          <span className="text-xs text-[var(--text-muted)] truncate max-w-[140px]" title={report.dataset_name}>
            {report.dataset_name}
          </span>
        </div>
        {report.timestamp && (
          <div className="text-[10px] text-[var(--text-dim)] mt-0.5 font-mono">
            {report.timestamp}
          </div>
        )}
      </div>
      {/* Score badge */}
      <div
        className="shrink-0 px-2.5 py-0.5 rounded-full text-xs font-bold tabular-nums"
        style={{ background: bg, color: fg }}
      >
        {(score * 100).toFixed(1)}%
      </div>
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

  // KPI stats
  const kpiStats = useMemo(() => {
    const totalEvals = reports.length
    const models = new Set(reports.map((r) => r.model_name))
    const datasets = new Set(reports.map((r) => r.dataset_name))
    const latest = reports.length > 0
      ? (reports[0].timestamp || reports[0].name)
      : t('dashboard.neverText')
    return { totalEvals, models: models.size, datasets: datasets.size, latest }
  }, [reports, t])

  // Heatmap data
  const heatmapData = useMemo(() => {
    return reports.map((r) => ({
      model: r.model_name,
      dataset: r.dataset_name,
      score: r.score,
      reportName: r.name,
    }))
  }, [reports])

  // Recent activity (up to 20)
  const recentReports = useMemo(() => {
    return reports.slice(0, 20)
  }, [reports])

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
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          <Card title={t('dashboard.scoreMatrix')}>
            <Skeleton lines={5} height={16} />
          </Card>
          <Card title={t('dashboard.recentActivity')}>
            <Skeleton lines={8} height={14} />
          </Card>
        </div>
      )}

      {/* ── Score Matrix + Recent Activity ── */}
      {hasData && !scanning && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          {/* Score Matrix */}
          <Card title={t('dashboard.scoreMatrix')} badge={<Badge>{heatmapData.length}</Badge>}>
            <ScoreHeatmap
              data={heatmapData}
              onCellClick={(cell) => {
                if (cell.reportName) {
                  navigate(`/reports/${encodeURIComponent(cell.reportName)}?root_path=${encodeURIComponent(rootPath)}`)
                }
              }}
            />
          </Card>

          {/* Recent Activity */}
          <Card
            title={t('dashboard.recentActivity')}
            badge={
              <button
                onClick={() => navigate(`/reports?root_path=${encodeURIComponent(rootPath)}`)}
                className="text-xs text-[var(--accent)] hover:underline ml-2"
              >
                {t('dashboard.viewAll')}
              </button>
            }
          >
            <div className="flex flex-col max-h-[480px] overflow-y-auto">
              {recentReports.map((r, i) => (
                <TimelineEntry
                  key={`${r.name}-${r.dataset_name}-${i}`}
                  report={r}
                  index={i}
                  onClick={() =>
                    navigate(`/reports/${encodeURIComponent(r.name)}?root_path=${encodeURIComponent(rootPath)}`)
                  }
                />
              ))}
            </div>
          </Card>
        </div>
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

      {/* ── Quick Actions ── */}
      <div>
        <h2 className="text-xs font-semibold uppercase tracking-wider text-[var(--text-muted)] mb-3">
          {t('dashboard.quickActions')}
        </h2>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3">
          <QuickActionCard
            icon={<PlayCircle size={18} />}
            title={t('dashboard.newEvaluation')}
            desc={t('dashboard.newEvalDesc')}
            onClick={() => navigate('/eval')}
            gradient="linear-gradient(135deg, #6366f1, #8b5cf6)"
            delay={0}
          />
          <QuickActionCard
            icon={<Gauge size={18} />}
            title={t('dashboard.newPerfTest')}
            desc={t('dashboard.newPerfDesc')}
            onClick={() => navigate('/perf')}
            gradient="linear-gradient(135deg, #10b981, #06b6d4)"
            delay={60}
          />
          <QuickActionCard
            icon={<FolderOpen size={18} />}
            title={t('dashboard.browseReports')}
            desc={t('dashboard.browseReportsDesc')}
            onClick={() => navigate(`/reports?root_path=${encodeURIComponent(rootPath)}`)}
            gradient="linear-gradient(135deg, #f59e0b, #f97316)"
            delay={120}
          />
          <QuickActionCard
            icon={<BookOpen size={18} />}
            title={t('dashboard.viewBenchmarks')}
            desc={t('dashboard.viewBenchmarksDesc')}
            onClick={() => navigate('/benchmarks')}
            gradient="linear-gradient(135deg, #ec4899, #8b5cf6)"
            delay={180}
          />
        </div>
      </div>
    </div>
  )
}
