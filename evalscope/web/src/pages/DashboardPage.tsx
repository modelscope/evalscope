import { useMemo, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { useReports } from '@/contexts/ReportsContext'
import { useLocale } from '@/contexts/LocaleContext'
import Sidebar from '@/components/sidebar/Sidebar'
import DatasetsOverview from '@/components/single/DatasetsOverview'
import ModelsOverview from '@/components/multi/ModelsOverview'
import LoadingSpinner from '@/components/common/LoadingSpinner'
import type { ReportData } from '@/api/types'
import { parseReportName } from '@/utils/reportParser'
import { scoreBg, rdYlGn } from '@/utils/colorScale'
import {
  BarChart3,
  FlaskConical,
  Gauge,
  CheckCircle2,
  Database,
  Clock,
  Layers3,
  ArrowRight,
  TrendingUp,
  Inbox,
} from 'lucide-react'

type Tab = 'single' | 'multi'

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
      className="kpi-card p-5 flex flex-col gap-3"
      style={{ animationDelay: `${delay}ms` }}
    >
      <div className="flex items-start justify-between">
        <div
          className="w-10 h-10 rounded-xl flex items-center justify-center text-white"
          style={{ background: gradient }}
        >
          {icon}
        </div>
        <TrendingUp size={14} className="text-[var(--color-ink-faint)] opacity-60" />
      </div>
      <div>
        <div className="text-2xl font-bold text-[var(--color-ink)] tracking-tight">{value}</div>
        <div className="text-xs text-[var(--color-ink-muted)] mt-0.5 font-medium">{label}</div>
      </div>
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
      className="group card-hover text-left p-5 rounded-2xl border border-[var(--color-border)] bg-[var(--color-surface)] flex items-start gap-4"
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
          <span className="text-sm font-semibold text-[var(--color-ink)]">{title}</span>
          <ArrowRight
            size={14}
            className="text-[var(--color-ink-faint)] group-hover:text-[var(--color-primary)] group-hover:translate-x-0.5 transition-all duration-200 shrink-0"
          />
        </div>
        <p className="text-xs text-[var(--color-ink-muted)] mt-0.5 leading-relaxed">{desc}</p>
      </div>
    </button>
  )
}

// ------------------------------------------------------------------ //
// Score Matrix                                                        //
// ------------------------------------------------------------------ //
interface ScoreMatrixProps {
  reports: ReportData[]
}

function ScoreMatrix({ reports }: ScoreMatrixProps) {
  const { t } = useLocale()

  // Build model x dataset matrix
  const models = useMemo(() => [...new Set(reports.map((r) => r.model_name))], [reports])
  const datasets = useMemo(() => [...new Set(reports.map((r) => r.dataset_name))], [reports])

  const matrix = useMemo(() => {
    const map: Record<string, Record<string, number>> = {}
    for (const r of reports) {
      if (!map[r.model_name]) map[r.model_name] = {}
      map[r.model_name][r.dataset_name] = r.score
    }
    return map
  }, [reports])

  if (models.length === 0 || datasets.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center py-12 gap-3">
        <Inbox size={32} className="text-[var(--color-ink-faint)]" />
        <p className="text-sm text-[var(--color-ink-muted)]">{t('dashboard.noReportsLoaded')}</p>
        <p className="text-xs text-[var(--color-ink-faint)]">{t('dashboard.noReportsHint')}</p>
      </div>
    )
  }

  return (
    <div className="overflow-x-auto">
      <table className="w-full text-xs border-separate border-spacing-0.5">
        <thead>
          <tr>
            <th className="text-left px-3 py-2 text-[var(--color-ink-faint)] font-medium w-40 shrink-0">
              {t('dashboard.model')} / {t('dashboard.dataset')}
            </th>
            {datasets.map((ds) => (
              <th key={ds} className="px-2 py-2 text-[var(--color-ink-muted)] font-medium max-w-[120px] text-center">
                <span className="truncate block" title={ds}>
                  {ds.length > 14 ? ds.slice(0, 14) + '…' : ds}
                </span>
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {models.map((model) => (
            <tr key={model}>
              <td className="px-3 py-1.5 text-[var(--color-ink-muted)] font-medium">
                <span className="truncate block max-w-[150px]" title={model}>
                  {model.length > 18 ? model.slice(0, 18) + '…' : model}
                </span>
              </td>
              {datasets.map((ds) => {
                const score = matrix[model]?.[ds]
                if (score === undefined) {
                  return (
                    <td key={ds} className="px-2 py-1.5 text-center">
                      <span className="text-[var(--color-ink-faint)]">—</span>
                    </td>
                  )
                }
                const bg = scoreBg(score, 0.45)
                const fg = rdYlGn(score)
                return (
                  <td
                    key={ds}
                    className="px-2 py-1 text-center rounded-lg"
                    style={{ background: bg }}
                  >
                    <span className="font-bold tabular-nums" style={{ color: fg }}>
                      {(score * 100).toFixed(1)}
                    </span>
                  </td>
                )
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

// ------------------------------------------------------------------ //
// Timeline Entry                                                      //
// ------------------------------------------------------------------ //
interface TimelineEntryProps {
  report: ReportData
  index: number
}

function TimelineEntry({ report, index }: TimelineEntryProps) {
  const score = report.score
  const scorePct = (score * 100).toFixed(1)
  const bg = scoreBg(score, 0.3)
  const fg = rdYlGn(score)

  return (
    <div
      className="flex items-start gap-3 py-3 border-b border-[var(--color-border-subtle)] last:border-0 group hover:bg-[var(--color-surface-hover)] -mx-3 px-3 rounded-lg transition-colors duration-150"
      style={{ animationDelay: `${index * 40}ms` }}
    >
      {/* Index badge */}
      <div className="w-6 h-6 rounded-full bg-[var(--color-surface-2)] border border-[var(--color-border-subtle)] flex items-center justify-center text-[10px] font-bold text-[var(--color-ink-faint)] shrink-0 mt-0.5">
        {index + 1}
      </div>
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2 flex-wrap">
          <span className="text-xs font-semibold text-[var(--color-ink)] truncate max-w-[180px]" title={report.model_name}>
            {report.model_name}
          </span>
          <span className="text-[10px] text-[var(--color-ink-faint)]">·</span>
          <span className="text-xs text-[var(--color-ink-muted)] truncate max-w-[140px]" title={report.dataset_name}>
            {report.dataset_name}
          </span>
        </div>
        <div className="text-[10px] text-[var(--color-ink-faint)] mt-0.5 font-mono">
          {report.name}
        </div>
      </div>
      {/* Score badge */}
      <div
        className="shrink-0 px-2.5 py-0.5 rounded-full text-xs font-bold tabular-nums"
        style={{ background: bg, color: fg }}
      >
        {scorePct}%
      </div>
    </div>
  )
}

// ------------------------------------------------------------------ //
// Section Header                                                      //
// ------------------------------------------------------------------ //
function SectionHeader({ title, icon, action }: { title: string; icon: React.ReactNode; action?: React.ReactNode }) {
  return (
    <div className="flex items-center justify-between mb-4">
      <div className="flex items-center gap-2">
        <div className="text-[var(--color-primary)]">{icon}</div>
        <h2 className="text-sm font-semibold text-[var(--color-ink)]">{title}</h2>
      </div>
      {action}
    </div>
  )
}

// ------------------------------------------------------------------ //
// Empty State                                                         //
// ------------------------------------------------------------------ //
function DashboardEmptyState({ text }: { text: string }) {
  return (
    <div className="flex flex-col items-center justify-center py-16 gap-4">
      <div className="w-16 h-16 rounded-2xl bg-[var(--color-surface-2)] border border-[var(--color-border-subtle)] flex items-center justify-center">
        <Inbox size={28} className="text-[var(--color-ink-faint)]" />
      </div>
      <div className="text-center">
        <p className="text-sm font-medium text-[var(--color-ink-muted)]">{text}</p>
      </div>
    </div>
  )
}

// ------------------------------------------------------------------ //
// Dashboard Page                                                      //
// ------------------------------------------------------------------ //
export default function DashboardPage() {
  const { t } = useLocale()
  const {
    selectedReports,
    loading,
    loadReport,
    loadMultiReports,
    reportCache,
    multiReportList,
    availableReports,
  } = useReports()
  const navigate = useNavigate()
  const [tab, setTab] = useState<Tab>('single')
  const [loaded, setLoaded] = useState(false)

  const uniqueModels = useMemo(() => {
    const models = new Set(selectedReports.map((r) => parseReportName(r).model))
    return models.size
  }, [selectedReports])

  const handleLoadView = async () => {
    if (selectedReports.length === 0) return
    setLoaded(false)
    if (uniqueModels > 1) {
      await loadMultiReports(selectedReports)
      setTab('multi')
    } else if (selectedReports.length > 0) {
      await loadReport(selectedReports[0])
      setTab('single')
    }
    setLoaded(true)
  }

  const singleReportData = useMemo<ReportData[]>(() => {
    if (selectedReports.length === 0) return []
    const cached = reportCache[selectedReports[0]]
    return cached?.report_list ?? []
  }, [selectedReports, reportCache])

  // KPI stats from cache
  const allCachedReports = useMemo<ReportData[]>(() => {
    const all: ReportData[] = []
    for (const key of Object.keys(reportCache)) {
      all.push(...(reportCache[key]?.report_list ?? []))
    }
    return all
  }, [reportCache])

  const kpiStats = useMemo(() => {
    const totalEvals = allCachedReports.length
    const models = new Set(allCachedReports.map((r) => r.model_name))
    const datasets = new Set(allCachedReports.map((r) => r.dataset_name))
    // Try to extract date from report name prefix
    const dates = Object.keys(reportCache)
      .map((name) => parseReportName(name).prefix)
      .filter(Boolean)
      .sort()
      .reverse()
    const latest = dates[0]?.replace(/_/g, ' ').slice(0, 16) ?? t('dashboard.neverText')
    return { totalEvals, models: models.size, datasets: datasets.size, latest }
  }, [allCachedReports, reportCache, t])

  // Timeline: last 10 reports from cache
  const timelineReports = useMemo<ReportData[]>(() => {
    return allCachedReports.slice(0, 10)
  }, [allCachedReports])

  // Matrix uses multiReportList if available, else singleReportData
  const matrixReports = useMemo<ReportData[]>(() => {
    if (multiReportList.length > 0) return multiReportList
    return allCachedReports
  }, [multiReportList, allCachedReports])

  const hasData = loaded && !loading

  return (
    <div className="flex gap-5 min-h-0">
      {/* ──────────── Left sidebar ──────────── */}
      <aside className="w-72 shrink-0 rounded-2xl border border-[var(--color-border)] bg-[var(--color-surface)] p-4 flex flex-col gap-3" style={{ boxShadow: 'var(--shadow-card)' }}>
        <Sidebar />
        <button
          onClick={handleLoadView}
          disabled={loading || selectedReports.length === 0}
          className="mt-1 w-full px-3 py-2.5 text-sm font-semibold rounded-xl text-white disabled:opacity-40 transition-all duration-200 btn-glow"
          style={{
            background: selectedReports.length === 0
              ? 'var(--color-surface-2)'
              : 'var(--gradient-primary)',
            cursor: selectedReports.length === 0 ? 'not-allowed' : 'pointer',
          }}
        >
          {loading ? (
            <span className="flex items-center justify-center gap-2">
              <svg className="animate-spin w-3.5 h-3.5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={3}>
                <circle cx="12" cy="12" r="9" strokeOpacity={0.25} />
                <path d="M21 12a9 9 0 11-9-9" />
              </svg>
              {t('common.loading')}
            </span>
          ) : (
            <span className="flex items-center justify-center gap-1.5">
              <Layers3 size={14} />
              {t('sidebar.loadBtn')}
            </span>
          )}
        </button>
      </aside>

      {/* ──────────── Main content ──────────── */}
      <div className="flex-1 min-w-0 flex flex-col gap-5">

        {/* ── Hero KPI stats ── */}
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-3 stagger-children">
          <KpiCard
            icon={<CheckCircle2 size={18} strokeWidth={2} />}
            value={String(kpiStats.totalEvals)}
            label={t('dashboard.totalEvaluations')}
            gradient="linear-gradient(135deg, #6366f1, #8b5cf6)"
            delay={0}
          />
          <KpiCard
            icon={<Layers3 size={18} strokeWidth={2} />}
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
            value={kpiStats.latest}
            label={t('dashboard.latestEval')}
            gradient="linear-gradient(135deg, #ec4899, #8b5cf6)"
            delay={180}
          />
        </div>

        {/* ── Loading ── */}
        {loading && <LoadingSpinner text={t('common.loading')} />}

        {/* ── Quick Actions (only if no data loaded yet) ── */}
        {!hasData && !loading && (
          <div>
            <SectionHeader
              title={t('dashboard.quickActions')}
              icon={<BarChart3 size={15} />}
            />
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
              <QuickActionCard
                icon={<FlaskConical size={18} />}
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
                icon={<BarChart3 size={18} />}
                title={t('dashboard.browseReports')}
                desc={t('dashboard.browseReportsDesc')}
                onClick={() => {
                  // focus sidebar – just scroll to it
                  document.querySelector('aside')?.scrollIntoView({ behavior: 'smooth' })
                }}
                gradient="linear-gradient(135deg, #f59e0b, #f97316)"
                delay={120}
              />
            </div>
          </div>
        )}

        {/* ── Loaded content ── */}
        {hasData && (
          <>
            {/* Tab bar */}
            <div className="flex gap-1 border-b border-[var(--color-border-subtle)]">
              {(['single', 'multi'] as const).map((tabId) => (
                <button
                  key={tabId}
                  onClick={() => setTab(tabId)}
                  className={`px-4 py-2 text-sm font-medium border-b-2 transition-all duration-200 ${
                    tab === tabId
                      ? 'border-[var(--color-primary)] text-[var(--color-primary)]'
                      : 'border-transparent text-[var(--color-ink-muted)] hover:text-[var(--color-ink)]'
                  }`}
                >
                  {tabId === 'single' ? t('visualization.singleModel') : t('visualization.multiModel')}
                </button>
              ))}
            </div>

            {tab === 'single' && singleReportData.length > 0 && (
              <div className="space-y-4">
                <DatasetsOverview reports={singleReportData} reportName={selectedReports[0] ?? ''} />
                <div className="flex justify-end">
                  <button
                    onClick={() => navigate('/dashboard/single')}
                    className="text-sm text-[var(--color-primary)] hover:underline flex items-center gap-1"
                  >
                    {t('single.datasetDetails')} <ArrowRight size={13} />
                  </button>
                </div>
              </div>
            )}
            {tab === 'multi' && multiReportList.length > 0 && (
              <div className="space-y-4">
                <ModelsOverview reports={multiReportList} />
                <div className="flex justify-end">
                  <button
                    onClick={() => navigate('/dashboard/multi')}
                    className="text-sm text-[var(--color-primary)] hover:underline flex items-center gap-1"
                  >
                    {t('multi.modelComparisonDetails')} <ArrowRight size={13} />
                  </button>
                </div>
              </div>
            )}

            {/* Bottom panels: Timeline + Matrix */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
              {/* Recent Activity Timeline */}
              <div className="rounded-2xl border border-[var(--color-border)] bg-[var(--color-surface)] p-5" style={{ boxShadow: 'var(--shadow-card)' }}>
                <SectionHeader
                  title={t('dashboard.recentActivity')}
                  icon={<Clock size={15} />}
                  action={
                    timelineReports.length > 0 ? (
                      <button
                        onClick={() => navigate('/dashboard/single')}
                        className="text-xs text-[var(--color-primary)] hover:underline"
                      >
                        {t('dashboard.viewAll')}
                      </button>
                    ) : undefined
                  }
                />
                {timelineReports.length > 0 ? (
                  <div className="flex flex-col">
                    {timelineReports.map((r, i) => (
                      <TimelineEntry key={`${r.name}-${r.dataset_name}`} report={r} index={i} />
                    ))}
                  </div>
                ) : (
                  <DashboardEmptyState text={t('dashboard.noReportsLoaded')} />
                )}
              </div>

              {/* Score Matrix */}
              <div className="rounded-2xl border border-[var(--color-border)] bg-[var(--color-surface)] p-5" style={{ boxShadow: 'var(--shadow-card)' }}>
                <SectionHeader
                  title={t('dashboard.scoreMatrix')}
                  icon={<TrendingUp size={15} />}
                />
                <ScoreMatrix reports={matrixReports} />
              </div>
            </div>
          </>
        )}

        {/* ── No data, no loading: show quick actions ── */}
        {!hasData && !loading && availableReports.length > 0 && (
          <div className="rounded-2xl border border-[var(--color-border)] bg-[var(--color-surface)] p-5" style={{ boxShadow: 'var(--shadow-card)' }}>
            <SectionHeader
              title={t('dashboard.recentActivity')}
              icon={<Clock size={15} />}
            />
            <DashboardEmptyState text={t('dashboard.noReportsLoaded')} />
          </div>
        )}

        {!hasData && !loading && availableReports.length === 0 && (
          <div className="rounded-2xl border border-[var(--color-border)] bg-[var(--color-surface)] p-5" style={{ boxShadow: 'var(--shadow-card)' }}>
            <SectionHeader title={t('dashboard.scoreMatrix')} icon={<TrendingUp size={15} />} />
            <DashboardEmptyState text={t('dashboard.noReportsYet')} />
          </div>
        )}
      </div>
    </div>
  )
}
