import { Link } from 'react-router-dom'
import { FileText, Gauge } from 'lucide-react'
import { useLocale } from '@/contexts/LocaleContext'
import { formatMetric } from '@/domain/metric'
import type { MetricSemantics } from '@/domain/metric'
import type { PerfRunSummary, ReportSummary } from '@/api/types'
import { primaryMetricsOf } from '@/domain/report/primaryMetrics'

/**
 * The last few runs, in the order they happened.
 *
 * The aggregated table above answers "how are my benchmarks doing"; it cannot answer "did the run I
 * just started finish", because it folds a run into a cell that may hold seventy of them. This strip
 * exists only for that second question, so it is deliberately short and carries no filter, search or
 * pagination -- the Evaluations and Performance pages already do all of that, better.
 */

/** How many runs to show. Enough to recognise the last thing you did, not a list page. */
const STRIP_LIMIT = 4

interface ActivityItem {
  kind: 'eval' | 'perf'
  timestamp: string
  target: string
  /** Already formatted, using each metric's own semantics. */
  result: string
  href: string
}

function formatShort(timestamp: string): string {
  return timestamp ? timestamp.replace('T', ' ').slice(5, 16) : ''
}

interface ActivityStripProps {
  reports: ReportSummary[]
  perfRuns: PerfRunSummary[]
  perfSemantics: Record<string, MetricSemantics>
  rootPath: string
}

export default function ActivityStrip({ reports, perfRuns, perfSemantics, rootPath }: ActivityStripProps) {
  const { t } = useLocale()
  const root = encodeURIComponent(rootPath)

  const items: ActivityItem[] = [
    ...reports.map((report): ActivityItem => {
      const refs = primaryMetricsOf(report)
      return {
        kind: 'eval',
        timestamp: report.timestamp || '',
        target: refs.map((ref) => ref.dataset_name).filter(Boolean).join(' · ') || report.dataset_name,
        // Each dataset keeps its own metric and scale; they are listed, never averaged.
        result: refs
          .map((ref) => formatMetric(ref.score, ref.semantics).primary)
          .join('  '),
        href: `/reports/${encodeURIComponent(report.name)}?root_path=${root}`,
      }
    }),
    ...perfRuns.map((run): ActivityItem => ({
      kind: 'perf',
      timestamp: run.timestamp || '',
      target: run.dataset || run.api_type || 'perf',
      result: formatMetric(run.best_rps, perfSemantics.best_rps).primary,
      href: `/perf-report?path=${encodeURIComponent(run.path)}&root_path=${root}`,
    })),
  ]
    .sort((a, b) => b.timestamp.localeCompare(a.timestamp))
    .slice(0, STRIP_LIMIT)

  if (items.length === 0) {
    return null
  }

  return (
    <section
      aria-label={t('dashboard.activityTitle')}
      className="rounded-[var(--radius)] border border-[var(--border)] bg-[var(--bg-card)]"
    >
      <div className="flex items-center justify-between gap-2 border-b border-[var(--border)] px-4 py-2">
        <span className="type-caption text-[var(--text-muted)]">{t('dashboard.activityTitle')}</span>
        <Link to="/reports" className="type-body-xs text-[var(--accent)] hover:underline">
          {t('dashboard.viewAll')}
        </Link>
      </div>
      <ul className="divide-y divide-[var(--border)]">
        {items.map((item) => (
          <li key={`${item.kind}-${item.href}`}>
            <Link
              to={item.href}
              className="flex min-h-9 flex-wrap items-center gap-x-3 gap-y-0.5 px-4 py-1.5 transition-colors hover:bg-[var(--bg-card2)]"
            >
              <span className="shrink-0 text-[var(--text-dim)]">
                {item.kind === 'eval' ? <FileText size={12} /> : <Gauge size={12} />}
              </span>
              <span className="type-caption-mono shrink-0 text-[var(--text-dim)]">{formatShort(item.timestamp)}</span>
              <span className="type-body-xs min-w-0 break-words text-[var(--text)]">{item.target}</span>
              <span className="type-caption-mono ml-auto shrink-0 tabular-nums text-[var(--text-muted)]">
                {item.result}
              </span>
            </Link>
          </li>
        ))}
      </ul>
    </section>
  )
}
