import { Link } from 'react-router-dom'
import { BookOpen, FileText, Gauge, GitCompare, Play, RotateCcw } from 'lucide-react'
import { useLocale } from '@/contexts/LocaleContext'
import type { PerfRunSummary, ReportSummary } from '@/api/types'
import { primaryMetricsOf } from '@/domain/report/primaryMetrics'

/**
 * What the user can do next, kept above the results.
 *
 * The dashboard used to open with counters and a weaker copy of the Evaluations list. Nothing on it
 * was an action, so the page cost a screen of height to say what the navigation bar already said.
 * This band is the part that only a landing page can do: the four entry points, and a one-click
 * path back into a configuration that was already run once.
 */

/** A past run, reduced to the fields that can be safely put in a URL. */
interface RepeatableRun {
  kind: 'eval' | 'perf'
  model: string
  /** Comma-separated dataset list, as the task form expects it. */
  datasets: string
  timestamp: string
  /** Prefilled task URL. */
  href: string
}

/** How many past configurations to offer. */
const REPEAT_LIMIT = 2

function formatShort(timestamp: string): string {
  return timestamp ? timestamp.replace('T', ' ').slice(5, 16) : ''
}

/**
 * Build the repeatable configurations from the most recent runs.
 *
 * Only the model and the dataset list travel in the URL. An API key must never go there -- it would
 * be captured by browser history and by every proxy log on the way -- so the task form still asks
 * for it, which also gives the user a chance to adjust the run before starting it.
 */
export function repeatableRuns(reports: ReportSummary[], perfRuns: PerfRunSummary[]): RepeatableRun[] {
  const fromEval: RepeatableRun[] = reports.map((report) => {
    const datasets = primaryMetricsOf(report)
      .map((ref) => ref.dataset_name)
      .filter(Boolean)
    const list = (datasets.length > 0 ? datasets : [report.dataset_name]).join(',')
    return {
      kind: 'eval' as const,
      model: report.model_name,
      datasets: list,
      timestamp: report.timestamp || '',
      href: `/tasks?tab=eval&model=${encodeURIComponent(report.model_name)}&dataset=${encodeURIComponent(list)}`,
    }
  })

  const fromPerf: RepeatableRun[] = perfRuns.map((run) => ({
    kind: 'perf' as const,
    model: run.model,
    datasets: run.dataset || run.api_type || '',
    timestamp: run.timestamp || '',
    // The perf form takes a different shape, so only the model is handed over.
    href: `/tasks?tab=perf&model=${encodeURIComponent(run.model)}`,
  }))

  return [...fromEval, ...fromPerf]
    .filter((run) => run.model)
    .sort((a, b) => b.timestamp.localeCompare(a.timestamp))
    .slice(0, REPEAT_LIMIT)
}

interface QuickActionsProps {
  reports: ReportSummary[]
  perfRuns: PerfRunSummary[]
}

export default function QuickActions({ reports, perfRuns }: QuickActionsProps) {
  const { t } = useLocale()
  const repeats = repeatableRuns(reports, perfRuns)

  const actions = [
    { to: '/tasks?tab=eval', icon: <Play size={15} strokeWidth={2} />, label: t('dashboard.actionRunEval'), primary: true },
    { to: '/tasks?tab=perf', icon: <Gauge size={15} strokeWidth={2} />, label: t('dashboard.actionRunPerf') },
    { to: '/compare', icon: <GitCompare size={15} strokeWidth={2} />, label: t('dashboard.actionCompare') },
    { to: '/benchmarks', icon: <BookOpen size={15} strokeWidth={2} />, label: t('dashboard.actionBrowse') },
  ]

  return (
    <section
      aria-label={t('dashboard.startTitle')}
      className="rounded-[var(--radius)] border border-[var(--border)] bg-[var(--bg-card)] p-4"
    >
      <div className="flex flex-wrap gap-2">
        {actions.map((action) => (
          <Link
            key={action.to}
            to={action.to}
            className={[
              'inline-flex items-center gap-2 rounded-[var(--radius-sm)] px-3 py-2 type-body-sm transition-colors',
              action.primary
                ? 'bg-[var(--accent)] text-[var(--text-on-filled)] hover:opacity-90'
                : 'border border-[var(--border)] text-[var(--text)] hover:bg-[var(--bg-card2)]',
            ].join(' ')}
          >
            {action.icon}
            {action.label}
          </Link>
        ))}
      </div>

      {repeats.length > 0 && (
        <div className="mt-4 border-t border-[var(--border)] pt-3">
          <span className="type-caption text-[var(--text-muted)]">{t('dashboard.repeatTitle')}</span>
          <ul className="mt-2 flex flex-col gap-1">
            {repeats.map((run) => (
              <li key={`${run.kind}-${run.model}-${run.timestamp}`}>
                <Link
                  to={run.href}
                  className="group flex min-h-8 flex-wrap items-center gap-x-3 gap-y-1 rounded-[var(--radius-sm)] px-2 py-1 transition-colors hover:bg-[var(--bg-card2)]"
                >
                  <span className="shrink-0 text-[var(--text-muted)]">
                    {run.kind === 'eval' ? <FileText size={13} /> : <Gauge size={13} />}
                  </span>
                  <span className="type-body-sm text-[var(--text)]">{run.model}</span>
                  <span className="type-caption-mono min-w-0 break-words text-[var(--text-muted)]">{run.datasets}</span>
                  <span className="type-caption-mono ml-auto shrink-0 text-[var(--text-dim)]">
                    {formatShort(run.timestamp)}
                  </span>
                  <span className="inline-flex shrink-0 items-center gap-1 type-caption text-[var(--accent)]">
                    <RotateCcw size={12} />
                    {t('dashboard.repeatAction')}
                  </span>
                </Link>
              </li>
            ))}
          </ul>
        </div>
      )}
    </section>
  )
}
