/**
 * Turning a past run back into a task the user can launch again.
 *
 * Lives outside the component that renders it so that file exports components only -- Fast Refresh
 * cannot preserve state in a module that mixes the two.
 */

import type { PerfRunSummary, ReportSummary } from '@/api/types'
import { primaryMetricsOf } from '@/domain/report/primaryMetrics'

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
