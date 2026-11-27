import { cn } from '@/lib/utils'
import { useLocale } from '@/contexts/LocaleContext'
import type { ReportData } from '@/api/types'
import { datasetLabel, primaryMetricOf, reportSampleCount } from '@/domain/report/primaryMetrics'
import { formatMetric, formatMetricIdentityLabel, getBoundedQualityRatio } from '@/domain/metric'
import { scoreColor } from '@/utils/colorScale'

interface Props {
  reports: ReportData[]
  active: string
  onChange: (dataset: string) => void
}

export default function DatasetNav({ reports, active, onChange }: Props) {
  const { t } = useLocale()
  return (
    <nav
      aria-label={t('reportDetail.datasets')}
      className="sticky top-4 flex max-h-[80vh] w-52 shrink-0 flex-col gap-1 overflow-y-auto rounded-[var(--radius)] border border-[var(--border)] bg-[var(--bg-card)] p-3"
    >
      <h2 className="border-b border-[var(--border)] px-3 pb-3 text-sm font-semibold">{t('reportDetail.datasets')}</h2>
      {reports.map((report) => {
        const primary = primaryMetricOf(report)
        const quality = getBoundedQualityRatio(primary?.score, primary?.semantics)
        return (
          <button
            key={report.dataset_name}
            type="button"
            aria-pressed={report.dataset_name === active}
            onClick={() => onChange(report.dataset_name)}
            className={cn(
              'min-h-11 rounded-[var(--radius-sm)] px-3 py-4 text-left transition-colors',
              report.dataset_name === active
                ? 'bg-[var(--accent-dim)] text-[var(--accent)]'
                : 'text-[var(--text)] hover:bg-[var(--bg-card2)]',
            )}
          >
            <span className="block break-words text-sm font-semibold" title={report.dataset_name}>
              {datasetLabel(report)}
            </span>
            <span className="mt-1 flex flex-wrap items-baseline gap-x-2 text-xs text-[var(--text-muted)]">
              {primary && (
                <span>{formatMetricIdentityLabel(primary.identity, primary.semantics, primary.legacy_name)}</span>
              )}
              <span
                className="font-mono font-semibold"
                style={{ color: quality == null ? undefined : scoreColor(quality) }}
              >
                {formatMetric(primary?.score, primary?.semantics).primary}
              </span>
            </span>
            <span className="mt-1 block text-xs text-[var(--text-muted)]">
              {t('diagnosis.sampleCount', { count: reportSampleCount(report) })}
            </span>
          </button>
        )
      })}
    </nav>
  )
}
