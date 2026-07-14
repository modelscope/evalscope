import { useMemo } from 'react'
import { useLocale } from '@/contexts/LocaleContext'
import { useReports } from '@/contexts/ReportsContext'
import { useTheme } from '@/contexts/ThemeContext'
import { useQueryParams } from '@/hooks/useQueryParams'
import { getPerfCompareChartUrl } from '@/api/perf'
import Breadcrumb from '@/components/ui/Breadcrumb'
import Card from '@/components/ui/Card'
import PlotlyChart from '@/components/charts/PlotlyChart'
import { GitCompareArrows } from 'lucide-react'

// Sweep charts grouped like the single-run detail page.
const LATENCY_CHARTS = ['latency', 'ttft', 'tpot'] as const
const THROUGHPUT_CHARTS = ['rps', 'throughput', 'success'] as const

const CHART_TITLES: Record<string, string> = {
  latency: 'Latency (s)',
  ttft: 'TTFT (ms)',
  tpot: 'TPOT (ms)',
  rps: 'Requests/sec',
  throughput: 'Tokens/sec',
  success: 'Success Rate (%)',
}

export default function PerfComparePage() {
  const { t } = useLocale()
  const { get } = useQueryParams()
  const { rootPath: ctxRoot } = useReports()
  const { theme } = useTheme()

  const rootPath = get('root_path') ?? ctxRoot
  const paths = useMemo(
    () => (get('paths') ?? '').split(';').map((p) => p.trim()).filter(Boolean),
    [get],
  )

  if (paths.length < 2) {
    return (
      <div className="page-enter flex flex-col gap-4">
        <Breadcrumb
          items={[
            { label: t('nav.performance'), href: `/performance?root_path=${encodeURIComponent(rootPath)}` },
            { label: t('performance.comparePageTitle') },
          ]}
        />
        <div className="py-16 text-center type-body-sm text-[var(--text-muted)]">
          {t('performance.selectToCompare')}
        </div>
      </div>
    )
  }

  return (
    <div className="page-enter flex flex-col gap-4">
      <Breadcrumb
        items={[
          { label: t('nav.performance'), href: `/performance?root_path=${encodeURIComponent(rootPath)}` },
          { label: t('performance.comparePageTitle') },
        ]}
      />

      {/* Header */}
      <div className="flex items-start gap-3 rounded-[var(--radius)] border border-[var(--border)] bg-[var(--bg-card)] p-5">
        <span className="text-[var(--accent)] shrink-0 mt-0.5">
          <GitCompareArrows size={20} />
        </span>
        <div className="flex flex-col gap-1 min-w-0">
          <h1 className="type-title-md text-[var(--text)]">
            {t('performance.comparing').replace('${n}', String(paths.length))}
          </h1>
          <div className="type-caption-mono text-[var(--text-muted)] break-all">{paths.join('  ·  ')}</div>
        </div>
      </div>

      {/* Latency group */}
      <Card title={t('performance.latencyGroup')}>
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          {LATENCY_CHARTS.map((ct) => (
            <PlotlyChart
              key={ct}
              src={getPerfCompareChartUrl(rootPath, paths, ct, theme)}
              title={CHART_TITLES[ct]}
              height={340}
            />
          ))}
        </div>
      </Card>

      {/* Throughput group */}
      <Card title={t('performance.throughputGroup')}>
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          {THROUGHPUT_CHARTS.map((ct) => (
            <PlotlyChart
              key={ct}
              src={getPerfCompareChartUrl(rootPath, paths, ct, theme)}
              title={CHART_TITLES[ct]}
              height={340}
            />
          ))}
        </div>
      </Card>
    </div>
  )
}
