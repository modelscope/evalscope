import { useEffect, useMemo, useState } from 'react'
import { useLocale } from '@/contexts/LocaleContext'
import { useReports } from '@/contexts/ReportsContext'
import { useTheme } from '@/contexts/ThemeContext'
import { useQueryParams } from '@/hooks/useQueryParams'
import { getPerfCompareChartUrl, getPerfDetail } from '@/api/perf'
import Breadcrumb from '@/components/ui/Breadcrumb'
import Card from '@/components/ui/Card'
import Skeleton from '@/components/ui/Skeleton'
import PlotlyChart from '@/components/charts/PlotlyChart'
import { LATENCY_CHARTS, THROUGHPUT_CHARTS, CHART_TITLES } from '@/utils/perf'
import { GitCompareArrows } from 'lucide-react'

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

  // Embedding/rerank runs omit TTFT/TPOT (mirrors PerfReportDetailPage). The
  // list view forwards the first run's mode via the `embedding` query param to
  // avoid a round-trip; fall back to a one-off detail fetch for direct links.
  const embeddingParam = get('embedding')
  const [fetchedEmbedding, setFetchedEmbedding] = useState<boolean | null>(null)
  const isEmbedding = embeddingParam != null ? embeddingParam === '1' : fetchedEmbedding

  const firstPath = paths[0] ?? ''
  useEffect(() => {
    if (embeddingParam != null || !firstPath) return
    let cancelled = false
    const probe = async () => {
      setFetchedEmbedding(null)
      try {
        const res = await getPerfDetail(rootPath, firstPath)
        if (!cancelled) setFetchedEmbedding(Boolean(res.is_embedding))
      } catch {
        if (!cancelled) setFetchedEmbedding(false)
      }
    }
    probe()
    return () => {
      cancelled = true
    }
  }, [embeddingParam, rootPath, firstPath])

  // Charts available for this run mode (embedding runs have no TTFT/TPOT).
  const latencyCharts = useMemo(
    () => (isEmbedding ? (['latency'] as const) : LATENCY_CHARTS),
    [isEmbedding],
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
            {t('performance.comparing', { n: paths.length })}
          </h1>
          <div className="type-caption-mono text-[var(--text-muted)] break-all">{paths.join('  ·  ')}</div>
        </div>
      </div>

      {/* Latency group */}
      <Card title={t('performance.latencyGroup')}>
        {isEmbedding === null ? (
          <Skeleton width="100%" height={340} />
        ) : (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            {latencyCharts.map((ct) => (
              <PlotlyChart
                key={ct}
                src={getPerfCompareChartUrl(rootPath, paths, ct, theme)}
                title={CHART_TITLES[ct]}
                height={340}
              />
            ))}
          </div>
        )}
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
