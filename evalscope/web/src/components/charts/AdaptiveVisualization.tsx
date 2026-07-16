import { useMemo, type ReactNode } from 'react'
import { cn } from '@/lib/utils'
import { useLocale } from '@/contexts/LocaleContext'
import { scoreColor } from '@/utils/colorScale'
import EmptyState from '@/components/common/EmptyState'
import PlotlyChart from '@/components/charts/PlotlyChart'
import { selectVisualization, type VizKind } from '@/domain/chart/adaptiveVisualization'
import { formatMetricByKey } from '@/domain/metric/registry'

/** A single normalized dimension driving the visualization. */
export interface VizDimension {
  /** Category label shown as an axis / bar / value label. */
  label: string
  /** Normalized value in the `[0, 1]` range. */
  value: number
}

export interface AdaptiveVisualizationProps {
  /**
   * Normalized dimensions of the report. Their count selects the visualization
   * form via {@link selectVisualization}.
   */
  dimensions: VizDimension[]
  /**
   * Radar chart iframe source, used only when there are >= 3 dimensions. The
   * caller injects the theme (see `ChartFrame`/`getChartUrl`); this component
   * does not construct the URL.
   */
  radarSrc?: string
  title?: string
  height?: number
  className?: string
}

/** Clamp a normalized value into `[0, 1]`. */
function clamp01(value: number): number {
  if (!Number.isFinite(value)) return 0
  return Math.max(0, Math.min(1, value))
}

/** Bordered card shell matching `PlotlyChart`'s frame for non-iframe variants. */
function ChartShell({
  title,
  ariaLabel,
  height,
  className,
  children,
}: {
  title?: string
  ariaLabel?: string
  height?: number
  className?: string
  children: ReactNode
}) {
  return (
    <div
      className={cn(
        'rounded-[var(--radius)] border border-[var(--border)] bg-[var(--bg-card)] overflow-hidden',
        className,
      )}
    >
      {title && (
        <div className="px-4 py-2.5 border-b border-[var(--border)]">
          <h4 className="type-label-xs">{title}</h4>
        </div>
      )}
      <div
        role="img"
        aria-label={ariaLabel}
        className="flex flex-col justify-center p-4"
        style={height ? { minHeight: height } : undefined}
      >
        {children}
      </div>
    </div>
  )
}

/**
 * Adaptive report visualization (Chart_Renderer, Req 3.1–3.5).
 *
 * The visualization form is chosen from the number of normalized dimensions via
 * {@link selectVisualization}:
 * - 0 dimensions → empty state, no chart is rendered (Req 3.5).
 * - 1 dimension  → the normalized value plus its category label; never a
 *   single-axis radar (Req 3.1).
 * - 2 dimensions → a grouped bar / dot plot; never a radar (Req 3.2).
 * - >= 3 dimensions → a radar chart with one axis per dimension (Req 3.3).
 */
export default function AdaptiveVisualization({
  dimensions,
  radarSrc,
  title,
  height = 400,
  className,
}: AdaptiveVisualizationProps) {
  const { t } = useLocale()
  const formatPercent = (value: number): string => formatMetricByKey('score', clamp01(value), t).primary

  const kind: VizKind = useMemo(() => selectVisualization(dimensions.length), [dimensions.length])

  // 0 dimensions → empty state, no chart (Req 3.5).
  if (kind === 'empty') {
    return (
      <ChartShell title={title} height={height} className={className}>
        <EmptyState title={t('charts.emptyTitle')} hint={t('common.noData')} />
      </ChartShell>
    )
  }

  // 1 dimension → prominent normalized value + category label (Req 3.1).
  if (kind === 'single-value') {
    const dim = dimensions[0]
    const color = scoreColor(clamp01(dim.value))
    return (
      <ChartShell
        title={title}
        ariaLabel={t('charts.singleValueAriaLabel')}
        height={height}
        className={className}
      >
        <div className="flex flex-col items-center justify-center gap-2 text-center">
          <span className="text-3xl font-mono font-semibold tabular-nums" style={{ color }}>
            {formatPercent(dim.value)}
          </span>
          <span className="type-body-sm text-[var(--text-muted)] break-words min-w-0">
            {dim.label}
          </span>
        </div>
      </ChartShell>
    )
  }

  // 2 dimensions → grouped bar / dot plot, never a radar (Req 3.2).
  if (kind === 'grouped-bar') {
    return (
      <ChartShell
        title={title}
        ariaLabel={t('charts.groupedBarAriaLabel')}
        height={height}
        className={className}
      >
        <div className="flex flex-col gap-4">
          {dimensions.map((dim, index) => {
            const norm = clamp01(dim.value)
            return (
              <div key={`${dim.label}-${index}`} className="flex flex-col gap-1.5">
                <div className="flex items-baseline justify-between gap-3">
                  <span className="type-body-sm text-[var(--text)] break-words min-w-0">
                    {dim.label}
                  </span>
                  <span
                    className="type-body-sm font-mono font-medium tabular-nums shrink-0"
                    style={{ color: scoreColor(norm) }}
                  >
                    {formatPercent(dim.value)}
                  </span>
                </div>
                <div className="h-2 w-full rounded-full bg-[var(--border)] overflow-hidden">
                  <div
                    className="h-full rounded-full transition-all duration-300"
                    style={{ width: `${norm * 100}%`, background: scoreColor(norm) }}
                  />
                </div>
              </div>
            )
          })}
        </div>
      </ChartShell>
    )
  }

  // >= 3 dimensions → radar is the preferred chart (Req 3.3).
  // The radar itself is rendered by the theme-aware chart iframe. When no radar
  // source is supplied we fall back to an empty state rather than a blank frame.
  if (!radarSrc) {
    return (
      <ChartShell title={title} height={height} className={className}>
        <EmptyState title={t('common.noData')} />
      </ChartShell>
    )
  }

  return (
    <PlotlyChart
      src={radarSrc}
      title={title}
      height={height}
      className={className}
      fallbackTable={{
        columns: ['Dimension', 'Score'],
        rows: dimensions.map((dimension) => ({ Dimension: dimension.label, Score: dimension.value })),
        scoreColumns: ['Score'],
      }}
    />
  )
}
