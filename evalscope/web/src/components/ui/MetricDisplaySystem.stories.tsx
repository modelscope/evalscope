import type { Meta, StoryObj } from '@storybook/react-vite'
import { useLocale } from '@/contexts/LocaleContext'
import { formatMetricByKey } from '@/domain/metric/registry'
import { MISSING_PLACEHOLDER } from '@/domain/metric/metricFormat'

/**
 * Storybook stories for the Metric_Display_System (Req 14.3).
 *
 * There is no single "metric display" component; the system is the centralized
 * `formatMetricByKey` entry point (`src/domain/metric/registry.ts`) that every
 * surface renders through. To give the visual-regression baseline a concrete
 * anchor, this file defines a small presentational `MetricDisplay` that renders
 * the `FormattedMetric` produced by that entry point, exposing every display
 * form the contract distinguishes:
 *
 * - Bounded ratio → percentage primary + 4-decimal raw (Req 1.2, 1.3).
 * - Unbounded metric → native unit, never a percentage (Req 1.4, 1.5, 1.7).
 * - Missing value → a distinct placeholder, never `0`/blank (Req 1.8).
 * - Undefined spec → 4-decimal raw with an "undefined display form" flag (Req 1.13).
 */

interface MetricDisplayProps {
  /** Raw/implementation-level metric key (canonical key or a known alias). */
  metricKey: string
  /** Raw metric value; `null` / `undefined` / `NaN` render as missing. */
  value: number | null | undefined
}

/**
 * Lightweight presentational wrapper over `formatMetricByKey`. It shows the
 * primary display string prominently plus the secondary raw value, unit and the
 * `isMissing` / `isSpecUndefined` flags so each metric form is visually distinct.
 */
function MetricDisplay({ metricKey, value }: MetricDisplayProps) {
  const { t } = useLocale()
  const formatted = formatMetricByKey(metricKey, value, t)

  return (
    <div className="inline-flex min-w-[220px] flex-col gap-3 rounded-[var(--radius)] border border-[var(--border)] bg-[var(--bg-card)] p-4">
      <code className="text-xs text-[var(--text-muted)]">{metricKey}</code>
      <span
        className={
          formatted.isMissing
            ? 'text-3xl font-bold tabular-nums text-[var(--text-muted)]'
            : 'text-3xl font-bold tabular-nums text-[var(--text)]'
        }
      >
        {formatted.primary}
      </span>
      <dl className="grid grid-cols-[auto_1fr] gap-x-3 gap-y-1 text-xs text-[var(--text-muted)]">
        <dt>raw</dt>
        <dd className="tabular-nums text-[var(--text)]">{formatted.raw}</dd>
        <dt>unit</dt>
        <dd className="text-[var(--text)]">{formatted.unitLabel || MISSING_PLACEHOLDER}</dd>
        <dt>isMissing</dt>
        <dd className="text-[var(--text)]">{String(formatted.isMissing)}</dd>
        <dt>specUndefined</dt>
        <dd className="text-[var(--text)]">{String(formatted.isSpecUndefined)}</dd>
      </dl>
    </div>
  )
}

const meta = {
  title: 'Metric Display System/FormattedMetric',
  component: MetricDisplay,
  parameters: {
    layout: 'centered',
  },
  args: {
    metricKey: 'accuracy',
    value: 0.9205,
  },
} satisfies Meta<typeof MetricDisplay>

export default meta

type Story = StoryObj<typeof meta>

/** Bounded ratio: percentage primary (1 decimal) + 4-decimal raw (Req 1.2, 1.3). */
export const BoundedRatioPercent: Story = {
  args: { metricKey: 'accuracy', value: 0.9205 },
}

/** Unbounded metric: keeps its native unit, no percentage conversion (Req 1.4, 1.7). */
export const UnboundedWithUnit: Story = {
  args: { metricKey: 'throughput', value: 1234.5 },
}

/** A value > 1 is never coerced to a percentage for unbounded metrics (Req 1.5). */
export const UnboundedGreaterThanOne: Story = {
  args: { metricKey: 'latency', value: 12.5 },
}

/** Missing value: distinct placeholder, never rendered as `0` or blank (Req 1.8). */
export const MissingValue: Story = {
  args: { metricKey: 'accuracy', value: null },
}

/** Unknown metric: undefined display form, 4-decimal raw, no inference (Req 1.13). */
export const SpecUndefined: Story = {
  args: { metricKey: 'unknown_metric', value: 42 },
}

/** Gallery of every distinguished display form side by side. */
export const Gallery: Story = {
  render: () => (
    <div className="flex flex-wrap gap-4">
      <MetricDisplay metricKey="accuracy" value={0.9205} />
      <MetricDisplay metricKey="success_rate" value={100} />
      <MetricDisplay metricKey="throughput" value={1234.5} />
      <MetricDisplay metricKey="latency" value={12.5} />
      <MetricDisplay metricKey="accuracy" value={null} />
      <MetricDisplay metricKey="unknown_metric" value={42} />
    </div>
  ),
}
