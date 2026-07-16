import type { ReactNode } from 'react'
import { useLocale } from '@/contexts/LocaleContext'
import type { PerfMetrics, PercentileStats } from '@/api/types'
import { cn } from '@/lib/utils'
import { formatMetric } from '@/domain/metric/metricFormat'
import { DEFAULT_METRIC_SPEC } from '@/domain/metric/MetricDisplaySpec'

/** Translate function contract shared with `formatMetric`. */
type Translate = (key: string) => string

/**
 * Identity translate used for the percentile / token tables: their units are
 * appended in JSX (or are unit-less), so no locale unit lookup is needed. The
 * value still flows through the centralized `formatMetric` primitive so
 * precision and round-half-up stay consistent with every other surface,
 * eliminating direct domain `toFixed` calls (Req 1.6, 15.5).
 */
const RAW_TRANSLATE: Translate = (key: string) => key

/**
 * Format a raw performance value at a fixed precision through the shared
 * `formatMetric` primitive (unbounded, no percentage conversion, round half up).
 * This replaces scattered `value.toFixed(n)` calls with the single centralized
 * formatting entry point (Req 1.4, 1.6, 15.5).
 */
function fmtRaw(value: number | null | undefined, precision: number, t: Translate = RAW_TRANSLATE): string {
  return formatMetric(value, { ...DEFAULT_METRIC_SPEC, rawPrecision: precision }, t).primary
}

interface PerfMetricsPanelProps {
  perfMetrics: PerfMetrics
}

// ── Design tokens (match CSS vars) ───────────────────────────────────────────
const C_LATENCY = 'var(--chart-latency)'
const C_TTFT    = 'var(--chart-ttft)'
const C_TPOT    = 'var(--chart-tpot)'
const C_TOKEN   = 'var(--chart-token)'

// ── Percentile table ──────────────────────────────────────────────────────────

interface PercTableProps {
  stats: PercentileStats
  unit: string
  accentCol?: string
  /** multiply values before display (e.g. ×1000 for ms) */
  scale?: number
}

function PercTable({ stats, unit, accentCol = 'var(--accent)', scale = 1 }: PercTableProps) {
  const fmt = (v: number | null) => fmtRaw(v === null ? null : v * scale, scale === 1000 ? 1 : 3)

  const cols: { label: string; key: keyof PercentileStats; accent?: boolean }[] = [
    { label: 'Mean', key: 'mean', accent: true },
    { label: 'Std',  key: 'std' },
    { label: 'Min',  key: 'min' },
    { label: 'P50',  key: '50%', accent: true },
    { label: 'P75',  key: '75%' },
    { label: 'P90',  key: '90%' },
    { label: 'P99',  key: '99%', accent: true },
    { label: 'Max',  key: 'max' },
  ]

  return (
    <table className="w-full border-collapse">
      <thead>
        <tr>
          {cols.map((c) => (
            <th
              key={c.label}
              className="type-table-xs px-2 py-1 text-right border-b border-[var(--border)] whitespace-nowrap"
              style={c.accent ? { color: accentCol } : undefined}
            >
              {c.label}
            </th>
          ))}
        </tr>
      </thead>
      <tbody>
        <tr>
          {cols.map((c) => (
            <td
              key={c.label}
              className={cn(
                'type-body-sm tabular-nums px-2 py-1.5 text-right whitespace-nowrap',
                c.accent ? 'text-[var(--text)] font-medium' : 'text-[var(--text-muted)]',
              )}
            >
              {fmt(stats[c.key])}{stats[c.key] === null ? '' : unit}
            </td>
          ))}
        </tr>
      </tbody>
    </table>
  )
}

// ── Metric section ────────────────────────────────────────────────────────────

interface MetricSectionProps {
  color: string
  dot?: boolean
  label: string
  sublabel?: string
  children: ReactNode
}

function MetricSection({ color, label, sublabel, children }: MetricSectionProps) {
  return (
    <div className="flex flex-col gap-2">
      <div className="flex items-center gap-2">
        <div
          className="w-[3px] h-4 rounded-[2px] shrink-0"
          style={{ background: color }}
        />
        <span className="type-body-sm-strong text-[var(--text)]">{label}</span>
        {sublabel && (
          <span className="type-body-xs text-[var(--text-muted)]">{sublabel}</span>
        )}
      </div>
      <div
        className="ml-px pl-3.5 border-l-2"
        style={{ borderColor: `${color}33` }}
      >
        {children}
      </div>
    </div>
  )
}

// ── Token usage table ─────────────────────────────────────────────────────────

interface TokenTableProps {
  usage: {
    input_tokens: PercentileStats
    output_tokens: PercentileStats
    total_tokens: PercentileStats
    total_input_tokens?: number
    total_output_tokens?: number
    total_tokens_count?: number
  }
  labels: { input: string; output: string; total: string; totalCount: string }
}

function TokenTable({ usage, labels }: TokenTableProps) {
  const rows = [
    { label: labels.input,  stats: usage.input_tokens,  count: usage.total_input_tokens },
    { label: labels.output, stats: usage.output_tokens, count: usage.total_output_tokens },
    { label: labels.total,  stats: usage.total_tokens,  count: usage.total_tokens_count },
  ]

  // whether any total counts are available (new-format reports only)
  const hasCount = rows.some((r) => r.count !== undefined)

  const headers = ['', 'Mean', '±Std', 'P50', 'P99', 'Min', 'Max', ...(hasCount ? [labels.totalCount] : [])]

  const cellBase = 'type-body-sm tabular-nums px-2 py-1.5 text-right whitespace-nowrap'

  return (
    <table className="w-full border-collapse">
      <thead>
        <tr>
          {headers.map((h) => (
            <th
              key={h}
              className={cn(
                'type-table-xs px-2 py-1 border-b border-[var(--border)] whitespace-nowrap',
                h === '' ? 'text-left' : 'text-right',
                h === labels.totalCount && 'text-[var(--text)]',
              )}
            >
              {h}
            </th>
          ))}
        </tr>
      </thead>
      <tbody>
        {rows.map((row, i) => (
          <tr
            key={row.label}
            className={i < rows.length - 1 ? 'border-b border-[var(--border)]' : ''}
          >
            <td className={cn(cellBase, 'text-left text-[var(--text-muted)] font-medium')}>
              {row.label}
            </td>
            <td className={cn(cellBase, 'text-[var(--text)] font-medium')}>
              {fmtRaw(row.stats.mean, 0)}
            </td>
            <td className={cn(cellBase, 'text-[var(--text-muted)]')}>
              {fmtRaw(row.stats.std, 0)}
            </td>
            <td className={cn(cellBase, 'text-[var(--text-muted)]')}>
              {fmtRaw(row.stats['50%'], 0)}
            </td>
            <td className={cn(cellBase, 'text-[var(--text-muted)]')}>
              {fmtRaw(row.stats['99%'], 0)}
            </td>
            <td className={cn(cellBase, 'text-[var(--text-muted)]')}>
              {fmtRaw(row.stats.min, 0)}
            </td>
            <td className={cn(cellBase, 'text-[var(--text-muted)]')}>
              {fmtRaw(row.stats.max, 0)}
            </td>
            {hasCount && (
              <td className={cn(cellBase, 'text-[var(--text)] font-semibold')}>
                {row.count !== undefined ? row.count.toLocaleString() : '—'}
              </td>
            )}
          </tr>
        ))}
      </tbody>
    </table>
  )
}

// ── Overview KPI strip ────────────────────────────────────────────────────────

function KpiStrip({
  items,
}: {
  items: { label: string; value: string; color: string }[]
}) {
  return (
    <div className="grid grid-cols-2 sm:grid-cols-3 xl:grid-cols-4 gap-px bg-[var(--border)] border border-[var(--border)] rounded-[var(--radius-sm)] overflow-hidden">
      {items.map((item) => (
        <div
          key={item.label}
          className="min-w-0 bg-[var(--bg-card)] px-3 py-2.5"
        >
          <div
            className="text-lg font-semibold tabular-nums leading-tight"
            style={{ color: item.color }}
          >
            {item.value}
          </div>
          <div className="text-[10px] text-[var(--text-muted)] mt-0.5 break-words">
            {item.label}
          </div>
        </div>
      ))}
    </div>
  )
}

// ── Separator ─────────────────────────────────────────────────────────────────

function Sep() {
  return <div className="h-px bg-[var(--border)]" />
}

// ── Main ──────────────────────────────────────────────────────────────────────

export default function PerfMetricsPanel({ perfMetrics }: PerfMetricsPanelProps) {
  const { t } = useLocale()
  const { n_samples, latency, throughput, usage, ttft, tpot } = perfMetrics.summary

  const kpis = [
    {
      label: t('reportDetail.samples'),
      value: String(n_samples),
      color: 'var(--text)',
    },
    {
      label: t('reportDetail.avgLatency'),
      value: `${fmtRaw(latency.mean, 3, t)}s`,
      color: C_LATENCY,
    },
    ...(ttft
      ? [{ label: t('reportDetail.ttft'), value: `${fmtRaw(ttft.mean * 1000, 0, t)}ms`, color: C_TTFT }]
      : []),
    ...(tpot
      ? [{ label: t('reportDetail.tpot'), value: `${fmtRaw(tpot.mean * 1000, 0, t)}ms`, color: C_TPOT }]
      : []),
    {
      label: t('reportDetail.outputTps'),
      value: `${fmtRaw(throughput.avg_output_tps, 1, t)} tok/s`,
      color: 'var(--text)',
    },
    ...(usage.total_input_tokens !== undefined
      ? [{ label: t('reportDetail.totalInputTokens'), value: usage.total_input_tokens.toLocaleString(), color: 'var(--text)' }]
      : []),
    ...(usage.total_output_tokens !== undefined
      ? [{ label: t('reportDetail.totalOutputTokens'), value: usage.total_output_tokens.toLocaleString(), color: 'var(--text)' }]
      : []),
  ]

  return (
    <div className="flex flex-col gap-4">

      {/* Overview strip */}
      <KpiStrip items={kpis} />

      <Sep />

      {/* Latency distribution */}
      <MetricSection color={C_LATENCY} label={t('reportDetail.latencyDist')} sublabel="(s)">
        <div className="overflow-x-auto">
          <PercTable stats={latency} unit="s" accentCol={C_LATENCY} />
        </div>
      </MetricSection>

      {/* TTFT — only when streaming */}
      {ttft && (
        <>
          <Sep />
          <MetricSection
            color={C_TTFT}
            label={t('reportDetail.ttft')}
            sublabel={`${t('reportDetail.ttftDesc')} (ms)`}
          >
            <div className="overflow-x-auto">
              <PercTable stats={ttft} unit="ms" accentCol={C_TTFT} scale={1000} />
            </div>
          </MetricSection>
        </>
      )}

      {/* TPOT — only when streaming */}
      {tpot && (
        <>
          <Sep />
          <MetricSection
            color={C_TPOT}
            label={t('reportDetail.tpot')}
            sublabel={`${t('reportDetail.tpotDesc')} (ms)`}
          >
            <div className="overflow-x-auto">
              <PercTable stats={tpot} unit="ms" accentCol={C_TPOT} scale={1000} />
            </div>
          </MetricSection>
        </>
      )}

      <Sep />

      {/* Token usage */}
      <MetricSection color={C_TOKEN} label={t('reportDetail.tokenUsage')} sublabel="(tokens)">
        <div className="overflow-x-auto">
          <TokenTable
            usage={usage}
            labels={{
              input:  t('reportDetail.tokenInput'),
              output: t('reportDetail.tokenOutput'),
              total:  t('reportDetail.tokenTotal'),
              totalCount: t('reportDetail.totalCount'),
            }}
          />
        </div>
      </MetricSection>

    </div>
  )
}
