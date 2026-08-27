import type { ReactNode } from 'react'
import { useLocale } from '@/contexts/LocaleContext'
import type { PerfMetrics, PercentileStats } from '@/api/types'
import { cn } from '@/lib/utils'
import { formatMetric } from '@/domain/metric'
import type { MetricSemantics } from '@/domain/metric'
import KpiStrip from '@/components/ui/KpiStrip'

function formatPerfValue(
  value: number | null | undefined,
  semantics: MetricSemantics,
): string {
  return formatMetric(value, semantics).primary
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
  semantics: MetricSemantics
  accentCol?: string
}

function PercTable({
  stats,
  semantics,
  accentCol = 'var(--accent)',
}: PercTableProps) {
  const fmt = (value: number | null | undefined) => formatPerfValue(value, semantics)

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
              {fmt(stats[c.key])}
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
    total_input_tokens?: number | null
    total_output_tokens?: number | null
    total_tokens_count?: number | null
  }
  labels: { input: string; output: string; total: string; totalCount: string }
  semantics: Record<string, MetricSemantics>
}

function TokenTable({ usage, labels, semantics }: TokenTableProps) {
  const rows = [
    { key: 'usage.input_tokens', label: labels.input,  stats: usage.input_tokens,  count: usage.total_input_tokens },
    { key: 'usage.output_tokens', label: labels.output, stats: usage.output_tokens, count: usage.total_output_tokens },
    { key: 'usage.total_tokens', label: labels.total,  stats: usage.total_tokens,  count: usage.total_tokens_count },
  ]

  // whether any total counts are available (new-format reports only)
  const hasCount = rows.some((r) => r.count != null)

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
              {formatPerfValue(row.stats.mean, semantics[row.key])}
            </td>
            <td className={cn(cellBase, 'text-[var(--text-muted)]')}>
              {formatPerfValue(row.stats.std, semantics[row.key])}
            </td>
            <td className={cn(cellBase, 'text-[var(--text-muted)]')}>
              {formatPerfValue(row.stats['50%'], semantics[row.key])}
            </td>
            <td className={cn(cellBase, 'text-[var(--text-muted)]')}>
              {formatPerfValue(row.stats['99%'], semantics[row.key])}
            </td>
            <td className={cn(cellBase, 'text-[var(--text-muted)]')}>
              {formatPerfValue(row.stats.min, semantics[row.key])}
            </td>
            <td className={cn(cellBase, 'text-[var(--text-muted)]')}>
              {formatPerfValue(row.stats.max, semantics[row.key])}
            </td>
            {hasCount && (
              <td className={cn(cellBase, 'text-[var(--text)] font-semibold')}>
                {row.count != null ? formatPerfValue(row.count, semantics[row.key]) : '—'}
              </td>
            )}
          </tr>
        ))}
      </tbody>
    </table>
  )
}

// ── Separator ─────────────────────────────────────────────────────────────────

function Sep() {
  return <div className="h-px bg-[var(--border)]" />
}

// ── Main ──────────────────────────────────────────────────────────────────────

export default function PerfMetricsPanel({ perfMetrics }: PerfMetricsPanelProps) {
  const { t } = useLocale()
  if (!perfMetrics.summary) return null

  const { n_samples, latency, throughput, usage, ttft, tpot } = perfMetrics.summary
  const semantics = perfMetrics.metric_semantics ?? {}

  const kpis = [
    {
      label: t('reportDetail.samples'),
      value: formatPerfValue(n_samples, semantics.n_samples),
      color: 'var(--text)',
    },
    {
      label: t('reportDetail.avgLatency'),
      value: formatPerfValue(latency.mean, semantics.latency),
      color: C_LATENCY,
    },
    ...(ttft
      ? [{ label: t('reportDetail.ttft'), value: formatPerfValue(ttft.mean, semantics.ttft), color: C_TTFT }]
      : []),
    ...(tpot
      ? [{ label: t('reportDetail.tpot'), value: formatPerfValue(tpot.mean, semantics.tpot), color: C_TPOT }]
      : []),
    {
      label: t('reportDetail.outputTps'),
      value: formatPerfValue(throughput.avg_output_tps, semantics['throughput.avg_output_tps']),
      color: 'var(--text)',
    },
    ...(usage.total_input_tokens !== undefined
      ? [{ label: t('reportDetail.totalInputTokens'), value: formatPerfValue(usage.total_input_tokens, semantics['usage.input_tokens']), color: 'var(--text)' }]
      : []),
    ...(usage.total_output_tokens !== undefined
      ? [{ label: t('reportDetail.totalOutputTokens'), value: formatPerfValue(usage.total_output_tokens, semantics['usage.output_tokens']), color: 'var(--text)' }]
      : []),
  ]

  return (
    <div className="flex flex-col gap-4">

      {/* Overview strip */}
      <KpiStrip items={kpis} layout="dense" />

      <Sep />

      {/* Latency distribution */}
      <MetricSection color={C_LATENCY} label={t('reportDetail.latencyDist')} sublabel="(s)">
        <div className="overflow-x-auto">
          <PercTable stats={latency} semantics={semantics.latency} accentCol={C_LATENCY} />
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
              <PercTable
                stats={ttft}
                semantics={semantics.ttft}
                accentCol={C_TTFT}
              />
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
              <PercTable
                stats={tpot}
                semantics={semantics.tpot}
                accentCol={C_TPOT}
              />
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
            semantics={semantics}
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
