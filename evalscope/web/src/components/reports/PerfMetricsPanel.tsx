import type { ReactNode } from 'react'
import { useLocale } from '@/contexts/LocaleContext'
import type { PerfMetrics, PercentileStats } from '@/api/types'
import { cn } from '@/lib/utils'

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
  const fmt = (v: number) => (v * scale).toFixed(scale === 1000 ? 1 : 3)

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
              {fmt(stats[c.key] as number)}{unit}
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
              {row.stats.mean.toFixed(0)}
            </td>
            <td className={cn(cellBase, 'text-[var(--text-muted)]')}>
              {row.stats.std.toFixed(0)}
            </td>
            <td className={cn(cellBase, 'text-[var(--text-muted)]')}>
              {row.stats['50%'].toFixed(0)}
            </td>
            <td className={cn(cellBase, 'text-[var(--text-muted)]')}>
              {row.stats['99%'].toFixed(0)}
            </td>
            <td className={cn(cellBase, 'text-[var(--text-muted)]')}>
              {row.stats.min.toFixed(0)}
            </td>
            <td className={cn(cellBase, 'text-[var(--text-muted)]')}>
              {row.stats.max.toFixed(0)}
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
    <div className="flex bg-[var(--bg-card)] border border-[var(--border)] rounded-[var(--radius-sm)] overflow-hidden">
      {items.map((item, i) => (
        <div
          key={item.label}
          className={cn(
            'flex-1 px-3 py-2.5',
            i < items.length - 1 && 'border-r border-[var(--border)]',
          )}
        >
          <div
            className="text-lg font-semibold tabular-nums leading-tight"
            style={{ color: item.color }}
          >
            {item.value}
          </div>
          <div className="text-[10px] text-[var(--text-muted)] mt-0.5 whitespace-nowrap">
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
      value: `${latency.mean.toFixed(3)}s`,
      color: C_LATENCY,
    },
    ...(ttft
      ? [{ label: t('reportDetail.ttft'), value: `${(ttft.mean * 1000).toFixed(0)}ms`, color: C_TTFT }]
      : []),
    ...(tpot
      ? [{ label: t('reportDetail.tpot'), value: `${(tpot.mean * 1000).toFixed(0)}ms`, color: C_TPOT }]
      : []),
    {
      label: t('reportDetail.outputTps'),
      value: `${throughput.avg_output_tps.toFixed(1)} tok/s`,
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
