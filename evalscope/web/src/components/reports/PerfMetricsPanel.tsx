import type { ReactNode, CSSProperties } from 'react'
import { useLocale } from '@/contexts/LocaleContext'
import type { PerfMetrics, PercentileStats } from '@/api/types'

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
    <table style={{ width: '100%', borderCollapse: 'collapse' }}>
      <thead>
        <tr>
          {cols.map((c) => (
            <th
              key={c.label}
              style={{
                fontSize: '0.68rem',
                fontWeight: 500,
                color: c.accent ? accentCol : 'var(--text-muted)',
                padding: '0.25rem 0.5rem',
                textAlign: 'right',
                borderBottom: '1px solid var(--border)',
                whiteSpace: 'nowrap',
              }}
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
              style={{
                fontSize: '0.82rem',
                fontVariantNumeric: 'tabular-nums',
                color: c.accent ? 'var(--text)' : 'var(--text-muted)',
                fontWeight: c.accent ? 500 : 400,
                padding: '0.3rem 0.5rem',
                textAlign: 'right',
                whiteSpace: 'nowrap',
              }}
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
    <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
        <div
          style={{
            width: '3px',
            height: '1rem',
            borderRadius: '2px',
            background: color,
            flexShrink: 0,
          }}
        />
        <span style={{ fontSize: '0.78rem', fontWeight: 600, color: 'var(--text)' }}>
          {label}
        </span>
        {sublabel && (
          <span style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>
            {sublabel}
          </span>
        )}
      </div>
      <div
        style={{
          borderLeft: `2px solid ${color}20`,
          marginLeft: '1px',
          paddingLeft: '0.85rem',
        }}
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

  const cellBase: CSSProperties = {
    fontSize: '0.8rem',
    fontVariantNumeric: 'tabular-nums',
    padding: '0.3rem 0.5rem',
    textAlign: 'right',
    whiteSpace: 'nowrap',
  }

  return (
    <table style={{ width: '100%', borderCollapse: 'collapse' }}>
      <thead>
        <tr>
          {headers.map((h) => (
            <th
              key={h}
              style={{
                fontSize: '0.68rem',
                fontWeight: 500,
                color: h === labels.totalCount ? 'var(--text)' : 'var(--text-muted)',
                padding: '0.25rem 0.5rem',
                textAlign: h === '' ? 'left' : 'right',
                borderBottom: '1px solid var(--border)',
                whiteSpace: 'nowrap',
              }}
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
            style={{ borderBottom: i < rows.length - 1 ? '1px solid var(--border)' : 'none' }}
          >
            <td style={{ ...cellBase, textAlign: 'left', color: 'var(--text-muted)', fontWeight: 500 }}>
              {row.label}
            </td>
            <td style={{ ...cellBase, color: 'var(--text)', fontWeight: 500 }}>
              {row.stats.mean.toFixed(0)}
            </td>
            <td style={{ ...cellBase, color: 'var(--text-muted)' }}>
              {row.stats.std.toFixed(0)}
            </td>
            <td style={{ ...cellBase, color: 'var(--text-muted)' }}>
              {row.stats['50%'].toFixed(0)}
            </td>
            <td style={{ ...cellBase, color: 'var(--text-muted)' }}>
              {row.stats['99%'].toFixed(0)}
            </td>
            <td style={{ ...cellBase, color: 'var(--text-muted)' }}>
              {row.stats.min.toFixed(0)}
            </td>
            <td style={{ ...cellBase, color: 'var(--text-muted)' }}>
              {row.stats.max.toFixed(0)}
            </td>
            {hasCount && (
              <td style={{ ...cellBase, color: 'var(--text)', fontWeight: 600 }}>
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
    <div
      style={{
        display: 'flex',
        gap: '0',
        background: 'var(--bg-deep)',
        border: '1px solid var(--border)',
        borderRadius: 'var(--radius-sm)',
        overflow: 'hidden',
      }}
    >
      {items.map((item, i) => (
        <div
          key={item.label}
          style={{
            flex: 1,
            padding: '0.65rem 0.75rem',
            borderRight: i < items.length - 1 ? '1px solid var(--border)' : 'none',
          }}
        >
          <div
            style={{
              fontSize: '1.1rem',
              fontWeight: 600,
              fontVariantNumeric: 'tabular-nums',
              color: item.color,
              lineHeight: 1.2,
            }}
          >
            {item.value}
          </div>
          <div
            style={{
              fontSize: '0.65rem',
              color: 'var(--text-muted)',
              marginTop: '0.15rem',
              whiteSpace: 'nowrap',
            }}
          >
            {item.label}
          </div>
        </div>
      ))}
    </div>
  )
}

// ── Separator ─────────────────────────────────────────────────────────────────

function Sep() {
  return <div style={{ height: '1px', background: 'var(--border)' }} />
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
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>

      {/* Overview strip */}
      <KpiStrip items={kpis} />

      <Sep />

      {/* Latency distribution */}
      <MetricSection color={C_LATENCY} label={t('reportDetail.latencyDist')} sublabel="(s)">
        <div style={{ overflowX: 'auto' }}>
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
            <div style={{ overflowX: 'auto' }}>
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
            <div style={{ overflowX: 'auto' }}>
              <PercTable stats={tpot} unit="ms" accentCol={C_TPOT} scale={1000} />
            </div>
          </MetricSection>
        </>
      )}

      <Sep />

      {/* Token usage */}
      <MetricSection color={C_TOKEN} label={t('reportDetail.tokenUsage')} sublabel="(tokens)">
        <div style={{ overflowX: 'auto' }}>
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
