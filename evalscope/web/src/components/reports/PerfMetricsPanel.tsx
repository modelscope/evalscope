import { useState } from 'react'
import { useLocale } from '@/contexts/LocaleContext'
import type { PerfMetrics, PercentileStats } from '@/api/types'
import { ChevronRight } from 'lucide-react'

interface PerfMetricsPanelProps {
  perfMetrics: PerfMetrics
}

function MiniStat({ value, label }: { value: string; label: string }) {
  return (
    <div
      style={{
        background: 'var(--bg-card2)',
        border: '1px solid var(--border)',
        borderRadius: 'var(--radius-sm)',
        padding: '0.75rem',
        textAlign: 'center',
      }}
    >
      <div
        style={{
          fontSize: '1.25rem',
          fontWeight: 700,
          color: 'var(--accent)',
          fontVariantNumeric: 'tabular-nums',
        }}
      >
        {value}
      </div>
      <div
        style={{
          fontSize: '0.7rem',
          color: 'var(--text-muted)',
          textTransform: 'uppercase',
          letterSpacing: '0.05em',
          marginTop: '0.25rem',
        }}
      >
        {label}
      </div>
    </div>
  )
}

function TokenLine({ label, stats }: { label: string; stats: PercentileStats }) {
  return (
    <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)', fontVariantNumeric: 'tabular-nums' }}>
      <span style={{ color: 'var(--text)', fontWeight: 500, width: '4rem', display: 'inline-block' }}>
        {label}
      </span>
      mean {stats.mean.toFixed(1)} ± {stats.std.toFixed(1)}
      <span style={{ opacity: 0.6 }}>
        {' '}
        (min {stats.min.toFixed(0)} ~ max {stats.max.toFixed(0)})
      </span>
    </div>
  )
}

export default function PerfMetricsPanel({ perfMetrics }: PerfMetricsPanelProps) {
  const { t } = useLocale()
  const [tokenOpen, setTokenOpen] = useState(false)

  const { n_samples, latency, throughput, usage } = perfMetrics.summary

  const percentiles: { label: string; value: number }[] = [
    { label: 'Min', value: latency.min },
    { label: 'P25', value: latency['25%'] },
    { label: 'P50', value: latency['50%'] },
    { label: 'P75', value: latency['75%'] },
    { label: 'P90', value: latency['90%'] },
    { label: 'P99', value: latency['99%'] },
    { label: 'Max', value: latency.max },
    { label: 'Std', value: latency.std },
  ]

  const thStyle: React.CSSProperties = {
    fontSize: '0.65rem',
    color: 'var(--text-muted)',
    textTransform: 'uppercase',
    letterSpacing: '0.05em',
    fontWeight: 600,
    padding: '0.4rem 0.6rem',
    textAlign: 'center',
    borderBottom: '1px solid var(--border)',
  }

  const tdStyle: React.CSSProperties = {
    fontSize: '0.85rem',
    fontVariantNumeric: 'tabular-nums',
    color: 'var(--text)',
    padding: '0.5rem 0.6rem',
    textAlign: 'center',
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
      {/* KPI Cards */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '0.75rem' }}>
        <MiniStat value={latency.mean.toFixed(2) + 's'} label={t('reportDetail.avgLatency')} />
        <MiniStat value={throughput.avg_output_tps.toFixed(1)} label={t('reportDetail.outputTps')} />
        <MiniStat value={throughput.avg_req_ps.toFixed(2)} label={t('reportDetail.reqPerSec')} />
        <MiniStat value={String(n_samples)} label={t('reportDetail.samples')} />
      </div>

      {/* Latency Distribution */}
      <div>
        <div
          style={{
            fontSize: '0.7rem',
            color: 'var(--text-muted)',
            textTransform: 'uppercase',
            letterSpacing: '0.05em',
            fontWeight: 600,
            marginBottom: '0.5rem',
          }}
        >
          {t('reportDetail.latencyDist')}
        </div>
        <div
          style={{
            background: 'var(--bg-deep)',
            borderRadius: 'var(--radius-sm)',
            border: '1px solid var(--border)',
            overflow: 'hidden',
          }}
        >
          <table style={{ width: '100%', borderCollapse: 'collapse' }}>
            <thead>
              <tr>
                {percentiles.map((p) => (
                  <th key={p.label} style={thStyle}>
                    {p.label}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              <tr>
                {percentiles.map((p) => (
                  <td key={p.label} style={tdStyle}>
                    {p.value.toFixed(2)}s
                  </td>
                ))}
              </tr>
            </tbody>
          </table>
        </div>
      </div>

      {/* Token Usage (collapsible) */}
      <div>
        <button
          onClick={() => setTokenOpen((o) => !o)}
          style={{
            cursor: 'pointer',
            color: 'var(--text-muted)',
            fontSize: '0.7rem',
            fontWeight: 600,
            textTransform: 'uppercase',
            letterSpacing: '0.05em',
            background: 'none',
            border: 'none',
            padding: 0,
            display: 'flex',
            alignItems: 'center',
            gap: '0.3rem',
          }}
        >
          <ChevronRight
            size={12}
            style={{
              transition: 'transform 0.15s ease',
              transform: tokenOpen ? 'rotate(90deg)' : 'rotate(0deg)',
            }}
          />
          {t('reportDetail.tokenUsage')}
        </button>
        {tokenOpen && (
          <div style={{ marginTop: '0.5rem', display: 'flex', flexDirection: 'column', gap: '0.35rem' }}>
            <TokenLine label="Input" stats={usage.input_tokens} />
            <TokenLine label="Output" stats={usage.output_tokens} />
            <TokenLine label="Total" stats={usage.total_tokens} />
          </div>
        )}
      </div>
    </div>
  )
}
