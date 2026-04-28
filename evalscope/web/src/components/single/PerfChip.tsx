import { Timer, Zap, Activity, Hash } from 'lucide-react'
import type { SamplePerfMetrics } from '@/api/types'

interface Props {
  metrics: SamplePerfMetrics
  /** 'green' for assistant bubble (default), 'neutral' for flat section card */
  variant?: 'green' | 'neutral'
}

function fmtSec(n: number | null | undefined): string | null {
  if (n == null) return null
  return `${n.toFixed(2)}s`
}

function fmtMs(n: number | null | undefined): string | null {
  if (n == null) return null
  return `${(n * 1000).toFixed(0)}ms`
}

interface ChipItemProps {
  icon: React.ReactNode
  label: string
  value: string
  color?: string
}

function ChipItem({ icon, label, value, color }: ChipItemProps) {
  return (
    <span
      style={{
        display: 'inline-flex',
        alignItems: 'center',
        gap: '0.2rem',
        whiteSpace: 'nowrap',
        color: color ?? 'var(--text-muted)',
      }}
    >
      {icon}
      <span style={{ opacity: 0.6, marginRight: '0.1rem' }}>{label}</span>
      <span style={{ fontWeight: 500 }}>{value}</span>
    </span>
  )
}

function Dot() {
  return <span style={{ opacity: 0.25, userSelect: 'none' }}>·</span>
}

export default function PerfChip({ metrics, variant = 'green' }: Props) {
  const { latency, ttft, tpot, input_tokens, output_tokens } = metrics

  const latStr = fmtSec(latency)
  const ttftStr = fmtMs(ttft)
  const tpotStr = fmtMs(tpot)

  const hasAny = latStr || ttftStr || tpotStr || input_tokens != null || output_tokens != null
  if (!hasAny) return null

  const isGreen = variant === 'green'
  const iconSize = 10
  const iconColor = isGreen ? 'var(--bubble-bot-color)' : 'var(--text-muted)'

  const items: React.ReactNode[] = []

  if (latStr) {
    items.push(
      <ChipItem
        key="lat"
        icon={<Timer size={iconSize} color={iconColor} />}
        label="Latency"
        value={latStr}
      />
    )
  }

  if (ttftStr) {
    items.push(
      <ChipItem
        key="ttft"
        icon={<Zap size={iconSize} color={iconColor} />}
        label="TTFT"
        value={ttftStr}
      />
    )
  }

  if (tpotStr) {
    items.push(
      <ChipItem
        key="tpot"
        icon={<Activity size={iconSize} color={iconColor} />}
        label="TPOT"
        value={tpotStr}
      />
    )
  }

  if (input_tokens != null || output_tokens != null) {
    items.push(
      <span
        key="tokens"
        style={{
          display: 'inline-flex',
          alignItems: 'center',
          gap: '0.25rem',
          whiteSpace: 'nowrap',
          color: 'var(--text-muted)',
        }}
      >
        <Hash size={iconSize} color={iconColor} />
        <span style={{ opacity: 0.6 }}>in</span>
        <span style={{ fontWeight: 500 }}>{input_tokens ?? 0}</span>
        <span style={{ opacity: 0.3 }}>/</span>
        <span style={{ opacity: 0.6 }}>out</span>
        <span style={{ fontWeight: 500 }}>{output_tokens ?? 0}</span>
        <span style={{ opacity: 0.5 }}>tok</span>
      </span>
    )
  }

  return (
    <div
      style={{
        display: 'flex',
        alignItems: 'center',
        flexWrap: 'wrap',
        gap: '0.25rem 0.5rem',
        marginTop: '0.5rem',
        padding: '0.25rem 0.6rem',
        borderRadius: '0.5rem',
        background: isGreen ? 'var(--bubble-reasoning-bg)' : 'var(--bubble-system-bg)',
        border: isGreen ? '1px solid var(--bubble-reasoning-border)' : '1px solid var(--color-border-subtle)',
        fontSize: '0.67rem',
        lineHeight: 1.6,
        fontVariantNumeric: 'tabular-nums',
        letterSpacing: '0.01em',
      }}
    >
      {items.map((item, i) => (
        <span key={i} style={{ display: 'inline-flex', alignItems: 'center', gap: '0.3rem' }}>
          {i > 0 && <Dot />}
          {item}
        </span>
      ))}
    </div>
  )
}
