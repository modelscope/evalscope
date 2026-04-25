import { useMemo } from 'react'
import { rdYlGn } from '@/utils/colorScale'
import { useLocale } from '@/contexts/LocaleContext'

interface DataPoint {
  label: string
  value: number   // 0 - 1 normalized
  rawValue?: number
}

interface Props {
  data: DataPoint[]
  size?: number
  /** Label to display in center */
  centerLabel?: string
}

const LEVELS = 5

/**
 * Pure SVG radar/spider chart.
 * Requires at least 3 data points; if fewer, renders a bar fallback.
 */
export default function RadarChart({ data, size = 320, centerLabel }: Props) {
  const { t } = useLocale()
  const cx = size / 2
  const cy = size / 2
  const radius = size * 0.36
  const labelRadius = size * 0.48

  const normalized = useMemo(() => {
    if (!data.length) return []
    // Values should already be 0-1 but clamp to be safe
    return data.map((d) => ({ ...d, value: Math.max(0, Math.min(1, d.value)) }))
  }, [data])

  /** Polar → cartesian, starting from top (−π/2), going clockwise */
  const polar = (angle: number, r: number) => ({
    x: cx + r * Math.cos(angle),
    y: cy + r * Math.sin(angle),
  })

  const n = normalized.length
  const angles = useMemo(
    () => Array.from({ length: n }, (_, i) => (2 * Math.PI * i) / n - Math.PI / 2),
    [n],
  )

  // Web polygon for a given radius fraction (0-1)
  const polygonPoints = (fraction: number) =>
    angles.map((a) => `${polar(a, radius * fraction).x},${polar(a, radius * fraction).y}`).join(' ')

  // Data polygon
  const dataPoints = useMemo(
    () => normalized.map((d, i) => polar(angles[i], radius * d.value)),
    [normalized, angles, radius],
  )

  const dataPolyline = dataPoints.map((p) => `${p.x},${p.y}`).join(' ')

  // Average score for coloring
  const avgScore = normalized.length ? normalized.reduce((s, d) => s + d.value, 0) / normalized.length : 0

  if (n < 3) {
    // Fallback: horizontal bar chart
    return <BarFallback data={normalized} />
  }

  return (
    <svg
      width={size}
      height={size}
      viewBox={`0 0 ${size} ${size}`}
      style={{ overflow: 'visible', maxWidth: '100%' }}
      aria-label={t('charts.radarAriaLabel')}
    >
      <defs>
        <radialGradient id="radarFill" cx="50%" cy="50%" r="50%">
          <stop offset="0%" stopColor={rdYlGn(avgScore)} stopOpacity="0.45" />
          <stop offset="100%" stopColor={rdYlGn(avgScore)} stopOpacity="0.08" />
        </radialGradient>
        <filter id="radarGlow">
          <feGaussianBlur stdDeviation="3" result="blur" />
          <feComposite in="SourceGraphic" in2="blur" operator="over" />
        </filter>
      </defs>

      {/* Grid rings */}
      {Array.from({ length: LEVELS }, (_, i) => {
        const frac = (i + 1) / LEVELS
        return (
          <polygon
            key={i}
            points={polygonPoints(frac)}
            fill="none"
            stroke="var(--color-border)"
            strokeWidth="1"
            opacity={0.6}
          />
        )
      })}

      {/* Spoke lines */}
      {angles.map((a, i) => {
        const outer = polar(a, radius)
        return (
          <line
            key={i}
            x1={cx}
            y1={cy}
            x2={outer.x}
            y2={outer.y}
            stroke="var(--color-border)"
            strokeWidth="1"
            opacity={0.5}
          />
        )
      })}

      {/* Ring level labels (0.2, 0.4, ...) */}
      {Array.from({ length: LEVELS }, (_, i) => {
        const frac = (i + 1) / LEVELS
        const p = polar(-Math.PI / 2, radius * frac)
        return (
          <text
            key={i}
            x={p.x + 3}
            y={p.y}
            fontSize="9"
            fill="var(--color-ink-faint)"
            dominantBaseline="middle"
          >
            {(frac * 100).toFixed(0)}
          </text>
        )
      })}

      {/* Data polygon fill */}
      <polygon
        points={dataPolyline}
        fill="url(#radarFill)"
        stroke={rdYlGn(avgScore)}
        strokeWidth="2"
        strokeLinejoin="round"
        style={{ filter: 'url(#radarGlow)' }}
      />

      {/* Data points */}
      {dataPoints.map((p, i) => (
        <g key={i}>
          <circle
            cx={p.x}
            cy={p.y}
            r={4}
            fill={rdYlGn(normalized[i].value)}
            stroke="var(--color-bg-page)"
            strokeWidth="1.5"
          />
          {/* Tooltip-like title */}
          <title>{`${normalized[i].label}: ${normalized[i].rawValue !== undefined ? normalized[i].rawValue : (normalized[i].value * 100).toFixed(1) + '%'}`}</title>
        </g>
      ))}

      {/* Axis labels */}
      {normalized.map((d, i) => {
        const p = polar(angles[i], labelRadius)
        const anchor = p.x < cx - 5 ? 'end' : p.x > cx + 5 ? 'start' : 'middle'
        // Shorten long labels
        const label = d.label.length > 14 ? d.label.slice(0, 13) + '…' : d.label
        return (
          <text
            key={i}
            x={p.x}
            y={p.y}
            textAnchor={anchor}
            dominantBaseline="middle"
            fontSize="11"
            fontWeight="500"
            fill="var(--color-ink-muted)"
          >
            <title>{d.label}</title>
            {label}
          </text>
        )
      })}

      {/* Center label */}
      {centerLabel && (
        <text
          x={cx}
          y={cy}
          textAnchor="middle"
          dominantBaseline="middle"
          fontSize="12"
          fontWeight="600"
          fill="var(--color-ink-muted)"
          opacity="0.6"
        >
          {centerLabel}
        </text>
      )}
    </svg>
  )
}

// ── Fallback bar chart for 1-2 datasets ─────────────────────────────────────
function BarFallback({ data }: { data: { label: string; value: number; rawValue?: number }[] }) {
  const barH = 28
  const labelW = 120
  const barMaxW = 220
  const gap = 8
  const totalH = data.length * (barH + gap) + 16

  return (
    <svg width={labelW + barMaxW + 60} height={totalH} style={{ maxWidth: '100%' }}>
      {data.map((d, i) => {
        const y = 8 + i * (barH + gap)
        const bw = barMaxW * d.value
        const display = d.rawValue !== undefined ? d.rawValue.toFixed(2) : (d.value * 100).toFixed(1) + '%'
        const label = d.label.length > 16 ? d.label.slice(0, 15) + '…' : d.label
        return (
          <g key={i}>
            <text x={labelW - 6} y={y + barH / 2} textAnchor="end" dominantBaseline="middle" fontSize="11" fill="var(--color-ink-muted)">
              <title>{d.label}</title>
              {label}
            </text>
            <rect x={labelW} y={y} width={Math.max(bw, 2)} height={barH} rx="4" fill={rdYlGn(d.value)} opacity="0.85" />
            <text x={labelW + bw + 6} y={y + barH / 2} dominantBaseline="middle" fontSize="11" fontWeight="600" fill="var(--color-ink)">
              {display}
            </text>
          </g>
        )
      })}
    </svg>
  )
}
