import { useState, useRef, type FC, type CSSProperties } from 'react'
import {
  Sparkles,
  Wrench,
  Cpu,
  AlertTriangle,
  CheckCircle2,
  ChevronDown,
  ChevronRight,
  ArrowRight,
  Zap,
} from 'lucide-react'
import { useLocale } from '@/contexts/LocaleContext'
import type { AgentTraceEvent, AgentTraceEventType } from '@/api/types'
import JsonViewer from '@/components/common/JsonViewer'

/* ─── Types ───────────────────────────────────────────────────── */

export interface AgentTraceProps {
  trace: import('@/api/types').AgentTrace
  className?: string
  /** Currently highlighted step (for cross-highlight with messages) */
  highlightedStep?: number | null
  /** Callback when a step is clicked */
  onStepClick?: (step: number) => void
}

/* ─── Event type config ───────────────────────────────────────── */

interface EventTypeConfig {
  icon: FC<{ size?: number; style?: CSSProperties }>
  color: string
  bgColor: string
  borderColor: string
  labelKey: string
}

function getEventTypeConfig(type: AgentTraceEventType): EventTypeConfig {
  switch (type) {
    case 'model_generate':
      return {
        icon: Sparkles,
        color: 'var(--bubble-bot-color)',
        bgColor: 'var(--bubble-bot-bg)',
        borderColor: 'var(--bubble-bot-border)',
        labelKey: 'trace.modelGenerate',
      }
    case 'tool_result':
      return {
        icon: Wrench,
        color: 'var(--bubble-tool-color)',
        bgColor: 'var(--bubble-tool-bg)',
        borderColor: 'var(--bubble-tool-border)',
        labelKey: 'trace.toolResult',
      }
    case 'env_exec':
      return {
        icon: Cpu,
        color: 'var(--text-muted)',
        bgColor: 'var(--bubble-system-bg)',
        borderColor: 'var(--bubble-system-border)',
        labelKey: 'trace.envExec',
      }
    case 'error':
      return {
        icon: AlertTriangle,
        color: 'var(--danger)',
        bgColor: 'var(--danger-bg)',
        borderColor: 'var(--danger-border, var(--danger))',
        labelKey: 'trace.error',
      }
    case 'submit':
      return {
        icon: CheckCircle2,
        color: 'var(--success, #10b981)',
        bgColor: 'var(--success-bg, rgba(16,185,129,.1))',
        borderColor: 'var(--success-border, rgba(16,185,129,.3))',
        labelKey: 'trace.submit',
      }
    case 'tool_call':
    default:
      return {
        icon: Wrench,
        color: 'var(--bubble-user-color)',
        bgColor: 'var(--bubble-user-bg)',
        borderColor: 'var(--bubble-user-border)',
        labelKey: 'trace.toolCall',
      }
  }
}

/* ─── Helpers ─────────────────────────────────────────────────── */

function fmtMs(ms: number | null | undefined): string {
  if (ms == null) return ''
  if (ms < 1000) return `${ms.toFixed(0)}ms`
  return `${(ms / 1000).toFixed(1)}s`
}

function fmtTokens(n: number | null | undefined): string {
  if (n == null) return '-'
  if (n >= 1000) return `${(n / 1000).toFixed(1)}k`
  return String(n)
}

/* ─── TraceEventCard ──────────────────────────────────────────── */

export interface TraceEventCardProps {
  event: AgentTraceEvent
  highlighted?: boolean
  onStepClick?: (step: number) => void
  cardRef?: (el: HTMLDivElement | null) => void
}

export function TraceEventCard({ event, highlighted, onStepClick, cardRef }: TraceEventCardProps) {
  const { t } = useLocale()
  const [payloadOpen, setPayloadOpen] = useState(false)
  const cfg = getEventTypeConfig(event.type)
  const Icon = cfg.icon
  const hasPayload = event.payload && Object.keys(event.payload).length > 0
  const label = t(cfg.labelKey) === cfg.labelKey ? event.type : t(cfg.labelKey)

  // Extract payload fields as typed strings to avoid `unknown` in JSX
  const p = event.payload
  const stopReason = typeof p.stop_reason === 'string' ? p.stop_reason : ''
  const payloadName = typeof p.name === 'string' ? p.name : ''
  const payloadError = p.error != null ? String(p.error) : ''
  const payloadPreview = typeof p.preview === 'string' ? p.preview : ''
  const payloadCommand = typeof p.command === 'string' ? p.command : ''
  const payloadMessage = p.message != null ? String(p.message) : ''
  const payloadFinalAnswer = p.final_answer != null ? String(p.final_answer) : ''

  return (
    <div
      ref={cardRef}
      className="rounded-xl overflow-hidden"
      style={{
        border: `1px solid ${highlighted ? cfg.color : cfg.borderColor}`,
        background: highlighted ? cfg.bgColor : 'var(--bg-card2)',
        boxShadow: highlighted ? `0 0 12px ${cfg.bgColor}` : 'none',
        transition: 'border-color 0.3s, background 0.3s, box-shadow 0.3s',
      }}
    >
      {/* Header row */}
      <div
        className="flex items-center gap-2.5 px-3.5 py-2.5"
        style={{ cursor: onStepClick ? 'pointer' : 'default' }}
        onClick={() => onStepClick?.(event.step)}
      >
        {/* Icon */}
        <div
          className="shrink-0 rounded-lg flex items-center justify-center"
          style={{
            width: 28,
            height: 28,
            background: cfg.bgColor,
            border: `1px solid ${cfg.borderColor}`,
          }}
        >
          <Icon size={14} style={{ color: cfg.color }} />
        </div>

        {/* Label + step */}
        <span className="font-semibold text-xs font-mono" style={{ color: cfg.color }}>
          {label}
        </span>
        <span
          className="text-[10px] font-mono px-1.5 py-0.5 rounded-full"
          style={{ background: 'var(--accent-dim)', color: 'var(--color-primary)' }}
        >
          Step {event.step}
        </span>

        <div className="flex-1" />

        {/* Latency badge */}
        {event.latency_ms != null && (
          <span
            className="flex items-center gap-1 text-[10px] font-mono font-medium px-2 py-0.5 rounded-full"
            style={{ background: 'var(--accent-dim)', color: 'var(--color-ink-muted)' }}
          >
            <Zap size={10} />
            {fmtMs(event.latency_ms)}
          </span>
        )}
      </div>

      {/* Body — event-type-specific details */}
      <div className="px-3.5 pb-3 pt-0 flex flex-col gap-1.5">
        {/* model_generate: token usage + stop_reason */}
        {event.type === 'model_generate' && (
          <>
            {event.token_usage && (
              <div className="flex items-center gap-3 text-[11px] font-mono" style={{ color: 'var(--color-ink-muted)' }}>
                <span>{t('trace.input')}: <b style={{ color: 'var(--color-ink)' }}>{fmtTokens(event.token_usage.input)}</b></span>
                <span style={{ opacity: 0.4 }}>→</span>
                <span>{t('trace.output')}: <b style={{ color: 'var(--color-ink)' }}>{fmtTokens(event.token_usage.output)}</b></span>
                <span style={{ opacity: 0.3 }}>|</span>
                <span>{t('trace.total')}: <b style={{ color: 'var(--color-ink)' }}>{fmtTokens(event.token_usage.total)}</b></span>
              </div>
            )}
            {stopReason && (
              <div className="text-[11px] font-mono" style={{ color: 'var(--color-ink-muted)' }}>
                {t('trace.stopReason')}: <span style={{ color: 'var(--color-ink)' }}>{stopReason}</span>
              </div>
            )}
          </>
        )}

        {/* tool_result: tool name + error + preview */}
        {event.type === 'tool_result' && (
          <>
            {payloadName && (
              <div className="text-[11px] font-mono" style={{ color: 'var(--color-ink-muted)' }}>
                {t('trace.toolName')}: <span style={{ color: 'var(--color-ink)' }}>{payloadName}</span>
              </div>
            )}
            {payloadError && (
              <div className="text-[11px] font-mono" style={{ color: 'var(--danger)' }}>
                Error: {payloadError}
              </div>
            )}
            {payloadPreview && (
              <div
                className="rounded-lg px-2.5 py-1.5 text-[11px] font-mono whitespace-pre-wrap break-all max-h-[80px] overflow-auto"
                style={{ background: 'var(--bg-deep)', color: 'var(--color-ink-muted)' }}
              >
                {payloadPreview.length > 300 ? payloadPreview.slice(0, 300) + '...' : payloadPreview}
              </div>
            )}
          </>
        )}

        {/* tool_call: tool name */}
        {event.type === 'tool_call' && payloadName && (
          <div className="text-[11px] font-mono" style={{ color: 'var(--color-ink-muted)' }}>
            {t('trace.toolName')}: <span style={{ color: 'var(--color-ink)' }}>{payloadName}</span>
          </div>
        )}

        {/* env_exec: command */}
        {event.type === 'env_exec' && payloadCommand && (
          <div
            className="rounded-lg px-2.5 py-1.5 text-[11px] font-mono whitespace-pre-wrap break-all"
            style={{ background: 'var(--bg-deep)', color: 'var(--color-ink-muted)' }}
          >
            $ {payloadCommand}
          </div>
        )}

        {/* error: error message */}
        {event.type === 'error' && (
          <div
            className="rounded-lg px-2.5 py-1.5 text-[11px] font-mono whitespace-pre-wrap break-all"
            style={{ background: 'var(--danger-bg)', color: 'var(--danger)' }}
          >
            {payloadMessage || t('trace.error')}
          </div>
        )}

        {/* submit: final_answer */}
        {event.type === 'submit' && payloadFinalAnswer && (
          <div
            className="rounded-lg px-2.5 py-1.5 text-[11px] whitespace-pre-wrap break-all max-h-[80px] overflow-auto"
            style={{ background: 'var(--success-bg, rgba(16,185,129,.1))', color: 'var(--color-ink-muted)' }}
          >
            {t('trace.finalAnswer')}: {payloadFinalAnswer.length > 200 ? payloadFinalAnswer.slice(0, 200) + '...' : payloadFinalAnswer}
          </div>
        )}

        {/* Expandable payload */}
        {hasPayload && (
          <div style={{ marginTop: '0.25rem' }}>
            <button
              onClick={() => setPayloadOpen(v => !v)}
              style={{
                display: 'inline-flex',
                alignItems: 'center',
                gap: '0.25rem',
                fontSize: '0.65rem',
                color: 'var(--color-ink-muted)',
                opacity: 0.6,
                background: 'none',
                border: 'none',
                cursor: 'pointer',
                padding: 0,
              }}
            >
              {payloadOpen ? <ChevronDown size={10} /> : <ChevronRight size={10} />}
              <span className="uppercase tracking-wide font-semibold">{t('trace.payload')}</span>
            </button>
            {payloadOpen && (
              <div
                style={{
                  marginTop: '0.25rem',
                  borderRadius: '0.375rem',
                  overflow: 'hidden',
                  border: '1px solid var(--color-border-subtle)',
                }}
              >
                <JsonViewer value={event.payload} maxHeight={180} />
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  )
}

/* ─── StepDivider ─────────────────────────────────────────────── */

export interface StepDividerProps {
  step: number
  totalLatencyMs?: number | null
  highlighted?: boolean
  onClick?: () => void
}

export function StepDivider({ step, totalLatencyMs, highlighted, onClick }: StepDividerProps) {
  const { t } = useLocale()
  return (
    <div
      className="flex items-center gap-2.5 my-3 cursor-pointer select-none"
      onClick={onClick}
      style={{ opacity: highlighted ? 1 : 0.7, transition: 'opacity 0.2s' }}
    >
      {/* Left line */}
      <div className="flex-1 h-px" style={{ background: 'var(--color-border-subtle)' }} />

      {/* Step badge */}
      <div
        className="flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-[10px] font-mono font-semibold"
        style={{
          background: highlighted ? 'var(--accent-dim)' : 'var(--bg-card2)',
          border: `1px solid ${highlighted ? 'var(--accent)' : 'var(--color-border-subtle)'}`,
          color: highlighted ? 'var(--accent)' : 'var(--color-ink-muted)',
          boxShadow: highlighted ? '0 0 8px var(--accent-dim)' : 'none',
          transition: 'all 0.3s',
        }}
      >
        {t('trace.step')} {step}
        {totalLatencyMs != null && (
          <>
            <span style={{ opacity: 0.4 }}>·</span>
            <span style={{ opacity: 0.8 }}>{fmtMs(totalLatencyMs)}</span>
          </>
        )}
      </div>

      {/* Right line */}
      <div className="flex-1 h-px" style={{ background: 'var(--color-border-subtle)' }} />
    </div>
  )
}

/* ─── AgentTraceFlowSummaryBar ────────────────────────────────── */

interface FlowSummaryBarProps {
  events: AgentTraceEvent[]
  onPillClick: (idx: number) => void
}

function AgentTraceFlowSummaryBar({ events, onPillClick }: FlowSummaryBarProps) {
  const { t } = useLocale()
  return (
    <div
      className="glass-card rounded-2xl px-4 py-3 mb-4"
      style={{ border: '1px solid var(--color-border)' }}
    >
      <div className="text-xs font-semibold uppercase tracking-wide mb-2.5" style={{ color: 'var(--color-ink-muted)' }}>
        {t('trace.flowSummary')}
      </div>
      <div className="flex flex-wrap items-center gap-1.5">
        {events.map((ev, i) => {
          const cfg = getEventTypeConfig(ev.type)
          const Icon = cfg.icon
          return (
            <div key={i} className="flex items-center gap-1.5">
              <button
                onClick={() => onPillClick(i)}
                className="flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] font-mono font-medium transition-all hover:scale-105 active:scale-95"
                style={{
                  background: cfg.bgColor,
                  border: `1px solid ${cfg.borderColor}`,
                  color: cfg.color,
                }}
                title={ev.type}
              >
                <Icon size={10} />
                <span>{t(cfg.labelKey) === cfg.labelKey ? ev.type : t(cfg.labelKey)}</span>
              </button>
              {i < events.length - 1 && (
                <ArrowRight size={10} style={{ color: 'var(--color-ink-faint)', flexShrink: 0 }} />
              )}
            </div>
          )
        })}
      </div>
    </div>
  )
}

/* ─── Main component ──────────────────────────────────────────── */

export default function TrajectoryView({ trace, className = '', highlightedStep, onStepClick }: AgentTraceProps) {
  const { t } = useLocale()
  const cardRefs = useRef<(HTMLDivElement | null)[]>([])

  // Group events by step
  const stepGroups = new Map<number, AgentTraceEvent[]>()
  for (const ev of trace.events) {
    if (!stepGroups.has(ev.step)) stepGroups.set(ev.step, [])
    stepGroups.get(ev.step)!.push(ev)
  }

  // Compute step-level total latency
  function stepTotalLatency(step: number): number | null {
    const evts = stepGroups.get(step)
    if (!evts) return null
    let total = 0
    let any = false
    for (const e of evts) {
      if (e.latency_ms != null) {
        total += e.latency_ms
        any = true
      }
    }
    return any ? total : null
  }

  function scrollToEvent(idx: number) {
    const el = cardRefs.current[idx]
    if (el) el.scrollIntoView({ behavior: 'smooth', block: 'nearest' })
  }

  // Build sorted list of (step, events) pairs
  const sortedSteps = Array.from(stepGroups.entries()).sort(([a], [b]) => a - b)

  // Build flat event index for scroll-to
  let eventIdx = 0

  return (
    <div className={`flex flex-col ${className}`}>
      {/* Flow summary bar */}
      <AgentTraceFlowSummaryBar events={trace.events} onPillClick={scrollToEvent} />

      {/* Trace header info */}
      <div className="flex flex-wrap items-center gap-3 mb-3 text-[11px] font-mono" style={{ color: 'var(--color-ink-muted)' }}>
        {trace.strategy && (
          <span>{t('trace.strategy')}: <b style={{ color: 'var(--color-ink)' }}>{trace.strategy}</b></span>
        )}
        {trace.environment && (
          <span>{t('trace.environment')}: <b style={{ color: 'var(--color-ink)' }}>{trace.environment}</b></span>
        )}
        <span>{t('trace.maxSteps')}: <b style={{ color: 'var(--color-ink)' }}>{trace.max_steps}</b></span>
      </div>

      {/* Step groups */}
      {sortedSteps.map(([step, events]) => {
        const isHighlighted = highlightedStep === step
        return (
          <div key={step}>
            <StepDivider
              step={step}
              totalLatencyMs={stepTotalLatency(step)}
              highlighted={isHighlighted}
              onClick={() => onStepClick?.(step)}
            />
            <div className="flex flex-col gap-2 ml-2">
              {events.map((ev) => {
                const idx = eventIdx++
                return (
                  <TraceEventCard
                    key={`${ev.type}-${idx}`}
                    event={ev}
                    highlighted={isHighlighted}
                    onStepClick={onStepClick}
                    cardRef={(el) => { cardRefs.current[idx] = el }}
                  />
                )
              })}
            </div>
          </div>
        )
      })}
    </div>
  )
}
