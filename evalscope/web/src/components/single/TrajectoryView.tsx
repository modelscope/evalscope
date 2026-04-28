import { useState, useRef } from 'react'
import {
  Cog,
  Target,
  Wrench,
  Sparkles,
  Circle,
  GitBranch,
  ChevronDown,
  ChevronRight,
  ArrowRight,
} from 'lucide-react'
import { useLocale } from '@/contexts/LocaleContext'

/* ─── Types ───────────────────────────────────────────────────── */

export interface TrajectoryStep {
  step_name: string
  step_type: string // "solver" | "scorer" | "tool" | "generate"
  timestamp: number
  messages_snapshot?: Array<{
    role: string // "system" | "user" | "assistant" | "tool"
    content: string
  }>
  metadata?: Record<string, unknown>
}

export interface TrajectoryViewProps {
  steps?: TrajectoryStep[]
  className?: string
}

/* ─── Mock demo data ──────────────────────────────────────────── */

const DEMO_STEPS: TrajectoryStep[] = [
  {
    step_name: 'system_message',
    step_type: 'solver',
    timestamp: Date.now() / 1000 - 4.2,
    messages_snapshot: [
      {
        role: 'system',
        content: 'You are a helpful AI assistant. Please answer the following question concisely and accurately.',
      },
    ],
    metadata: { latency_ms: 0.8 },
  },
  {
    step_name: 'prompt_template',
    step_type: 'solver',
    timestamp: Date.now() / 1000 - 3.8,
    messages_snapshot: [
      {
        role: 'system',
        content: 'You are a helpful AI assistant. Please answer the following question concisely and accurately.',
      },
      {
        role: 'user',
        content: `Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?`,
      },
    ],
    metadata: { template: 'gsm8k_default', latency_ms: 1.2 },
  },
  {
    step_name: 'generate',
    step_type: 'generate',
    timestamp: Date.now() / 1000 - 2.1,
    messages_snapshot: [
      {
        role: 'user',
        content: `Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?`,
      },
      {
        role: 'assistant',
        content: `Janet's ducks lay 16 eggs per day.\nShe eats 3 for breakfast and uses 4 for baking, so she uses 3 + 4 = 7 eggs.\nShe sells 16 - 7 = 9 eggs at the farmers' market.\nShe makes 9 × $2 = **$18** per day.`,
      },
    ],
    metadata: { model: 'qwen-plus', tokens_input: 128, tokens_output: 64, latency_ms: 1820 },
  },
  {
    step_name: 'extract_answer',
    step_type: 'solver',
    timestamp: Date.now() / 1000 - 0.8,
    messages_snapshot: [
      {
        role: 'assistant',
        content: '18',
      },
    ],
    metadata: { pattern: '(\\d+)', extracted: '18', latency_ms: 3.1 },
  },
  {
    step_name: 'accuracy',
    step_type: 'scorer',
    timestamp: Date.now() / 1000 - 0.3,
    metadata: { gold: '18', pred: '18', score: 1.0, method: 'exact_match', latency_ms: 0.5 },
  },
]

/* ─── Helpers ─────────────────────────────────────────────────── */

function formatRelTime(ts: number, t: (key: string) => string): string {
  const delta = Date.now() / 1000 - ts
  if (delta < 1) return t('common.justNow')
  if (delta < 60) return t('common.secondsAgo').replace('${n}', delta.toFixed(1))
  if (delta < 3600) return t('common.minutesAgo').replace('${n}', String(Math.round(delta / 60)))
  return t('common.hoursAgo').replace('${n}', String(Math.round(delta / 3600)))
}

/* ─── Step type config ────────────────────────────────────────── */

interface StepTypeConfig {
  icon: React.FC<{ size?: number; style?: React.CSSProperties }>
  color: string
  bgColor: string
  borderColor: string
  glowColor: string
}

function getStepTypeConfig(stepType: string): StepTypeConfig {
  switch (stepType) {
    case 'solver':
      return {
        icon: Cog,
        color: 'var(--bubble-user-color)',
        bgColor: 'var(--bubble-user-icon-bg)',
        borderColor: 'var(--bubble-user-border)',
        glowColor: 'var(--bubble-user-bg)',
      }
    case 'generate':
      return {
        icon: Sparkles,
        color: 'var(--bubble-bot-color)',
        bgColor: 'var(--bubble-bot-icon-bg)',
        borderColor: 'var(--bubble-bot-border)',
        glowColor: 'var(--bubble-bot-bg)',
      }
    case 'tool':
      return {
        icon: Wrench,
        color: 'var(--bubble-tool-color)',
        bgColor: 'var(--bubble-tool-icon-bg)',
        borderColor: 'var(--bubble-tool-border)',
        glowColor: 'var(--bubble-tool-bg)',
      }
    case 'scorer':
      return {
        icon: Target,
        color: 'var(--purple)',
        bgColor: 'var(--accent-dim)',
        borderColor: 'var(--border-md)',
        glowColor: 'var(--accent-dim)',
      }
    default:
      return {
        icon: Circle,
        color: 'var(--text-muted)',
        bgColor: 'var(--bubble-system-bg)',
        borderColor: 'var(--bubble-system-border)',
        glowColor: 'var(--bubble-system-bg)',
      }
  }
}

/* ─── Mini message bubble ─────────────────────────────────────── */

function MiniMessageBubble({
  role,
  content,
}: {
  role: string
  content: string
}) {
  const [expanded, setExpanded] = useState(false)
  const isUser = role === 'user'
  const isAssistant = role === 'assistant'
  const isSystem = role === 'system'
  const isTool = role === 'tool'

  const preview = content.length > 180 ? content.slice(0, 180) + '…' : content

  let bubbleStyle: React.CSSProperties = {}
  let labelColor = 'var(--color-ink-muted)'
  let labelText = role

  if (isUser) {
    bubbleStyle = {
      background: 'var(--bubble-user-bg)',
      border: '1px solid var(--bubble-user-border)',
    }
    labelColor = 'var(--bubble-user-color)'
    labelText = 'user'
  } else if (isAssistant) {
    bubbleStyle = {
      background: 'var(--bubble-bot-bg)',
      border: '1px solid var(--bubble-bot-border)',
    }
    labelColor = 'var(--bubble-bot-color)'
    labelText = 'assistant'
  } else if (isSystem) {
    bubbleStyle = {
      background: 'var(--bubble-system-bg)',
      border: '1px solid var(--bubble-system-border)',
    }
    labelColor = 'var(--text-muted)'
    labelText = 'system'
  } else if (isTool) {
    bubbleStyle = {
      background: 'var(--bubble-tool-bg)',
      border: '1px solid var(--bubble-tool-border)',
    }
    labelColor = 'var(--bubble-tool-color)'
    labelText = 'tool'
  }

  return (
    <div
      className="rounded-xl px-3 py-2 text-xs cursor-pointer select-none"
      style={bubbleStyle}
      onClick={() => content.length > 180 && setExpanded((v) => !v)}
    >
      <div className="flex items-center gap-1.5 mb-1">
        <span className="font-mono font-semibold uppercase tracking-wide text-[10px]" style={{ color: labelColor }}>
          {labelText}
        </span>
        {content.length > 180 && (
          <span style={{ color: labelColor, opacity: 0.6 }}>
            {expanded ? <ChevronDown size={10} /> : <ChevronRight size={10} />}
          </span>
        )}
      </div>
      <pre className="whitespace-pre-wrap break-words font-sans text-xs" style={{ color: 'var(--color-ink)', margin: 0 }}>
        {expanded ? content : preview}
      </pre>
    </div>
  )
}

/* ─── Metadata table ──────────────────────────────────────────── */

function MetaTable({ data }: { data: Record<string, unknown> }) {
  return (
    <div className="rounded-xl overflow-hidden" style={{ border: '1px solid var(--color-border-subtle)' }}>
      <table className="w-full text-xs">
        <tbody>
          {Object.entries(data).map(([k, v], i) => (
            <tr
              key={k}
              style={{
                background: i % 2 === 0 ? 'var(--accent-dim)' : 'transparent',
                borderTop: i > 0 ? '1px solid var(--color-border-subtle)' : undefined,
              }}
            >
              <td
                className="px-3 py-1.5 font-mono font-medium whitespace-nowrap"
                style={{ color: 'var(--color-primary)', width: '40%' }}
              >
                {k}
              </td>
              <td className="px-3 py-1.5 font-mono break-all" style={{ color: 'var(--color-ink-muted)' }}>
                {typeof v === 'object' ? JSON.stringify(v) : String(v)}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

/* ─── Step card ───────────────────────────────────────────────── */

function StepCard({
  step,
  index,
  isLast,
  cardRef,
}: {
  step: TrajectoryStep
  index: number
  isLast: boolean
  cardRef?: (el: HTMLDivElement | null) => void
}) {
  const { t } = useLocale()
  const [expanded, setExpanded] = useState(false)
  const [activeTab, setActiveTab] = useState<'messages' | 'metadata'>('messages')
  const cfg = getStepTypeConfig(step.step_type)
  const Icon = cfg.icon
  const hasMessages = step.messages_snapshot && step.messages_snapshot.length > 0
  const hasMeta = step.metadata && Object.keys(step.metadata).length > 0
  const hasContent = hasMessages || hasMeta

  const stepTypeLabel =
    t(`trajectory.stepType.${step.step_type}`) === `trajectory.stepType.${step.step_type}`
      ? step.step_type
      : t(`trajectory.stepType.${step.step_type}`)

  return (
    <div
      className="flex gap-0 items-stretch"
      style={{
        animation: `fadeInUp 400ms ease-out ${index * 80}ms both`,
      }}
    >
      {/* Timeline spine */}
      <div className="flex flex-col items-center" style={{ width: 40, flexShrink: 0 }}>
        {/* Node */}
        <div
          className="relative flex items-center justify-center rounded-full shrink-0 z-10"
          style={{
            width: 36,
            height: 36,
            background: cfg.bgColor,
            border: `2px solid ${cfg.borderColor}`,
            boxShadow: `0 0 12px ${cfg.glowColor}, 0 2px 8px rgba(0,0,0,0.3)`,
            marginTop: 4,
          }}
        >
          <Icon size={16} style={{ color: cfg.color }} />
        </div>
        {/* Connecting line */}
        {!isLast && (
          <div
            className="flex-1 w-px mt-1"
            style={{
              background: `linear-gradient(to bottom, ${cfg.borderColor}, var(--color-border-subtle))`,
              minHeight: 24,
            }}
          />
        )}
      </div>

      {/* Card */}
      <div
        ref={cardRef}
        className="flex-1 mb-4 ml-3 glass-card rounded-2xl overflow-hidden"
        style={{ border: `1px solid ${expanded ? cfg.borderColor : 'var(--color-border)'}`, transition: 'border-color 250ms' }}
      >
        {/* Header */}
        <button
          className="w-full flex items-center gap-3 px-4 py-3 text-left hover:bg-[var(--color-surface-hover)] transition-colors"
          onClick={() => hasContent && setExpanded((v) => !v)}
          style={{ cursor: hasContent ? 'pointer' : 'default' }}
        >
          {/* Step name */}
          <span className="font-semibold text-sm font-mono flex-1 min-w-0 truncate" style={{ color: 'var(--color-ink)' }}>
            {step.step_name}
          </span>

          {/* Type badge */}
          <span
            className="text-[10px] font-semibold uppercase tracking-wider px-2 py-0.5 rounded-full shrink-0"
            style={{ background: cfg.bgColor, color: cfg.color, border: `1px solid ${cfg.borderColor}` }}
          >
            {stepTypeLabel}
          </span>

          {/* Timestamp */}
          <span className="text-xs shrink-0" style={{ color: 'var(--color-ink-faint)' }}>
            {formatRelTime(step.timestamp, t)}
          </span>

          {/* Expand icon */}
          {hasContent && (
            <span style={{ color: 'var(--color-ink-faint)', flexShrink: 0 }}>
              {expanded ? <ChevronDown size={15} /> : <ChevronRight size={15} />}
            </span>
          )}
        </button>

        {/* Expandable body */}
        <div
          style={{
            overflow: 'hidden',
            maxHeight: expanded ? '9999px' : 0,
            transition: 'max-height 300ms cubic-bezier(0.4, 0, 0.2, 1)',
          }}
        >
          <div style={{ borderTop: '1px solid var(--color-border-subtle)' }}>
            {/* Tabs (when both exist) */}
            {hasMessages && hasMeta && (
              <div className="flex gap-0 px-4 pt-3">
                {(['messages', 'metadata'] as const).map((tab) => (
                  <button
                    key={tab}
                    className="px-3 py-1 text-xs font-medium rounded-lg mr-1 transition-colors"
                    style={{
                      background: activeTab === tab ? cfg.bgColor : 'transparent',
                      color: activeTab === tab ? cfg.color : 'var(--color-ink-muted)',
                      border: `1px solid ${activeTab === tab ? cfg.borderColor : 'transparent'}`,
                    }}
                    onClick={() => setActiveTab(tab)}
                  >
                    {tab === 'messages' ? t('trajectory.messages') : t('trajectory.metadata')}
                  </button>
                ))}
              </div>
            )}

            <div className="px-4 pb-4 pt-3 flex flex-col gap-2">
              {/* Messages */}
              {hasMessages && (!hasMeta || activeTab === 'messages') &&
                step.messages_snapshot!.map((msg, i) => (
                  <MiniMessageBubble key={i} role={msg.role} content={msg.content} />
                ))}

              {/* Metadata */}
              {hasMeta && (!hasMessages || activeTab === 'metadata') && (
                <MetaTable data={step.metadata!} />
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

/* ─── Flow summary bar ────────────────────────────────────────── */

function FlowSummaryBar({
  steps,
  onPillClick,
}: {
  steps: TrajectoryStep[]
  onPillClick: (idx: number) => void
}) {
  const { t } = useLocale()
  return (
    <div
      className="glass-card rounded-2xl px-4 py-3 mb-6"
      style={{ border: '1px solid var(--color-border)' }}
    >
      <div className="text-xs font-semibold uppercase tracking-wide mb-3" style={{ color: 'var(--color-ink-muted)' }}>
        {t('trajectory.flowSummary')}
      </div>
      <div className="flex flex-wrap items-center gap-1.5">
        {steps.map((step, i) => {
          const cfg = getStepTypeConfig(step.step_type)
          return (
            <div key={i} className="flex items-center gap-1.5">
              <button
                onClick={() => onPillClick(i)}
                className="flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-mono font-medium transition-all hover:scale-105 active:scale-95"
                style={{
                  background: cfg.bgColor,
                  border: `1px solid ${cfg.borderColor}`,
                  color: cfg.color,
                  boxShadow: `0 0 8px ${cfg.glowColor}`,
                  transition: 'all var(--transition-fast)',
                }}
                title={step.step_type}
              >
                <cfg.icon size={11} />
                <span>{step.step_name}</span>
              </button>
              {i < steps.length - 1 && (
                <ArrowRight size={12} style={{ color: 'var(--color-ink-faint)', flexShrink: 0 }} />
              )}
            </div>
          )
        })}
      </div>
    </div>
  )
}

/* ─── Empty state ─────────────────────────────────────────────── */

function EmptyState({ onShowDemo }: { onShowDemo: () => void }) {
  const { t } = useLocale()
  return (
    <div className="flex flex-col items-center justify-center py-16 px-6">
      <div
        className="rounded-3xl p-12 flex flex-col items-center gap-5 max-w-md w-full"
        style={{
          border: '2px dashed var(--bubble-system-border)',
          background: 'var(--bubble-system-bg)',
        }}
      >
        {/* Icon cluster */}
        <div className="relative">
          <div
            className="w-16 h-16 rounded-2xl flex items-center justify-center"
            style={{
              background: 'var(--bubble-user-icon-bg)',
              border: '1px solid var(--bubble-user-border)',
              boxShadow: '0 0 24px var(--bubble-user-bg)',
            }}
          >
            <GitBranch size={28} style={{ color: 'var(--bubble-user-color)' }} />
          </div>
          {/* Floating decorative dots */}
          <div
            className="absolute -top-1 -right-1 w-3 h-3 rounded-full"
            style={{ background: 'var(--bubble-bot-icon-bg)', border: '1px solid var(--bubble-bot-border)' }}
          />
          <div
            className="absolute -bottom-1 -left-1 w-2 h-2 rounded-full"
            style={{ background: 'var(--accent-dim)', border: '1px solid var(--border-md)' }}
          />
        </div>

        {/* Text */}
        <div className="text-center">
          <h3 className="text-base font-semibold mb-2" style={{ color: 'var(--color-ink)' }}>
            {t('trajectory.noData')}
          </h3>
          <p className="text-sm leading-relaxed" style={{ color: 'var(--color-ink-muted)' }}>
            {t('trajectory.noDataDesc')}
          </p>
        </div>

        {/* Demo button */}
        <button
          onClick={onShowDemo}
          className="flex items-center gap-2 px-4 py-2 rounded-xl text-sm font-medium transition-all hover:scale-105 active:scale-95 btn-glow"
          style={{
            background: 'var(--bubble-user-bg)',
            border: '1px solid var(--bubble-user-border)',
            color: 'var(--bubble-user-color)',
          }}
        >
          <Sparkles size={14} />
          {t('trajectory.showDemo')}
        </button>
      </div>
    </div>
  )
}

/* ─── Main component ──────────────────────────────────────────── */

export default function TrajectoryView({ steps, className = '' }: TrajectoryViewProps) {
  const { t } = useLocale()
  const [showDemo, setShowDemo] = useState(false)
  const stepRefs = useRef<(HTMLDivElement | null)[]>([])

  const activeSteps = (steps && steps.length > 0) ? steps : (showDemo ? DEMO_STEPS : [])
  const isEmpty = !steps || steps.length === 0

  function scrollToStep(idx: number) {
    const el = stepRefs.current[idx]
    if (el) {
      el.scrollIntoView({ behavior: 'smooth', block: 'nearest' })
    }
  }

  return (
    <div className={`flex flex-col ${className}`}>
      {/* Empty state */}
      {isEmpty && !showDemo && (
        <EmptyState onShowDemo={() => setShowDemo(true)} />
      )}

      {/* Demo mode banner */}
      {isEmpty && showDemo && (
        <div
          className="flex items-center justify-between px-4 py-2 rounded-xl mb-4 text-xs font-medium"
          style={{
            background: 'var(--warning-bg)',
            border: '1px solid var(--warning-border)',
            color: 'var(--yellow)',
          }}
        >
          <div className="flex items-center gap-2">
            <Sparkles size={13} />
            <span>Demo Preview — {t('trajectory.noData')}</span>
          </div>
          <button
            onClick={() => setShowDemo(false)}
            className="hover:opacity-70 transition-opacity"
          >
            {t('trajectory.hideDemo')}
          </button>
        </div>
      )}

      {/* Content */}
      {activeSteps.length > 0 && (
        <>
          {/* Flow summary bar */}
          <FlowSummaryBar steps={activeSteps} onPillClick={scrollToStep} />

          {/* Timeline */}
          <div className="flex flex-col">
            {activeSteps.map((step, i) => (
              <StepCard
                key={`${step.step_name}-${i}`}
                step={step}
                index={i}
                isLast={i === activeSteps.length - 1}
                cardRef={(el) => { stepRefs.current[i] = el }}
              />
            ))}
          </div>
        </>
      )}
    </div>
  )
}
