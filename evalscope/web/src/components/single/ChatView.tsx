import { useState } from 'react'
import { User, Bot, Target, ChevronDown, ChevronRight, Shield } from 'lucide-react'
import type { PredictionRow } from '@/api/types'
import { useLocale } from '@/contexts/LocaleContext'
import MarkdownRenderer from '@/components/common/MarkdownRenderer'
import ScoreBadge from '@/components/common/ScoreBadge'

interface Props {
  prediction: PredictionRow
  threshold?: number
}

/* ─── helpers ─────────────────────────────────────────────────── */

function hasSystemPrompt(input: string): boolean {
  const lower = input.trim().toLowerCase()
  return (
    lower.startsWith('<|system|>') ||
    lower.startsWith('[system]') ||
    lower.startsWith('system:') ||
    /^```[\s\S]*?system/i.test(input.trim()) ||
    (input.includes('<|system|>') && input.includes('<|user|>'))
  )
}

function parseSystemUser(input: string): { system: string; user: string } {
  // multi-turn format with <|system|> ... <|user|>
  const sysMatch = input.match(/<\|system\|>([\s\S]*?)(?:<\|user\|>|$)/i)
  const userMatch = input.match(/<\|user\|>([\s\S]*?)(?:<\|assistant\|>|$)/i)
  if (sysMatch) {
    return {
      system: sysMatch[1].trim(),
      user: userMatch ? userMatch[1].trim() : input.replace(/<\|system\|>[\s\S]*?<\|user\|>/i, '').trim(),
    }
  }
  // [system] prefix
  const bracketMatch = input.match(/^\[system\]([\s\S]*?)(?:\[user\]|$)/i)
  if (bracketMatch) {
    return {
      system: bracketMatch[1].trim(),
      user: input.replace(/^\[system\][\s\S]*?(?:\[user\])/i, '').trim(),
    }
  }
  // system: prefix
  const colonMatch = input.match(/^system:\s*([\s\S]*?)(?:\nuser:|$)/i)
  if (colonMatch) {
    return {
      system: colonMatch[1].trim(),
      user: input.replace(/^system:\s*[\s\S]*?\nuser:\s*/i, '').trim(),
    }
  }
  return { system: '', user: input }
}

function extractToolCalls(text: string): { before: string; calls: string[]; after: string } {
  const calls: string[] = []
  let remaining = text

  // <tool_call> ... </tool_call>
  const toolCallRegex = /<tool_call>([\s\S]*?)<\/tool_call>/g
  remaining = remaining.replace(toolCallRegex, (match) => {
    calls.push(match)
    return ''
  })

  // JSON function call pattern: {"name": "...", "arguments": {...}}
  const jsonFnRegex = /\{"name"\s*:\s*"[^"]+"\s*,\s*"arguments"\s*:\s*\{[\s\S]*?\}\s*\}/g
  remaining = remaining.replace(jsonFnRegex, (match) => {
    calls.push(match)
    return ''
  })

  return { before: remaining.trim(), calls, after: '' }
}

/* ─── sub-components ──────────────────────────────────────────── */

function SystemBanner({ content }: { content: string }) {
  const { t } = useLocale()
  const [expanded, setExpanded] = useState(false)
  const preview = content.length > 120 ? content.slice(0, 120) + '…' : content
  return (
    <div
      className="rounded-xl border px-4 py-3 text-sm flex gap-3 items-start cursor-pointer select-none"
      style={{
        background: 'rgba(148,163,184,0.06)',
        borderColor: 'rgba(148,163,184,0.2)',
        color: 'var(--color-ink-muted)',
      }}
      onClick={() => setExpanded((v) => !v)}
    >
      <Shield size={15} className="mt-0.5 shrink-0 opacity-70" />
      <div className="flex-1 min-w-0">
        <span className="font-semibold text-xs uppercase tracking-wide opacity-70 mr-2">
          {t('prediction.systemPrompt')}
        </span>
        <span className="font-mono text-xs">{expanded ? content : preview}</span>
      </div>
      {content.length > 120 &&
        (expanded ? <ChevronDown size={14} className="shrink-0" /> : <ChevronRight size={14} className="shrink-0" />)}
    </div>
  )
}

function ToolCallBlock({ raw }: { raw: string }) {
  const { t } = useLocale()
  const [open, setOpen] = useState(false)
  let fnName = t('prediction.toolCall')
  try {
    const inner = raw.replace(/<\/?tool_call>/g, '').trim()
    const parsed = JSON.parse(inner)
    if (parsed.name) fnName = parsed.name
  } catch { /* noop */ }
  return (
    <div className="mt-2 rounded-xl border border-[var(--color-border)] overflow-hidden text-xs">
      <button
        onClick={() => setOpen((v) => !v)}
        className="flex items-center gap-2 w-full px-3 py-2 text-left font-mono hover:bg-[var(--color-surface-hover)] transition-colors"
        style={{ background: 'rgba(99,102,241,0.08)', color: 'var(--color-primary)' }}
      >
        {open ? <ChevronDown size={13} /> : <ChevronRight size={13} />}
        <span className="font-semibold">{t('prediction.toolCall')}:</span>
        <span className="font-normal opacity-80">{fnName}</span>
      </button>
      {open && (
        <pre
          className="px-3 py-2 text-xs font-mono whitespace-pre-wrap break-all overflow-auto max-h-[200px]"
          style={{ background: 'var(--color-surface)', color: 'var(--color-ink-muted)' }}
        >
          {raw.replace(/<\/?tool_call>/g, '').trim()}
        </pre>
      )}
    </div>
  )
}

function UserBubble({ content }: { content: string }) {
  return (
    <div className="flex gap-3 items-start" style={{ animation: 'fadeInUp 300ms ease-out both' }}>
      <div
        className="shrink-0 w-8 h-8 rounded-full flex items-center justify-center mt-1"
        style={{ background: 'rgba(99,102,241,0.2)', border: '1px solid rgba(99,102,241,0.35)' }}
      >
        <User size={15} style={{ color: '#818cf8' }} />
      </div>
      <div
        className="flex-1 rounded-2xl px-4 py-3 text-sm shadow"
        style={{
          maxWidth: '80%',
          background: 'rgba(99,102,241,0.12)',
          border: '1px solid rgba(99,102,241,0.25)',
          boxShadow: '0 2px 8px rgba(99,102,241,0.12)',
        }}
      >
        <MarkdownRenderer content={content} />
      </div>
    </div>
  )
}

function AssistantBubble({ content }: { content: string }) {
  const { calls, before } = extractToolCalls(content)
  return (
    <div
      className="flex gap-3 items-start justify-end"
      style={{ animation: 'fadeInUp 300ms ease-out 80ms both' }}
    >
      <div
        className="flex-1 rounded-2xl px-4 py-3 text-sm"
        style={{
          maxWidth: '80%',
          background: 'rgba(16,185,129,0.1)',
          border: '1px solid rgba(16,185,129,0.25)',
          boxShadow: '0 2px 8px rgba(16,185,129,0.1)',
        }}
      >
        {before && <MarkdownRenderer content={before} />}
        {calls.map((c, i) => <ToolCallBlock key={i} raw={c} />)}
      </div>
      <div
        className="shrink-0 w-8 h-8 rounded-full flex items-center justify-center mt-1"
        style={{ background: 'rgba(16,185,129,0.15)', border: '1px solid rgba(16,185,129,0.3)' }}
      >
        <Bot size={15} style={{ color: '#34d399' }} />
      </div>
    </div>
  )
}

function GoldCard({ content, threshold }: { content: string; threshold: number }) {
  const { t } = useLocale()
  return (
    <div
      className="rounded-2xl px-4 py-3 text-sm"
      style={{
        background: 'rgba(245,158,11,0.08)',
        border: '1px solid rgba(245,158,11,0.25)',
        animation: 'fadeInUp 300ms ease-out 160ms both',
      }}
    >
      <div className="flex items-center gap-2 mb-2">
        <Target size={14} style={{ color: '#fbbf24' }} />
        <span className="text-xs font-semibold uppercase tracking-wide" style={{ color: '#fbbf24' }}>
          {t('prediction.expectedAnswer')}
        </span>
      </div>
      <MarkdownRenderer content={content} />
      <div className="mt-2">
        <ScoreBadge score={threshold} threshold={threshold} />
      </div>
    </div>
  )
}

/* ─── main export ─────────────────────────────────────────────── */

export default function ChatView({ prediction, threshold = 0.99 }: Props) {
  const isSystemMsg = hasSystemPrompt(prediction.Input)
  const { system, user } = isSystemMsg ? parseSystemUser(prediction.Input) : { system: '', user: prediction.Input }

  const showPred =
    prediction.Pred &&
    prediction.Generated &&
    prediction.Pred.trim() !== prediction.Generated.trim()

  return (
    <div className="flex flex-col gap-4 py-2">
      {/* System prompt banner */}
      {system && <SystemBanner content={system} />}

      {/* User message */}
      <UserBubble content={user || prediction.Input} />

      {/* Assistant: Generated */}
      {prediction.Generated && <AssistantBubble content={prediction.Generated} />}

      {/* Pred (only if different) */}
      {showPred && (
        <div className="pl-11">
          <div
            className="rounded-xl px-3 py-2 text-xs"
            style={{
              background: 'rgba(139,92,246,0.08)',
              border: '1px solid rgba(139,92,246,0.2)',
            }}
          >
            <span className="font-semibold uppercase tracking-wide text-xs opacity-60 mr-2" style={{ color: '#a78bfa' }}>
              Pred
            </span>
            <MarkdownRenderer content={prediction.Pred} />
          </div>
        </div>
      )}

      {/* Divider */}
      <div style={{ borderTop: '1px solid var(--color-border-subtle)' }} />

      {/* Gold + Score */}
      <GoldCard content={prediction.Gold} threshold={threshold} />

      {/* Score */}
      <div
        className="flex items-center gap-3 px-1"
        style={{ animation: 'fadeInUp 300ms ease-out 200ms both' }}
      >
        <span className="text-xs text-[var(--color-ink-muted)] uppercase tracking-wide font-semibold">Score</span>
        <ScoreBadge score={prediction.NScore} threshold={threshold} />
      </div>
    </div>
  )
}
