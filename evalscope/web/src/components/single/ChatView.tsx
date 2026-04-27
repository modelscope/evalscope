import { useState, useCallback } from 'react'
import { User, Bot, Shield, ChevronDown, ChevronRight, ClipboardCheck, Copy, Check } from 'lucide-react'
import type { PredictionRow } from '@/api/types'
import { useLocale } from '@/contexts/LocaleContext'
import MarkdownRenderer from '@/components/common/MarkdownRenderer'
import ScoreBadge from '@/components/common/ScoreBadge'
import JsonViewer from '@/components/common/JsonViewer'
import PerfChip from './PerfChip'

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
      <div className="bubble-wrap" style={{ maxWidth: '80%', position: 'relative' }}>
        <div
          className="flex-1 rounded-2xl px-4 py-3 text-sm shadow"
          style={{
            background: 'rgba(99,102,241,0.12)',
            border: '1px solid rgba(99,102,241,0.25)',
            boxShadow: '0 2px 8px rgba(99,102,241,0.12)',
          }}
        >
          <MarkdownRenderer content={content} />
        </div>
        {/* Copy button — floats above top-right, visible on hover */}
        <div className="bubble-copy-btn" style={{ position: 'absolute', top: '-0.75rem', right: '0.25rem' }}>
          <CopyButton text={content} variant="indigo" />
        </div>
      </div>
    </div>
  )
}

interface AssistantBubbleProps {
  content: string
  perfMetrics?: PredictionRow['PerfMetrics']
}

type CopyVariant = 'green' | 'indigo'

function CopyButton({ text, variant = 'green' }: { text: string; variant?: CopyVariant }) {
  const { t } = useLocale()
  const [copied, setCopied] = useState(false)
  const handleCopy = useCallback(async () => {
    try {
      await navigator.clipboard.writeText(text)
      setCopied(true)
      setTimeout(() => setCopied(false), 1800)
    } catch {
      // fallback for non-https
      const el = document.createElement('textarea')
      el.value = text
      document.body.appendChild(el)
      el.select()
      document.execCommand('copy')
      document.body.removeChild(el)
      setCopied(true)
      setTimeout(() => setCopied(false), 1800)
    }
  }, [text])

  const activeColor = variant === 'green' ? '#34d399' : '#818cf8'
  const borderColor = variant === 'green' ? 'rgba(16,185,129,0.2)' : 'rgba(99,102,241,0.2)'
  const bgColor = variant === 'green' ? 'rgba(16,185,129,0.06)' : 'rgba(99,102,241,0.06)'

  return (
    <button
      onClick={handleCopy}
      title={t('prediction.copyContent')}
      style={{
        display: 'inline-flex',
        alignItems: 'center',
        gap: '0.25rem',
        padding: '0.2rem 0.5rem',
        borderRadius: '0.4rem',
        border: `1px solid ${borderColor}`,
        background: bgColor,
        color: copied ? activeColor : 'rgba(148,163,184,0.7)',
        fontSize: '0.68rem',
        cursor: 'pointer',
        transition: 'all 0.15s',
      }}
    >
      {copied ? <Check size={11} /> : <Copy size={11} />}
      <span>{copied ? t('prediction.copySuccess') : t('prediction.copyContent')}</span>
    </button>
  )
}

function AssistantBubble({ content, perfMetrics }: AssistantBubbleProps) {
  const { calls, before } = extractToolCalls(content)
  return (
    <div
      className="flex gap-3 items-start justify-end"
      style={{ animation: 'fadeInUp 300ms ease-out 80ms both' }}
    >
      <div className="bubble-wrap" style={{ maxWidth: '80%', position: 'relative' }}>
        <div
          className="flex-1 rounded-2xl px-4 py-3 text-sm"
          style={{
            background: 'rgba(16,185,129,0.1)',
            border: '1px solid rgba(16,185,129,0.25)',
            boxShadow: '0 2px 8px rgba(16,185,129,0.1)',
          }}
        >
          {before && <MarkdownRenderer content={before} />}
          {calls.map((c, i) => <ToolCallBlock key={i} raw={c} />)}
          {perfMetrics && <PerfChip metrics={perfMetrics} variant="green" />}
        </div>
        {/* Copy button — floats above top-right, visible on hover */}
        <div className="bubble-copy-btn" style={{ position: 'absolute', top: '-0.75rem', right: '0.25rem' }}>
          <CopyButton text={content} variant="green" />
        </div>
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

/* ─── Collapsible JSON block ────────────────────────────────────── */

interface CollapsibleJsonProps {
  label: string
  value: unknown
  maxHeight?: number
  defaultOpen?: boolean
}

function CollapsibleJson({ label, value, maxHeight = 200, defaultOpen = false }: CollapsibleJsonProps) {
  const [open, setOpen] = useState(defaultOpen)

  const isEmpty = value == null ||
    (typeof value === 'object' && Object.keys(value as object).length === 0) ||
    value === '{}'

  if (isEmpty) return null

  return (
    <div style={{ marginTop: '0.5rem' }}>
      <button
        onClick={() => setOpen((v) => !v)}
        style={{
          display: 'inline-flex',
          alignItems: 'center',
          gap: '0.3rem',
          fontSize: '0.7rem',
          color: 'var(--color-ink-muted)',
          opacity: 0.7,
          background: 'none',
          border: 'none',
          cursor: 'pointer',
          padding: '0.1rem 0',
          transition: 'opacity 0.15s',
        }}
        onMouseEnter={(e) => (e.currentTarget.style.opacity = '1')}
        onMouseLeave={(e) => (e.currentTarget.style.opacity = '0.7')}
      >
        {open ? <ChevronDown size={11} /> : <ChevronRight size={11} />}
        <span className="uppercase tracking-wide font-semibold">{label}</span>
      </button>
      {open && (
        <div
          style={{
            marginTop: '0.35rem',
            borderRadius: '0.5rem',
            overflow: 'hidden',
            border: '1px solid var(--color-border-subtle)',
          }}
        >
          <JsonViewer value={value} maxHeight={maxHeight} />
        </div>
      )}
    </div>
  )
}

/* ─── Eval Result Panel ─────────────────────────────────────────── */

interface EvalResultPanelProps {
  pred: string
  gold: string
  nScore: number
  score: Record<string, unknown>
  metadata: unknown
  threshold: number
  showPred: boolean
}

function EvalResultPanel({ pred, gold, nScore, score, metadata, threshold, showPred }: EvalResultPanelProps) {
  const { t } = useLocale()

  const isSameAsGenerated = pred === '*Same as Generated*' || pred === ''

  const metaStr = (() => {
    if (metadata == null) return ''
    if (typeof metadata === 'object') return JSON.stringify(metadata)
    return String(metadata)
  })()
  const hasMetadata = metaStr && metaStr !== '{}' && metaStr !== 'null'

  return (
    <div
      style={{
        borderRadius: '0.875rem',
        border: '1px solid var(--color-border-subtle)',
        background: 'rgba(255,255,255,0.02)',
        overflow: 'hidden',
        animation: 'fadeInUp 300ms ease-out 160ms both',
      }}
    >
      {/* Header */}
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: '0.4rem',
          padding: '0.5rem 1rem',
          borderBottom: '1px solid var(--color-border-subtle)',
          background: 'rgba(255,255,255,0.02)',
        }}
      >
        <ClipboardCheck size={13} style={{ color: 'var(--color-ink-muted)', opacity: 0.6 }} />
        <span
          style={{
            fontSize: '0.65rem',
            fontWeight: 700,
            textTransform: 'uppercase',
            letterSpacing: '0.1em',
            color: 'var(--color-ink-muted)',
            opacity: 0.7,
          }}
        >
          {t('prediction.evalResult')}
        </span>
      </div>

      {/* 3-column grid */}
      <div
        style={{
          display: 'grid',
          gridTemplateColumns: showPred ? 'minmax(80px,auto) 1fr minmax(100px,auto)' : '1fr minmax(100px,auto)',
          gap: '0',
          padding: '0.75rem 1rem',
        }}
      >
        {/* PRED column */}
        {showPred && (
          <div style={{ paddingRight: '1rem', borderRight: '1px solid var(--color-border-subtle)' }}>
            <div
              style={{
                fontSize: '0.65rem',
                fontWeight: 700,
                textTransform: 'uppercase',
                letterSpacing: '0.08em',
                color: '#a78bfa',
                opacity: 0.8,
                marginBottom: '0.35rem',
              }}
            >
              {t('prediction.pred')}
            </div>
            {isSameAsGenerated ? (
              <span style={{ fontSize: '0.75rem', color: 'var(--color-ink-muted)', opacity: 0.5, fontStyle: 'italic' }}>
                = Generated
              </span>
            ) : (
              <div style={{ fontSize: '0.875rem' }}>
                <MarkdownRenderer content={pred} />
              </div>
            )}
          </div>
        )}

        {/* EXPECTED ANSWER column */}
        <div
          style={{
            padding: showPred ? '0 1rem' : '0 1rem 0 0',
            borderRight: '1px solid var(--color-border-subtle)',
          }}
        >
          <div
            style={{
              fontSize: '0.65rem',
              fontWeight: 700,
              textTransform: 'uppercase',
              letterSpacing: '0.08em',
              color: '#fbbf24',
              opacity: 0.9,
              marginBottom: '0.35rem',
            }}
          >
            {t('prediction.expectedAnswer')}
          </div>
          <div style={{ fontSize: '0.875rem' }}>
            <MarkdownRenderer content={gold} />
          </div>
        </div>

        {/* SCORE column */}
        <div style={{ paddingLeft: '1rem' }}>
          <div
            style={{
              fontSize: '0.65rem',
              fontWeight: 700,
              textTransform: 'uppercase',
              letterSpacing: '0.08em',
              color: '#22d3ee',
              opacity: 0.9,
              marginBottom: '0.35rem',
            }}
          >
            {t('prediction.score')}
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.25rem' }}>
            <ScoreBadge score={nScore} threshold={threshold} />
            <span style={{ fontSize: '0.65rem', color: 'var(--color-ink-muted)', opacity: 0.5 }}>
              thr: {threshold}
            </span>
          </div>
        </div>
      </div>

      {/* Collapsible JSON sections */}
      {(Object.keys(score).length > 0 || hasMetadata) && (
        <div
          style={{
            padding: '0 1rem 0.75rem',
            display: 'flex',
            flexDirection: 'column',
            gap: '0.25rem',
          }}
        >
          {Object.keys(score).length > 0 && (
            <CollapsibleJson
              label={t('prediction.scoreJson')}
              value={score}
              maxHeight={200}
            />
          )}
          {hasMetadata && (
            <CollapsibleJson
              label={t('prediction.metadata')}
              value={metadata}
              maxHeight={250}
            />
          )}
        </div>
      )}
    </div>
  )
}

/* ─── main export ─────────────────────────────────────────────── */

export default function ChatView({ prediction, threshold = 0.99 }: Props) {
  const isSystemMsg = hasSystemPrompt(prediction.Input)
  const { system, user } = isSystemMsg ? parseSystemUser(prediction.Input) : { system: '', user: prediction.Input }

  const showPred =
    prediction.Pred &&
    prediction.Pred !== '*Same as Generated*' &&
    prediction.Generated &&
    prediction.Pred.trim() !== prediction.Generated.trim()

  return (
    <div className="flex flex-col gap-4 py-2">
      {/* System prompt banner */}
      {system && <SystemBanner content={system} />}

      {/* User message */}
      <UserBubble content={user || prediction.Input} />

      {/* Assistant: Generated + PerfChip */}
      {prediction.Generated && (
        <AssistantBubble
          content={prediction.Generated}
          perfMetrics={prediction.PerfMetrics}
        />
      )}

      {/* Divider */}
      <div style={{ borderTop: '1px solid var(--color-border-subtle)' }} />

      {/* Eval Result Panel */}
      <EvalResultPanel
        pred={prediction.Pred}
        gold={prediction.Gold}
        nScore={prediction.NScore}
        score={prediction.Score}
        metadata={prediction.Metadata}
        threshold={threshold}
        showPred={!!showPred}
      />
    </div>
  )
}
