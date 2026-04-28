import React, { useState, useCallback, useEffect, useRef } from 'react'
import { createPortal } from 'react-dom'
import { User, Bot, Shield, Wrench, ChevronDown, ChevronRight, ClipboardCheck, Copy, Check, X, Gauge, Target, Scissors, FileJson, Database } from 'lucide-react'
import type { PredictionRow, ChatMessage, SamplePerfMetrics, ContentBlock } from '@/api/types'
import { useLocale } from '@/contexts/LocaleContext'
import MarkdownRenderer from '@/components/common/MarkdownRenderer'
import ScoreBadge from '@/components/common/ScoreBadge'
import JsonViewer from '@/components/common/JsonViewer'
import PerfChip from './PerfChip'

interface Props {
  prediction: PredictionRow
  threshold?: number
  highlightMsgId?: string
}

/* ─── helpers (legacy string-based fallback) ──────────────────── */

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
  const sysMatch = input.match(/<\|system\|>([\s\S]*?)(?:<\|user\|>|$)/i)
  const userMatch = input.match(/<\|user\|>([\s\S]*?)(?:<\|assistant\|>|$)/i)
  if (sysMatch) {
    return {
      system: sysMatch[1].trim(),
      user: userMatch ? userMatch[1].trim() : input.replace(/<\|system\|>[\s\S]*?<\|user\|>/i, '').trim(),
    }
  }
  const bracketMatch = input.match(/^\[system\]([\s\S]*?)(?:\[user\]|$)/i)
  if (bracketMatch) {
    return {
      system: bracketMatch[1].trim(),
      user: input.replace(/^\[system\][\s\S]*?(?:\[user\])/i, '').trim(),
    }
  }
  const colonMatch = input.match(/^system:\s*([\s\S]*?)(?:\nuser:|$)/i)
  if (colonMatch) {
    return {
      system: colonMatch[1].trim(),
      user: input.replace(/^system:\s*[\s\S]*?\nuser:\s*/i, '').trim(),
    }
  }
  return { system: '', user: input }
}

/** Extract plain text from string or ContentBlock[] for clipboard copy.
 *
 * For multimodal blocks (image / audio / video) a short placeholder is
 * included so copied text stays meaningful even without media.
 */
function contentToText(content: string | ContentBlock[]): string {
  if (typeof content === 'string') return content
  return content
    .map(b => {
      if (b.type === 'text') return b.text ?? ''
      if (b.type === 'reasoning') return b.reasoning ?? ''
      if (b.type === 'image') return '[image]'
      if (b.type === 'audio') return '[audio]'
      if (b.type === 'video') return '[video]'
      return ''
    })
    .join('\n\n')
    .trim()
}

function extractToolCalls(text: string): { before: string; calls: string[]; after: string } {
  const calls: string[] = []
  let remaining = text

  const toolCallRegex = /<tool_call>([\s\S]*?)<\/tool_call>/g
  remaining = remaining.replace(toolCallRegex, (match) => {
    calls.push(match)
    return ''
  })

  const jsonFnRegex = /\{"name"\s*:\s*"[^"]+"\s*,\s*"arguments"\s*:\s*\{[\s\S]*?\}\s*\}/g
  remaining = remaining.replace(jsonFnRegex, (match) => {
    calls.push(match)
    return ''
  })

  return { before: remaining.trim(), calls, after: '' }
}

/* ─── Multimodal block renderers ────────────────────────────────── */

/**
 * Renders a single image content block.
 * Supports both absolute URLs and base64 data-URIs produced by the backend.
 * Click thumbnail to open a full-screen lightbox (portal-mounted to avoid
 * overflow-hidden clipping from ancestor containers).
 */
function ImageBlock({ src }: { src: string }) {
  const [open, setOpen] = useState(false)
  // Normalise: if src is raw base64 (no scheme), prefix with a data URI
  const imgSrc = src.startsWith('http') || src.startsWith('data:')
    ? src
    : `data:image/jpeg;base64,${src}`
  return (
    <div style={{ marginTop: '0.5rem', marginBottom: '0.25rem' }}>
      <img
        src={imgSrc}
        alt=""
        onClick={() => setOpen(true)}
        className="cursor-zoom-in hover:scale-[1.02] transition-transform"
        style={{
          maxWidth: '100%',
          maxHeight: '360px',
          borderRadius: '0.5rem',
          border: '1px solid var(--color-border-subtle)',
          display: 'block',
        }}
      />
      {open && createPortal(
        <div
          className="fixed inset-0 z-[9999] flex items-center justify-center"
          style={{ background: 'var(--overlay-bg)', backdropFilter: 'blur(6px)' }}
          onClick={() => setOpen(false)}
        >
          <div
            className="relative max-w-[90vw] max-h-[90vh]"
            onClick={(e) => e.stopPropagation()}
          >
            <button
              onClick={() => setOpen(false)}
              className="absolute -top-3 -right-3 z-10 rounded-full p-1 bg-[var(--color-surface)] border border-[var(--color-border)] hover:bg-[var(--color-surface-hover)] transition-colors"
            >
              <X size={16} />
            </button>
            <img
              src={imgSrc}
              alt=""
              className="max-w-full max-h-[85vh] rounded-xl object-contain shadow-2xl"
            />
          </div>
        </div>,
        document.body
      )}
    </div>
  )
}

/**
 * Renders a single audio content block.
 * Supports both URLs and base64 data-URIs.
 */
function AudioBlock({ src, format }: { src: string; format?: string }) {
  const mimeType = format === 'mp3' ? 'audio/mpeg' : format === 'wav' ? 'audio/wav' : 'audio/mpeg'
  const audioSrc = src.startsWith('http') || src.startsWith('data:')
    ? src
    : `data:${mimeType};base64,${src}`
  return (
    <div style={{ marginTop: '0.5rem', marginBottom: '0.25rem' }}>
      {/* eslint-disable-next-line jsx-a11y/media-has-caption */}
      <audio
        controls
        src={audioSrc}
        style={{
          width: '100%',
          borderRadius: '0.4rem',
        }}
      />
    </div>
  )
}

/**
 * Render a list of ContentBlocks into React nodes.
 * This is the primary multimodal rendering path used by both UserBubble
 * and AssistantBubble when content is an array.
 *
 * Layout convention:
 *  - reasoning blocks   → collapsible ReasoningBlock (above text, assistant-only)
 *  - text blocks        → MarkdownRenderer
 *  - image blocks       → inline <img>
 *  - audio blocks       → HTML5 <audio> player
 *  - video/data blocks  → plain-text placeholder (not yet fully supported)
 */
function renderContentBlocks(
  blocks: ContentBlock[],
  opts: { includeReasoning?: boolean } = {},
): React.ReactNode[] {
  const nodes: React.ReactNode[] = []
  blocks.forEach((b, i) => {
    if (b.type === 'reasoning' && opts.includeReasoning) {
      nodes.push(<ReasoningBlock key={`r${i}`} text={b.reasoning ?? ''} />)
    } else if (b.type === 'text') {
      if (b.text) nodes.push(<MarkdownRenderer key={`t${i}`} content={b.text} />)
    } else if (b.type === 'image') {
      if (b.image) nodes.push(<ImageBlock key={`img${i}`} src={b.image} />)
    } else if (b.type === 'audio') {
      if (b.audio) nodes.push(<AudioBlock key={`aud${i}`} src={b.audio} format={b.format} />)
    } else if (b.type === 'video') {
      // Video rendering not yet fully supported – show a placeholder
      nodes.push(
        <span key={`vid${i}`} style={{ fontSize: '0.8rem', opacity: 0.6, fontStyle: 'italic' }}>
          [video]
        </span>
      )
    }
    // 'data' blocks are intentionally skipped (opaque provider payload)
  })
  return nodes
}

/* ─── shared sub-components ───────────────────────────────────── */

type CopyVariant = 'green' | 'indigo' | 'neutral'

function CopyButton({ text, variant = 'green' }: { text: string; variant?: CopyVariant }) {
  const { t } = useLocale()
  const [copied, setCopied] = useState(false)
  const handleCopy = useCallback(async () => {
    try {
      await navigator.clipboard.writeText(text)
      setCopied(true)
      setTimeout(() => setCopied(false), 1800)
    } catch {
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

  const activeColor = variant === 'green' ? 'var(--bubble-bot-color)' : variant === 'indigo' ? 'var(--bubble-user-color)' : 'var(--text-muted)'
  const borderColor =
    variant === 'green'
      ? 'var(--bubble-bot-border)'
      : variant === 'indigo'
        ? 'var(--bubble-user-border)'
        : 'var(--color-border-subtle)'
  const bgColor =
    variant === 'green'
      ? 'var(--bubble-bot-bg)'
      : variant === 'indigo'
        ? 'var(--bubble-user-bg)'
        : 'var(--bubble-system-bg)'

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
        color: copied ? activeColor : 'var(--text-muted)',
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

/** Small chip showing a message's 8-char id, with copy-to-clipboard on click. */
function MsgIdChip({ msgId, variant = 'neutral' }: { msgId: string; variant?: CopyVariant }) {
  const { t } = useLocale()
  const [copied, setCopied] = useState(false)
  const handleCopy = useCallback(async () => {
    try {
      await navigator.clipboard.writeText(msgId)
    } catch {
      const el = document.createElement('textarea')
      el.value = msgId
      document.body.appendChild(el)
      el.select()
      document.execCommand('copy')
      document.body.removeChild(el)
    }
    setCopied(true)
    setTimeout(() => setCopied(false), 1800)
  }, [msgId])

  const activeColor = variant === 'green' ? 'var(--bubble-bot-color)' : variant === 'indigo' ? 'var(--bubble-user-color)' : 'var(--accent)'
  const borderColor = variant === 'green' ? 'var(--bubble-bot-border)' : variant === 'indigo' ? 'var(--bubble-user-border)' : 'var(--color-border-subtle)'
  const bgColor = variant === 'green' ? 'var(--bubble-bot-bg)' : variant === 'indigo' ? 'var(--bubble-user-bg)' : 'var(--bubble-system-bg)'

  return (
    <button
      onClick={handleCopy}
      title={t('prediction.copyMsgId')}
      style={{
        display: 'inline-flex',
        alignItems: 'center',
        gap: '0.25rem',
        padding: '0.2rem 0.5rem',
        borderRadius: '0.4rem',
        border: `1px solid ${borderColor}`,
        background: bgColor,
        color: copied ? activeColor : 'var(--text-muted)',
        fontSize: '0.68rem',
        fontFamily: 'monospace',
        cursor: 'pointer',
        transition: 'all 0.15s',
      }}
    >
      {copied ? <Check size={11} /> : <Copy size={11} />}
      <span>{msgId}</span>
    </button>
  )
}

function SystemBanner({ content }: { content: string }) {
  const { t } = useLocale()
  const [expanded, setExpanded] = useState(false)
  const preview = content.length > 120 ? content.slice(0, 120) + '…' : content
  return (
    <div
      className="rounded-xl border px-4 py-3 text-sm flex gap-3 items-start cursor-pointer select-none"
      style={{
        background: 'var(--bubble-system-bg)',
        borderColor: 'var(--bubble-system-border)',
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
  } catch {
    /* noop */
  }
  return (
    <div className="mt-2 rounded-xl border border-[var(--color-border)] overflow-hidden text-xs">
      <button
        onClick={() => setOpen((v) => !v)}
        className="flex items-center gap-2 w-full px-3 py-2 text-left font-mono hover:bg-[var(--color-surface-hover)] transition-colors"
        style={{ background: 'var(--bubble-user-bg)', color: 'var(--color-primary)' }}
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

/**
 * User message bubble.
 *
 * Current path  : content is `string | ContentBlock[]` from structured cache.
 *   - string    → rendered as markdown (legacy / simple)
 *   - ContentBlock[] → rendered via `renderContentBlocks` (multimodal path)
 *
 * Legacy fallback: caller passes a plain string (from `prediction.Input`).
 */
function UserBubble({ content, turnBadge, msgId, highlightId }: { content: string | ContentBlock[]; turnBadge?: string; msgId?: string; highlightId?: string }) {
  const copyText = contentToText(content)
  const isHighlighted = !!(msgId && highlightId && msgId.startsWith(highlightId))
  const ref = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (isHighlighted && ref.current) {
      ref.current.scrollIntoView({ behavior: 'smooth', block: 'center' })
    }
  }, [isHighlighted])

  return (
    <div ref={ref} className="flex gap-3 items-start" style={{ animation: 'fadeInUp 300ms ease-out both' }}>
      <div
        className="shrink-0 w-8 h-8 rounded-full flex items-center justify-center mt-1"
        style={{ background: 'var(--bubble-user-icon-bg)', border: '1px solid var(--bubble-user-icon-border)' }}
      >
        <User size={15} style={{ color: 'var(--bubble-user-color)' }} />
      </div>
      <div className="bubble-wrap" style={{ maxWidth: '80%' }}>
        {turnBadge && (
          <div
            style={{
              fontSize: '0.6rem',
              fontWeight: 600,
              textTransform: 'uppercase',
              letterSpacing: '0.08em',
              color: 'var(--bubble-user-color)',
              opacity: 0.7,
              marginBottom: '0.2rem',
            }}
          >
            {turnBadge}
          </div>
        )}
        <div
          className="flex-1 rounded-2xl px-4 py-3 text-sm shadow"
          style={{
            background: isHighlighted ? 'var(--bubble-user-bg-hl)' : 'var(--bubble-user-bg)',
            border: isHighlighted ? '2px solid var(--bubble-user-border-hl)' : '1px solid var(--bubble-user-border)',
            boxShadow: '0 2px 8px var(--bubble-user-bg)',
            transition: 'border 0.4s, background 0.4s',
          }}
        >
          <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '0.25rem', marginBottom: '0.4rem' }}>
            {msgId && <MsgIdChip msgId={msgId} variant="indigo" />}
            <CopyButton text={copyText} variant="indigo" />
          </div>
          {Array.isArray(content)
            ? renderContentBlocks(content)
            : <MarkdownRenderer content={content} />}
        </div>
      </div>
    </div>
  )
}

interface AssistantBubbleProps {
  content: string | ContentBlock[]
  perfMetrics?: SamplePerfMetrics | null
  turnBadge?: string
  msgId?: string
  highlightId?: string
}

/** Collapsible reasoning block rendered above the main answer. */
function ReasoningBlock({ text }: { text: string }) {
  const { t } = useLocale()
  const [open, setOpen] = useState(false)
  return (
    <div
      style={{
        marginBottom: '0.5rem',
        borderRadius: '0.75rem',
        border: '1px solid var(--bubble-reasoning-border)',
        background: 'var(--bubble-reasoning-bg)',
        overflow: 'hidden',
      }}
    >
      <button
        onClick={() => setOpen(o => !o)}
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: '0.35rem',
          width: '100%',
          padding: '0.4rem 0.75rem',
          background: 'none',
          border: 'none',
          cursor: 'pointer',
          color: 'var(--bubble-bot-color)',
          fontSize: '0.72rem',
          fontWeight: 600,
          textTransform: 'uppercase',
          letterSpacing: '0.06em',
        }}
      >
        {open ? <ChevronDown size={12} /> : <ChevronRight size={12} />}
        {open ? t('prediction.hideReasoning') : t('prediction.showReasoning')}
      </button>
      {open && (
        <div
          style={{
            padding: '0.5rem 0.75rem 0.75rem',
            fontSize: '0.8rem',
            color: 'var(--text)',
            borderTop: '1px solid var(--bubble-reasoning-border)',
          }}
        >
          <MarkdownRenderer content={text} />
        </div>
      )}
    </div>
  )
}

/**
 * Assistant message bubble.
 *
 * Current path  : content is `string | ContentBlock[]`.
 *   - ContentBlock[] → `renderContentBlocks` handles reasoning + text + image + audio
 *     Tool-call extraction still runs on the first text block for compatibility.
 *
 * Legacy fallback: content is a plain string (from `score.prediction` in old caches).
 */
function AssistantBubble({ content, perfMetrics, turnBadge, msgId, highlightId }: AssistantBubbleProps) {
  // Normalise to blocks so rendering logic is uniform
  const blocks: ContentBlock[] = typeof content === 'string'
    ? [{ type: 'text', text: content }]
    : content

  // Extract tool-call tags from the first text block only (legacy pattern support)
  const firstTextBlock = blocks.find(b => b.type === 'text')
  const { calls, before } = extractToolCalls(firstTextBlock?.text ?? '')

  // Rebuild blocks with tool-calls stripped from the first text block
  const displayBlocks: ContentBlock[] = blocks.map(b =>
    b === firstTextBlock ? { ...b, text: before } : b
  )

  const copyText = contentToText(content)
  const isHighlighted = !!(msgId && highlightId && msgId.startsWith(highlightId))
  const ref = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (isHighlighted && ref.current) {
      ref.current.scrollIntoView({ behavior: 'smooth', block: 'center' })
    }
  }, [isHighlighted])

  return (
    <div
      ref={ref}
      className="flex gap-3 items-start justify-end"
      style={{ animation: 'fadeInUp 300ms ease-out 80ms both' }}
    >
      <div className="bubble-wrap" style={{ maxWidth: '80%' }}>
        {turnBadge && (
          <div
            style={{
              fontSize: '0.6rem',
              fontWeight: 600,
              textTransform: 'uppercase',
              letterSpacing: '0.08em',
              color: 'var(--bubble-bot-color)',
              opacity: 0.7,
              marginBottom: '0.2rem',
              textAlign: 'right',
            }}
          >
            {turnBadge}
          </div>
        )}
        <div
          className="flex-1 rounded-2xl px-4 py-3 text-sm"
          style={{
            background: isHighlighted ? 'var(--bubble-bot-bg-hl)' : 'var(--bubble-bot-bg)',
            border: isHighlighted ? '2px solid var(--bubble-bot-border-hl)' : '1px solid var(--bubble-bot-border)',
            boxShadow: '0 2px 8px var(--bubble-bot-bg)',
            transition: 'border 0.4s, background 0.4s',
          }}
        >
          <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '0.25rem', marginBottom: '0.4rem' }}>
            {msgId && <MsgIdChip msgId={msgId} variant="green" />}
            <CopyButton text={copyText} variant="green" />
          </div>
          {renderContentBlocks(displayBlocks, { includeReasoning: true })}
          {calls.map((c, i) => (
            <ToolCallBlock key={i} raw={c} />
          ))}
          {perfMetrics && <PerfChip metrics={perfMetrics} variant="green" />}
        </div>
      </div>
      <div
        className="shrink-0 w-8 h-8 rounded-full flex items-center justify-center mt-1"
        style={{ background: 'var(--bubble-bot-icon-bg)', border: '1px solid var(--bubble-bot-icon-border)' }}
      >
        <Bot size={15} style={{ color: 'var(--bubble-bot-color)' }} />
      </div>
    </div>
  )
}

/** Collapsible tool-result message bubble */
function ToolResultBubble({ content }: { content: string }) {
  const { t } = useLocale()
  const [open, setOpen] = useState(false)
  return (
    <div className="flex gap-3 items-start" style={{ animation: 'fadeInUp 300ms ease-out both' }}>
      <div
        className="shrink-0 w-8 h-8 rounded-full flex items-center justify-center mt-1"
        style={{ background: 'var(--bubble-tool-icon-bg)', border: '1px solid var(--bubble-tool-icon-border)' }}
      >
        <Wrench size={14} style={{ color: 'var(--bubble-tool-color)' }} />
      </div>
      <div style={{ maxWidth: '80%', flex: 1 }}>
        <div
          className="rounded-xl border overflow-hidden text-xs"
          style={{ borderColor: 'var(--bubble-tool-border)', background: 'var(--bubble-tool-bg)' }}
        >
          <button
            onClick={() => setOpen((v) => !v)}
            className="flex items-center gap-2 w-full px-3 py-2 text-left"
            style={{ color: 'var(--bubble-tool-color)' }}
          >
            {open ? <ChevronDown size={12} /> : <ChevronRight size={12} />}
            <span className="font-semibold uppercase tracking-wide" style={{ fontSize: '0.65rem' }}>
              {t('prediction.toolResult')}
            </span>
          </button>
          {open && (
            <pre
              className="px-3 py-2 text-xs font-mono whitespace-pre-wrap break-all overflow-auto max-h-[200px]"
              style={{ background: 'var(--color-surface)', color: 'var(--color-ink-muted)' }}
            >
              {content}
            </pre>
          )}
        </div>
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
  icon?: React.ReactNode
}

function CollapsibleJson({ label, value, maxHeight = 200, defaultOpen = false, icon }: CollapsibleJsonProps) {
  const [open, setOpen] = useState(defaultOpen)

  const isEmpty =
    value == null ||
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
        {icon && <span style={{ display: 'inline-flex', alignItems: 'center' }}>{icon}</span>}
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
        border: '1px solid var(--border-md)',
        background: 'var(--bg-card2)',
        overflow: 'hidden',
        boxShadow: 'var(--shadow-sm)',
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
          borderBottom: '1px solid var(--border-md)',
          background: 'var(--bg-deep)',
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
          <div style={{ paddingRight: '1rem', borderRight: '1px solid var(--border-md)' }}>
            <div
              style={{
                fontSize: '0.65rem',
                fontWeight: 700,
                textTransform: 'uppercase',
                letterSpacing: '0.08em',
                color: 'var(--purple)',
                opacity: 0.8,
                marginBottom: '0.35rem',
                display: 'flex',
                alignItems: 'center',
                gap: '0.25rem',
              }}
            >
              <Scissors size={11} />
              {t('prediction.extractedAnswer')}
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
            borderRight: '1px solid var(--border-md)',
          }}
        >
          <div
            style={{
              fontSize: '0.65rem',
              fontWeight: 700,
              textTransform: 'uppercase',
              letterSpacing: '0.08em',
              color: 'var(--yellow)',
              opacity: 0.9,
              marginBottom: '0.35rem',
              display: 'flex',
              alignItems: 'center',
              gap: '0.25rem',
            }}
          >
            <Target size={11} />
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
              color: 'var(--cyan)',
              opacity: 0.9,
              marginBottom: '0.35rem',
              display: 'flex',
              alignItems: 'center',
              gap: '0.25rem',
            }}
          >
            <Gauge size={11} />
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
            <CollapsibleJson label={t('prediction.scoreJson')} value={score} maxHeight={200} defaultOpen={true} icon={<FileJson size={11} />} />
          )}
          {hasMetadata && (
            <CollapsibleJson label={t('prediction.metadata')} value={metadata} maxHeight={250} defaultOpen={true} icon={<Database size={11} />} />
          )}
        </div>
      )}
    </div>
  )
}

/* ─── Structured multi-turn renderer ───────────────────────────── */

function StructuredMessages({
  messages,
  isMultiTurn,
  highlightId,
}: {
  messages: ChatMessage[]
  isMultiTurn: boolean
  highlightId?: string
}) {
  const { t } = useLocale()

  // Compute per-role turn index for badge labeling (only user & assistant)
  const turnCounters = { user: 0, assistant: 0 }

  return (
    <>
      {messages.map((msg, idx) => {
        if (msg.role === 'system') {
          return <SystemBanner key={idx} content={contentToText(msg.content)} />
        }

        if (msg.role === 'user') {
          turnCounters.user += 1
          const badge =
            isMultiTurn
              ? t('prediction.turnN').replace('${n}', String(turnCounters.user))
              : undefined
          return <UserBubble key={idx} content={msg.content} turnBadge={badge} msgId={msg.id} highlightId={highlightId} />
        }

        if (msg.role === 'assistant') {
          turnCounters.assistant += 1
          const badge =
            isMultiTurn
              ? t('prediction.turnN').replace('${n}', String(turnCounters.assistant))
              : undefined
          return (
            <AssistantBubble
              key={idx}
              content={msg.content}
              perfMetrics={msg.perf_metrics}
              turnBadge={badge}
              msgId={msg.id}
              highlightId={highlightId}
            />
          )
        }

        if (msg.role === 'tool') {
          return <ToolResultBubble key={idx} content={contentToText(msg.content)} />
        }

        return null
      })}
    </>
  )
}

/* ─── main export ─────────────────────────────────────────────── */

export default function ChatView({ prediction, threshold = 0.99, highlightMsgId }: Props) {
  const showPred =
    prediction.Pred &&
    prediction.Pred !== '*Same as Generated*' &&
    prediction.Generated &&
    prediction.Pred.trim() !== prediction.Generated.trim()

  // Determine if we have structured messages and if it's truly multi-turn
  const messages = prediction.Messages
  const hasStructured = !!(messages && messages.length > 0)

  // Multi-turn = more than one user message (excluding system)
  const userCount = hasStructured ? messages!.filter((m) => m.role === 'user').length : 0
  const isMultiTurn = userCount > 1

  return (
    <div className="flex flex-col gap-4 py-2">
      {hasStructured ? (
        /* ── Structured rendering (new-format caches) ── */
        <StructuredMessages messages={messages!} isMultiTurn={isMultiTurn} highlightId={highlightMsgId} />
      ) : (
        /* ── Legacy string-based fallback ── */
        (() => {
          const isSystemMsg = hasSystemPrompt(prediction.Input)
          const { system, user } = isSystemMsg
            ? parseSystemUser(prediction.Input)
            : { system: '', user: prediction.Input }

          return (
            <>
              {system && <SystemBanner content={system} />}
              <UserBubble content={user || prediction.Input} />
              {prediction.Generated && (
                <AssistantBubble content={prediction.Generated} perfMetrics={prediction.PerfMetrics} />
              )}
            </>
          )
        })()
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
