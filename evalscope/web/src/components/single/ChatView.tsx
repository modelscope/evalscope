import React, { useState, useCallback, useEffect, useRef, useMemo } from 'react'
import { createPortal } from 'react-dom'
import {
  User,
  Bot,
  Shield,
  Wrench,
  ChevronDown,
  ChevronRight,
  Copy,
  Check,
  X,
  Gauge,
  Target,
  Scissors,
  FileJson,
  Database,
  Sparkles,
  Cpu,
  AlertTriangle,
  ClipboardCheck,
} from 'lucide-react'
import type {
  PredictionRow,
  ChatMessage,
  ContentBlock,
  AgentTrace,
  AgentTraceEvent,
  ToolCall,
} from '@/api/types'
import { useLocale } from '@/contexts/LocaleContext'
import MarkdownRenderer from '@/components/common/MarkdownRenderer'
import ScoreBadge from '@/components/common/ScoreBadge'
import JsonViewer from '@/components/common/JsonViewer'

interface Props {
  prediction: PredictionRow
  threshold?: number
  highlightMsgId?: string
}

/* ─── Helpers ─────────────────────────────────────────────────── */

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

/** Extract plain text from string or ContentBlock[] for clipboard copy. */
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

function fmtMs(ms: number | null | undefined): string {
  if (ms == null) return ''
  if (ms < 1000) return `${ms.toFixed(0)}ms`
  return `${(ms / 1000).toFixed(1)}s`
}

function fmtTokens(n: number | null | undefined): string | null {
  if (n == null) return null
  if (n >= 1000) return `${(n / 1000).toFixed(1)}k`
  return String(n)
}

/** One-line preview for arguments JSON (truncate). */
function argsPreview(args: unknown, max = 100): string {
  if (args == null) return ''
  let s: string
  try {
    s = typeof args === 'string' ? args : JSON.stringify(args)
  } catch {
    s = String(args)
  }
  s = s.replace(/\s+/g, ' ').trim()
  return s.length > max ? s.slice(0, max) + '…' : s
}

/* ─── Role palette ──────────────────────────────────────────── */

type Role = 'system' | 'user' | 'assistant' | 'tool'

interface RolePalette {
  icon: React.FC<{ size?: number; style?: React.CSSProperties; className?: string }>
  barColor: string
  tintBg: string
  tintBgHl: string
  borderHl: string
  labelColor: string
  label: string
}

function rolePalette(role: Role, t: (k: string) => string): RolePalette {
  switch (role) {
    case 'user':
      return {
        icon: User,
        barColor: 'var(--bubble-user-color)',
        tintBg: 'var(--bubble-user-bg)',
        tintBgHl: 'var(--bubble-user-bg-hl)',
        borderHl: 'var(--bubble-user-border-hl)',
        labelColor: 'var(--bubble-user-color)',
        label: 'User',
      }
    case 'assistant':
      return {
        icon: Bot,
        barColor: 'var(--bubble-bot-color)',
        tintBg: 'transparent',
        tintBgHl: 'var(--bubble-bot-bg-hl)',
        borderHl: 'var(--bubble-bot-border-hl)',
        labelColor: 'var(--bubble-bot-color)',
        label: 'Assistant',
      }
    case 'tool':
      return {
        icon: Wrench,
        barColor: 'var(--bubble-tool-color)',
        tintBg: 'var(--bubble-tool-bg)',
        tintBgHl: 'var(--bubble-tool-bg)',
        borderHl: 'var(--bubble-tool-border)',
        labelColor: 'var(--bubble-tool-color)',
        label: t('prediction.toolResult'),
      }
    case 'system':
    default:
      return {
        icon: Shield,
        barColor: 'var(--text-muted)',
        tintBg: 'var(--bubble-system-bg)',
        tintBgHl: 'var(--bubble-system-bg)',
        borderHl: 'var(--bubble-system-border)',
        labelColor: 'var(--text-muted)',
        label: t('prediction.systemPrompt'),
      }
  }
}

/* ─── Multimodal block renderers ────────────────────────────── */

/** Clickable image thumbnail with fullscreen lightbox. */
function ImageBlock({ src }: { src: string }) {
  const [open, setOpen] = useState(false)
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
          <div className="relative max-w-[90vw] max-h-[90vh]" onClick={e => e.stopPropagation()}>
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

function AudioBlock({ src, format }: { src: string; format?: string }) {
  const mimeType = format === 'mp3' ? 'audio/mpeg' : format === 'wav' ? 'audio/wav' : 'audio/mpeg'
  const audioSrc = src.startsWith('http') || src.startsWith('data:')
    ? src
    : `data:${mimeType};base64,${src}`
  return (
    <div style={{ marginTop: '0.5rem', marginBottom: '0.25rem' }}>
      {/* eslint-disable-next-line jsx-a11y/media-has-caption */}
      <audio controls src={audioSrc} style={{ width: '100%', borderRadius: '0.4rem' }} />
    </div>
  )
}

/** Collapsible reasoning block rendered above the main answer. */
function ReasoningBlock({ text }: { text: string }) {
  const { t } = useLocale()
  const [open, setOpen] = useState(false)
  return (
    <div
      style={{
        marginBottom: '0.5rem',
        borderRadius: '0.5rem',
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
          padding: '0.35rem 0.7rem',
          background: 'none',
          border: 'none',
          cursor: 'pointer',
          color: 'var(--bubble-bot-color)',
          fontSize: '0.7rem',
          fontWeight: 600,
          letterSpacing: '0.04em',
        }}
      >
        {open ? <ChevronDown size={12} /> : <ChevronRight size={12} />}
        {open ? t('prediction.hideReasoning') : t('prediction.showReasoning')}
        <span style={{ opacity: 0.5, fontWeight: 400 }}>· {text.length} chars</span>
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

/** Render ContentBlock[] into React nodes. */
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
      nodes.push(
        <span key={`vid${i}`} style={{ fontSize: '0.8rem', opacity: 0.6, fontStyle: 'italic' }}>
          [video]
        </span>
      )
    }
  })
  return nodes
}

/* ─── Shared UI: copy buttons, chips ────────────────────────── */

function CopyIconButton({ text, title }: { text: string; title?: string }) {
  const { t } = useLocale()
  const [copied, setCopied] = useState(false)
  const handleCopy = useCallback(async (e: React.MouseEvent) => {
    e.stopPropagation()
    try {
      await navigator.clipboard.writeText(text)
    } catch {
      const el = document.createElement('textarea')
      el.value = text
      document.body.appendChild(el)
      el.select()
      document.execCommand('copy')
      document.body.removeChild(el)
    }
    setCopied(true)
    setTimeout(() => setCopied(false), 1500)
  }, [text])
  return (
    <button
      onClick={handleCopy}
      title={title ?? t('prediction.copyContent')}
      style={{
        display: 'inline-flex',
        alignItems: 'center',
        justifyContent: 'center',
        width: 22,
        height: 22,
        padding: 0,
        borderRadius: 4,
        border: 'none',
        background: 'transparent',
        color: copied ? 'var(--bubble-bot-color)' : 'var(--text-muted)',
        cursor: 'pointer',
        opacity: 0.55,
        transition: 'opacity 0.15s, color 0.15s',
      }}
      onMouseEnter={e => (e.currentTarget.style.opacity = '1')}
      onMouseLeave={e => (e.currentTarget.style.opacity = copied ? '1' : '0.55')}
    >
      {copied ? <Check size={13} /> : <Copy size={13} />}
    </button>
  )
}

function MsgIdChip({ msgId }: { msgId: string }) {
  const { t } = useLocale()
  const [copied, setCopied] = useState(false)
  const handleCopy = useCallback(async (e: React.MouseEvent) => {
    e.stopPropagation()
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
    setTimeout(() => setCopied(false), 1500)
  }, [msgId])
  return (
    <button
      onClick={handleCopy}
      title={t('prediction.copyMsgId')}
      style={{
        display: 'inline-flex',
        alignItems: 'center',
        gap: 3,
        padding: '1px 6px',
        borderRadius: 4,
        border: '1px solid var(--color-border-subtle)',
        background: 'var(--bg-deep)',
        color: copied ? 'var(--accent)' : 'var(--text-muted)',
        fontSize: '0.62rem',
        fontFamily: 'var(--font-mono, monospace)',
        cursor: 'pointer',
        opacity: 0.7,
        transition: 'opacity 0.15s, color 0.15s',
      }}
      onMouseEnter={e => (e.currentTarget.style.opacity = '1')}
      onMouseLeave={e => (e.currentTarget.style.opacity = '0.7')}
    >
      {copied ? <Check size={10} /> : <Copy size={10} />}
      <span>{msgId}</span>
    </button>
  )
}

/** Compact perf chip rendered inline inside a message header. */
function HeaderPerfChip({
  latency,
  inTok,
  outTok,
  stopReason,
}: {
  latency?: number | null
  inTok?: number | null
  outTok?: number | null
  stopReason?: string
}) {
  const parts: string[] = []
  if (latency != null) parts.push(fmtMs(latency))
  const inS = fmtTokens(inTok ?? null)
  const outS = fmtTokens(outTok ?? null)
  if (inS != null || outS != null) parts.push(`${inS ?? '0'} → ${outS ?? '0'} tok`)
  if (stopReason) parts.push(`stop: ${stopReason}`)
  if (parts.length === 0) return null
  return (
    <span
      style={{
        fontSize: '0.65rem',
        fontFamily: 'var(--font-mono, monospace)',
        color: 'var(--text-muted)',
        opacity: 0.8,
        whiteSpace: 'nowrap',
      }}
    >
      {parts.join(' · ')}
    </span>
  )
}

/* ─── MessageRow: unified full-width row for all roles ───── */

interface MessageRowProps {
  role: Role
  content: string | ContentBlock[]
  msgId?: string
  model?: string | null
  highlightId?: string
  /** extra right-side header content (e.g. perf chip). */
  headerExtra?: React.ReactNode
  /** inline children below content (e.g. ToolCallsGroup for assistant). */
  children?: React.ReactNode
  /** override role label text. */
  labelOverride?: string
  /** hide role icon & label (used for compact observation). */
  compact?: boolean
  /** tool error, shown as warning banner inside content. */
  toolError?: { type?: string | null; message: string } | null
  /** tool function name (to show in header when role=tool). */
  toolFunction?: string | null
}

function MessageRow({
  role,
  content,
  msgId,
  model,
  highlightId,
  headerExtra,
  children,
  labelOverride,
  compact,
  toolError,
  toolFunction,
}: MessageRowProps) {
  const { t } = useLocale()
  const palette = rolePalette(role, t)
  const RoleIcon = palette.icon
  const copyText = contentToText(content)
  const isHighlighted = !!(msgId && highlightId && msgId.startsWith(highlightId))
  const ref = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (isHighlighted && ref.current) {
      ref.current.scrollIntoView({ behavior: 'smooth', block: 'center' })
    }
  }, [isHighlighted])

  const bgColor = isHighlighted ? palette.tintBgHl : palette.tintBg
  const borderLeft = isHighlighted
    ? `3px solid ${palette.borderHl}`
    : `3px solid ${palette.barColor}`

  return (
    <div
      ref={ref}
      style={{
        display: 'flex',
        width: '100%',
        background: bgColor,
        borderLeft,
        borderRadius: '0.5rem',
        padding: '0.6rem 0.85rem',
        transition: 'background 0.3s, border-color 0.3s',
        animation: 'fadeInUp 240ms ease-out both',
      }}
    >
      <div style={{ flex: 1, minWidth: 0 }}>
        {/* Header row */}
        {!compact && (
          <div
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '0.5rem',
              marginBottom: '0.4rem',
              flexWrap: 'wrap',
            }}
          >
            <RoleIcon size={13} style={{ color: palette.labelColor, flexShrink: 0 }} />
            <span
              style={{
                fontSize: '0.75rem',
                fontWeight: 600,
                color: palette.labelColor,
              }}
            >
              {labelOverride ?? palette.label}
            </span>
            {toolFunction && (
              <span
                style={{
                  padding: '1px 7px',
                  borderRadius: 4,
                  background: 'var(--bg-deep)',
                  border: '1px solid var(--color-border-subtle)',
                  fontSize: '0.65rem',
                  fontFamily: 'var(--font-mono, monospace)',
                  color: 'var(--text-muted)',
                }}
              >
                {toolFunction}
              </span>
            )}
            {model && (
              <span
                style={{
                  padding: '1px 7px',
                  borderRadius: 4,
                  background: 'var(--bg-deep)',
                  border: '1px solid var(--color-border-subtle)',
                  fontSize: '0.65rem',
                  fontFamily: 'var(--font-mono, monospace)',
                  color: 'var(--text-muted)',
                }}
              >
                {model}
              </span>
            )}
            {headerExtra}
            <span style={{ flex: 1 }} />
            {msgId && <MsgIdChip msgId={msgId} />}
            {copyText && <CopyIconButton text={copyText} />}
          </div>
        )}

        {/* Tool error banner */}
        {toolError && (
          <div
            style={{
              marginBottom: '0.4rem',
              padding: '0.4rem 0.6rem',
              borderRadius: '0.4rem',
              background: 'var(--danger-bg)',
              border: '1px solid var(--danger-border, var(--danger))',
              color: 'var(--danger)',
              fontSize: '0.72rem',
              fontFamily: 'var(--font-mono, monospace)',
            }}
          >
            {toolError.type ? `[${toolError.type}] ` : ''}
            {toolError.message}
          </div>
        )}

        {/* Content */}
        <div style={{ fontSize: '0.85rem', lineHeight: 1.55 }}>
          {Array.isArray(content)
            ? renderContentBlocks(content, { includeReasoning: role === 'assistant' })
            : <MarkdownRenderer content={content} />}
        </div>

        {children}
      </div>
    </div>
  )
}

/* ─── Tool message as a compact observation (folded under tool call) */

function ToolObservation({ msg }: { msg: ChatMessage }) {
  const { t } = useLocale()
  const [open, setOpen] = useState(false)
  const text = contentToText(msg.content)
  const preview = text.replace(/\s+/g, ' ').trim()
  const previewShort = preview.length > 140 ? preview.slice(0, 140) + '…' : preview
  const hasError = !!msg.error

  return (
    <div
      style={{
        borderLeft: '2px solid var(--bubble-tool-border)',
        paddingLeft: '0.6rem',
        marginTop: '0.25rem',
      }}
    >
      <button
        onClick={() => setOpen(v => !v)}
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: '0.4rem',
          width: '100%',
          background: 'none',
          border: 'none',
          cursor: 'pointer',
          padding: '0.2rem 0',
          textAlign: 'left',
          color: hasError ? 'var(--danger)' : 'var(--text-muted)',
          fontSize: '0.72rem',
        }}
      >
        {open ? <ChevronDown size={11} /> : <ChevronRight size={11} />}
        <span
          style={{
            display: 'inline-block',
            width: 6,
            height: 6,
            borderRadius: '50%',
            background: hasError ? 'var(--danger)' : 'var(--bubble-bot-color)',
            flexShrink: 0,
          }}
        />
        <span
          style={{
            fontFamily: 'var(--font-mono, monospace)',
            fontSize: '0.7rem',
            overflow: 'hidden',
            textOverflow: 'ellipsis',
            whiteSpace: 'nowrap',
            flex: 1,
          }}
        >
          {hasError
            ? `${t('trace.error')}: ${msg.error?.message ?? ''}`
            : previewShort || t('trace.stdout')}
        </span>
        {msg.id && <span style={{ opacity: 0.4, fontSize: '0.6rem', fontFamily: 'var(--font-mono, monospace)' }}>{msg.id}</span>}
      </button>
      {open && (
        <pre
          style={{
            margin: '0.25rem 0 0.4rem 0',
            padding: '0.4rem 0.6rem',
            background: 'var(--bg-deep)',
            borderRadius: '0.35rem',
            fontSize: '0.7rem',
            fontFamily: 'var(--font-mono, monospace)',
            color: 'var(--color-ink-muted)',
            whiteSpace: 'pre-wrap',
            wordBreak: 'break-all',
            maxHeight: 260,
            overflow: 'auto',
          }}
        >
          {text}
        </pre>
      )}
    </div>
  )
}

/* ─── ToolCallsGroup: HF-style collapsible tool-call summary ── */

interface ToolCallEntry {
  id: string
  function: string
  arguments: unknown
  /** resolved tool/observation message, if any. */
  result?: ChatMessage
  /** latency_ms from trace tool_result event, if any. */
  latencyMs?: number | null
}

function ToolCallsGroup({ calls }: { calls: ToolCallEntry[] }) {
  const { t } = useLocale()
  const [summaryOpen, setSummaryOpen] = useState(true)

  if (calls.length === 0) return null

  const funcNames = Array.from(new Set(calls.map(c => c.function).filter(Boolean)))
  const summaryLabel =
    t('trace.toolCallsCount').replace('${n}', String(calls.length)) +
    (funcNames.length > 0 ? ` (${funcNames.join(', ')})` : '')

  return (
    <div style={{ marginTop: '0.6rem' }}>
      <button
        onClick={() => setSummaryOpen(v => !v)}
        style={{
          display: 'inline-flex',
          alignItems: 'center',
          gap: '0.35rem',
          background: 'none',
          border: 'none',
          cursor: 'pointer',
          padding: '0.2rem 0',
          fontSize: '0.72rem',
          fontFamily: 'var(--font-mono, monospace)',
          color: 'var(--text-muted)',
          opacity: 0.85,
        }}
      >
        {summaryOpen ? <ChevronDown size={12} /> : <ChevronRight size={12} />}
        <span style={{ fontWeight: 600 }}>{summaryLabel}</span>
      </button>
      {summaryOpen && (
        <div style={{ marginTop: '0.4rem', display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
          {calls.map((call, i) => (
            <ToolCallEntryRow key={call.id || i} entry={call} />
          ))}
        </div>
      )}
    </div>
  )
}

function ToolCallEntryRow({ entry }: { entry: ToolCallEntry }) {
  const { t } = useLocale()
  const [open, setOpen] = useState(false)
  const preview = argsPreview(entry.arguments)

  return (
    <div
      style={{
        borderLeft: '3px solid var(--bubble-tool-border)',
        paddingLeft: '0.7rem',
      }}
    >
      {/* Header row */}
      <button
        onClick={() => setOpen(v => !v)}
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: '0.45rem',
          width: '100%',
          background: 'none',
          border: 'none',
          cursor: 'pointer',
          padding: '0.2rem 0',
          textAlign: 'left',
        }}
      >
        <Wrench size={12} style={{ color: 'var(--bubble-tool-color)', flexShrink: 0 }} />
        <span
          style={{
            fontSize: '0.75rem',
            fontFamily: 'var(--font-mono, monospace)',
            fontWeight: 600,
            color: 'var(--bubble-tool-color)',
          }}
        >
          {entry.function}
        </span>
        {preview && (
          <span
            style={{
              fontSize: '0.7rem',
              fontFamily: 'var(--font-mono, monospace)',
              color: 'var(--text-muted)',
              opacity: 0.75,
              overflow: 'hidden',
              textOverflow: 'ellipsis',
              whiteSpace: 'nowrap',
              flex: 1,
            }}
          >
            {preview}
          </span>
        )}
        {entry.latencyMs != null && (
          <span
            style={{
              fontSize: '0.65rem',
              fontFamily: 'var(--font-mono, monospace)',
              color: 'var(--text-muted)',
              opacity: 0.6,
              whiteSpace: 'nowrap',
            }}
          >
            {fmtMs(entry.latencyMs)}
          </span>
        )}
        {entry.id && (
          <span
            style={{
              fontSize: '0.6rem',
              fontFamily: 'var(--font-mono, monospace)',
              color: 'var(--text-dim)',
              opacity: 0.5,
              whiteSpace: 'nowrap',
            }}
          >
            #{entry.id.slice(0, 8)}
          </span>
        )}
        {open ? <ChevronDown size={11} style={{ opacity: 0.5 }} /> : <ChevronRight size={11} style={{ opacity: 0.5 }} />}
      </button>

      {/* Expanded arguments */}
      {open && entry.arguments != null && (
        <div style={{ marginTop: '0.3rem' }}>
          <div
            style={{
              fontSize: '0.62rem',
              color: 'var(--text-muted)',
              opacity: 0.6,
              marginBottom: '0.2rem',
              letterSpacing: '0.04em',
              textTransform: 'uppercase',
              fontWeight: 600,
            }}
          >
            {t('trace.arguments')}
          </div>
          <pre
            style={{
              margin: 0,
              padding: '0.4rem 0.6rem',
              background: 'var(--bg-deep)',
              borderRadius: '0.35rem',
              fontSize: '0.7rem',
              fontFamily: 'var(--font-mono, monospace)',
              color: 'var(--color-ink-muted)',
              whiteSpace: 'pre-wrap',
              wordBreak: 'break-all',
              maxHeight: 200,
              overflow: 'auto',
            }}
          >
            {typeof entry.arguments === 'string'
              ? entry.arguments
              : JSON.stringify(entry.arguments, null, 2)}
          </pre>
        </div>
      )}

      {/* Result observation */}
      {entry.result && <ToolObservation msg={entry.result} />}
    </div>
  )
}

/* ─── Trace event extras (env_exec / loop error) ────────── */

function EnvExecRow({ event }: { event: AgentTraceEvent }) {
  const cmd = typeof event.payload.command === 'string' ? event.payload.command : ''
  if (!cmd) return null
  return (
    <div
      style={{
        display: 'flex',
        alignItems: 'flex-start',
        gap: '0.5rem',
        padding: '0.35rem 0.6rem',
        background: 'var(--bg-deep)',
        border: '1px solid var(--color-border-subtle)',
        borderRadius: '0.4rem',
        fontSize: '0.72rem',
        fontFamily: 'var(--font-mono, monospace)',
        color: 'var(--color-ink-muted)',
        marginTop: '0.4rem',
      }}
    >
      <Cpu size={12} style={{ color: 'var(--text-muted)', marginTop: 2, flexShrink: 0 }} />
      <span style={{ whiteSpace: 'pre-wrap', wordBreak: 'break-all', flex: 1 }}>$ {cmd}</span>
      {event.latency_ms != null && (
        <span style={{ opacity: 0.6, whiteSpace: 'nowrap' }}>{fmtMs(event.latency_ms)}</span>
      )}
    </div>
  )
}

function LoopErrorRow({ event }: { event: AgentTraceEvent }) {
  const msg = event.payload.message != null ? String(event.payload.message) : ''
  if (!msg) return null
  return (
    <div
      style={{
        display: 'flex',
        alignItems: 'flex-start',
        gap: '0.5rem',
        padding: '0.4rem 0.6rem',
        background: 'var(--danger-bg)',
        border: '1px solid var(--danger-border, var(--danger))',
        borderRadius: '0.4rem',
        fontSize: '0.72rem',
        fontFamily: 'var(--font-mono, monospace)',
        color: 'var(--danger)',
        marginTop: '0.4rem',
      }}
    >
      <AlertTriangle size={12} style={{ marginTop: 2, flexShrink: 0 }} />
      <span style={{ whiteSpace: 'pre-wrap', wordBreak: 'break-all', flex: 1 }}>{msg}</span>
    </div>
  )
}

/* ─── Structured multi-turn renderer (no trace) ────────── */

function StructuredMessages({
  messages,
  highlightId,
}: {
  messages: ChatMessage[]
  highlightId?: string
}) {
  // Build id->message for tool_call_id resolution
  const byToolCallId = new Map<string, ChatMessage>()
  for (const m of messages) {
    if (m.role === 'tool' && m.tool_call_id) byToolCallId.set(m.tool_call_id, m)
  }

  const rendered: React.ReactNode[] = []
  const consumedToolIds = new Set<string>()

  messages.forEach((msg, idx) => {
    if (msg.role === 'system') {
      rendered.push(
        <MessageRow
          key={idx}
          role="system"
          content={msg.content}
          msgId={msg.id}
          highlightId={highlightId}
        />
      )
      return
    }

    if (msg.role === 'user') {
      rendered.push(
        <MessageRow
          key={idx}
          role="user"
          content={msg.content}
          msgId={msg.id}
          highlightId={highlightId}
        />
      )
      return
    }

    if (msg.role === 'assistant') {
      const pm = msg.perf_metrics
      const headerPerf = pm ? (
        <HeaderPerfChip
          latency={pm.latency != null ? pm.latency * 1000 : null}
          inTok={pm.input_tokens}
          outTok={pm.output_tokens}
        />
      ) : undefined

      // Build tool call entries (resolve results via tool_call_id)
      const entries: ToolCallEntry[] = (msg.tool_calls ?? []).map(tc => {
        const result = byToolCallId.get(tc.id)
        if (result?.id) consumedToolIds.add(result.id)
        return {
          id: tc.id,
          function: tc.function,
          arguments: tc.arguments,
          result,
        }
      })

      rendered.push(
        <MessageRow
          key={idx}
          role="assistant"
          content={msg.content}
          msgId={msg.id}
          model={msg.model}
          highlightId={highlightId}
          headerExtra={headerPerf}
        >
          {entries.length > 0 && <ToolCallsGroup calls={entries} />}
        </MessageRow>
      )
      return
    }

    if (msg.role === 'tool') {
      // Skip tool messages already consumed by a ToolCallsGroup above
      if (msg.id && consumedToolIds.has(msg.id)) return
      rendered.push(
        <MessageRow
          key={idx}
          role="tool"
          content={msg.content}
          msgId={msg.id}
          highlightId={highlightId}
          toolError={msg.error ?? null}
          toolFunction={msg.function}
        />
      )
      return
    }
  })

  return <>{rendered}</>
}

/* ─── Trace-aware step grouping ───────────────────────── */

interface StepGroup {
  step: number
  /** Pre-agent messages (system/user) — only for step -1. */
  preAgentMessages: ChatMessage[]
  assistant: ChatMessage | null
  tools: ChatMessage[]
  traceEvents: AgentTraceEvent[]
  totalLatencyMs: number | null
}

function buildStepGroups(messages: ChatMessage[], trace: AgentTrace): StepGroup[] {
  const messageById = new Map<string, ChatMessage>()
  for (const m of messages) if (m.id) messageById.set(m.id, m)

  const stepEvents = new Map<number, AgentTraceEvent[]>()
  for (const ev of trace.events) {
    if (!stepEvents.has(ev.step)) stepEvents.set(ev.step, [])
    stepEvents.get(ev.step)!.push(ev)
  }

  const referencedIds = new Set<string>()
  for (const ev of trace.events) if (ev.message_id) referencedIds.add(ev.message_id)

  const preAgent: ChatMessage[] = []
  for (const m of messages) {
    if (m.id && referencedIds.has(m.id)) break
    preAgent.push(m)
  }

  const groups: StepGroup[] = []
  if (preAgent.length > 0) {
    groups.push({
      step: -1,
      preAgentMessages: preAgent,
      assistant: null,
      tools: [],
      traceEvents: [],
      totalLatencyMs: null,
    })
  }

  const sortedSteps = Array.from(stepEvents.keys()).sort((a, b) => a - b)
  for (const step of sortedSteps) {
    const events = stepEvents.get(step)!
    let assistant: ChatMessage | null = null
    const tools: ChatMessage[] = []
    const seenToolIds = new Set<string>()
    for (const ev of events) {
      if (!ev.message_id) continue
      const msg = messageById.get(ev.message_id)
      if (!msg) continue
      if (msg.role === 'assistant') {
        if (!assistant) assistant = msg
      } else if (msg.role === 'tool' || msg.role === 'user') {
        if (msg.id && !seenToolIds.has(msg.id)) {
          seenToolIds.add(msg.id)
          tools.push(msg)
        }
      }
    }
    let totalLatency: number | null = null
    for (const e of events) {
      if (e.latency_ms != null) totalLatency = (totalLatency ?? 0) + e.latency_ms
    }
    groups.push({
      step,
      preAgentMessages: [],
      assistant,
      tools,
      traceEvents: events,
      totalLatencyMs: totalLatency,
    })
  }

  return groups
}

/* ─── Timeline node (left sidebar, compact) ─────────── */

function TimelineNode({
  step,
  totalLatencyMs,
  isActive,
  isLast,
  onClick,
}: {
  step: number
  totalLatencyMs: number | null
  isActive: boolean
  isLast: boolean
  onClick: () => void
}) {
  const [hovered, setHovered] = useState(false)
  return (
    <div
      className="flex flex-col items-center"
      style={{ alignSelf: 'stretch', paddingTop: 8 }}
    >
      <button
        onClick={onClick}
        onMouseEnter={() => setHovered(true)}
        onMouseLeave={() => setHovered(false)}
        style={{
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          gap: 2,
          background: 'none',
          border: 'none',
          cursor: 'pointer',
          padding: 0,
          flexShrink: 0,
        }}
      >
        <div
          style={{
            width: isActive ? 24 : 18,
            height: isActive ? 24 : 18,
            borderRadius: '50%',
            background: isActive
              ? 'var(--accent)'
              : hovered
                ? 'var(--accent-dim)'
                : 'var(--bg-card2)',
            border: `2px solid ${isActive ? 'var(--accent)' : hovered ? 'var(--accent)' : 'var(--color-border-subtle)'}`,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            transition: 'all 0.2s',
            boxShadow: isActive ? '0 0 8px var(--accent-dim)' : 'none',
          }}
        >
          <span
            style={{
              fontSize: isActive ? 10 : 9,
              fontFamily: 'var(--font-mono, monospace)',
              fontWeight: 700,
              color: isActive ? '#fff' : 'var(--color-ink-muted)',
            }}
          >
            {step}
          </span>
        </div>
        {totalLatencyMs != null && (
          <span
            style={{
              fontSize: 9,
              fontFamily: 'var(--font-mono, monospace)',
              color: isActive ? 'var(--accent)' : 'var(--text-muted)',
              opacity: isActive ? 1 : 0.6,
              whiteSpace: 'nowrap',
            }}
          >
            {fmtMs(totalLatencyMs)}
          </span>
        )}
      </button>
      {/* Connector line */}
      {!isLast && (
        <div
          style={{
            width: 2,
            flex: 1,
            minHeight: 16,
            marginTop: 4,
            background: isActive ? 'var(--accent)' : 'var(--color-border-subtle)',
            opacity: isActive ? 0.5 : 0.2,
          }}
        />
      )}
    </div>
  )
}

/* ─── StepBlock: a single step rendered full-width ───── */

function StepBlock({
  group,
  highlightId,
  highlighted,
  onStepClick,
}: {
  group: StepGroup
  highlightId?: string
  highlighted: boolean
  onStepClick: (step: number) => void
}) {
  const { t } = useLocale()

  // Pre-agent messages (system/user) — render as-is, no wrapper
  if (group.step === -1) {
    return (
      <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
        {group.preAgentMessages.map((msg, idx) => (
          <MessageRow
            key={idx}
            role={msg.role as Role}
            content={msg.content}
            msgId={msg.id}
            highlightId={highlightId}
          />
        ))}
      </div>
    )
  }

  const modelGen = group.traceEvents.find(e => e.type === 'model_generate')
  const toolCallEvents = group.traceEvents.filter(e => e.type === 'tool_call')
  const toolResultEvents = group.traceEvents.filter(e => e.type === 'tool_result')
  const envExecs = group.traceEvents.filter(e => e.type === 'env_exec')
  const loopErrors = group.traceEvents.filter(e => e.type === 'error')

  // Build assistant header perf info from model_generate event (preferred)
  const mg = modelGen
  const stopReason = mg?.payload?.stop_reason ? String(mg.payload.stop_reason) : undefined
  const modelGenLatMs = mg?.latency_ms ?? null
  const inTok = mg?.token_usage?.input ?? null
  const outTok = mg?.token_usage?.output ?? null

  // Fallback to assistant.perf_metrics
  const ap = group.assistant?.perf_metrics
  const headerLat = modelGenLatMs ?? (ap?.latency != null ? ap.latency * 1000 : null)
  const headerIn = inTok ?? ap?.input_tokens ?? null
  const headerOut = outTok ?? ap?.output_tokens ?? null

  const headerPerf = (
    <HeaderPerfChip
      latency={headerLat}
      inTok={headerIn}
      outTok={headerOut}
      stopReason={stopReason}
    />
  )

  // Build ToolCallEntry[] — prefer assistant.tool_calls, fallback to tool_call events.
  const toolResultByCallId = new Map<string, AgentTraceEvent>()
  for (const ev of toolResultEvents) {
    const id = typeof ev.payload.id === 'string' ? ev.payload.id : null
    if (id) toolResultByCallId.set(id, ev)
  }
  const toolMsgByCallId = new Map<string, ChatMessage>()
  for (const m of group.tools) {
    if (m.tool_call_id) toolMsgByCallId.set(m.tool_call_id, m)
  }

  let entries: ToolCallEntry[] = []
  if (group.assistant?.tool_calls && group.assistant.tool_calls.length > 0) {
    entries = group.assistant.tool_calls.map((tc: ToolCall) => {
      const resultEv = toolResultByCallId.get(tc.id)
      return {
        id: tc.id,
        function: tc.function,
        arguments: tc.arguments,
        result: toolMsgByCallId.get(tc.id),
        latencyMs: resultEv?.latency_ms ?? null,
      }
    })
  } else if (toolCallEvents.length > 0) {
    // Fallback: reconstruct from tool_call events
    entries = toolCallEvents.map(ev => {
      const id = typeof ev.payload.id === 'string' ? ev.payload.id : ''
      const name = typeof ev.payload.name === 'string' ? ev.payload.name : ''
      const args = ev.payload.arguments
      const resultEv = toolResultByCallId.get(id)
      return {
        id,
        function: name,
        arguments: args,
        result: id ? toolMsgByCallId.get(id) : undefined,
        latencyMs: resultEv?.latency_ms ?? null,
      }
    })
  }

  // Residual tool messages not linked to any call (rare; textual_block mode)
  const linkedToolIds = new Set<string>()
  for (const e of entries) if (e.result?.id) linkedToolIds.add(e.result.id)
  const residualTools = group.tools.filter(m => !(m.id && linkedToolIds.has(m.id)))

  return (
    <div
      style={{
        borderRadius: '0.6rem',
        background: highlighted ? 'var(--accent-dim)' : 'transparent',
        borderLeft: highlighted ? '2px solid var(--accent)' : '2px solid transparent',
        padding: highlighted ? '0.2rem 0 0.2rem 0.4rem' : '0.2rem 0',
        transition: 'background 0.3s, border-color 0.3s',
      }}
    >
      {/* Step header strip */}
      <button
        onClick={() => onStepClick(group.step)}
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: '0.5rem',
          width: '100%',
          background: 'none',
          border: 'none',
          borderBottom: '1px dashed var(--color-border-subtle)',
          padding: '0.3rem 0 0.4rem 0',
          marginBottom: '0.5rem',
          cursor: 'pointer',
        }}
      >
        <span
          style={{
            fontSize: '0.7rem',
            fontWeight: 700,
            fontFamily: 'var(--font-mono, monospace)',
            color: highlighted ? 'var(--accent)' : 'var(--text-muted)',
            letterSpacing: '0.04em',
          }}
        >
          {t('trace.step')} {group.step}
        </span>
        {group.totalLatencyMs != null && (
          <span
            style={{
              fontSize: '0.65rem',
              fontFamily: 'var(--font-mono, monospace)',
              color: 'var(--text-muted)',
              opacity: 0.7,
            }}
          >
            {fmtMs(group.totalLatencyMs)}
          </span>
        )}
        {/* Event pills */}
        <div style={{ display: 'flex', gap: 4, flexWrap: 'wrap', flex: 1 }}>
          {group.traceEvents.map((ev, i) => (
            <TraceEventPill key={i} event={ev} />
          ))}
        </div>
      </button>

      <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
        {/* Assistant row */}
        {group.assistant && (
          <MessageRow
            role="assistant"
            content={group.assistant.content}
            msgId={group.assistant.id}
            model={group.assistant.model}
            highlightId={highlightId}
            headerExtra={headerPerf}
          >
            {entries.length > 0 && <ToolCallsGroup calls={entries} />}
          </MessageRow>
        )}

        {/* env_exec entries */}
        {envExecs.map((ev, i) => (
          <EnvExecRow key={`env-${i}`} event={ev} />
        ))}

        {/* Loop-level errors (not tied to a tool_call) */}
        {loopErrors.map((ev, i) => (
          <LoopErrorRow key={`err-${i}`} event={ev} />
        ))}

        {/* Any residual tool messages not linked via tool_call_id */}
        {residualTools.map((m, i) => (
          <MessageRow
            key={`residual-${m.id ?? i}`}
            role="tool"
            content={m.content}
            msgId={m.id}
            highlightId={highlightId}
            toolError={m.error ?? null}
            toolFunction={m.function}
          />
        ))}
      </div>
    </div>
  )
}

function TraceEventPill({ event }: { event: AgentTraceEvent }) {
  const { t } = useLocale()
  const cfg = (() => {
    switch (event.type) {
      case 'model_generate':
        return { Icon: Sparkles, color: 'var(--bubble-bot-color)', labelKey: 'trace.modelGenerate' }
      case 'tool_call':
        return { Icon: Wrench, color: 'var(--bubble-user-color)', labelKey: 'trace.toolCall' }
      case 'tool_result':
        return { Icon: Wrench, color: 'var(--bubble-tool-color)', labelKey: 'trace.toolResult' }
      case 'env_exec':
        return { Icon: Cpu, color: 'var(--text-muted)', labelKey: 'trace.envExec' }
      case 'error':
        return { Icon: AlertTriangle, color: 'var(--danger)', labelKey: 'trace.error' }
      case 'submit':
        return { Icon: Check, color: 'var(--success, #10b981)', labelKey: 'trace.submit' }
      default:
        return { Icon: Wrench, color: 'var(--text-muted)', labelKey: event.type }
    }
  })()
  const Icon = cfg.Icon
  const label = t(cfg.labelKey)
  return (
    <span
      style={{
        display: 'inline-flex',
        alignItems: 'center',
        gap: 3,
        padding: '1px 6px',
        borderRadius: 3,
        background: 'transparent',
        border: `1px solid ${cfg.color}`,
        color: cfg.color,
        fontSize: '0.58rem',
        fontFamily: 'var(--font-mono, monospace)',
        fontWeight: 500,
        opacity: 0.85,
        whiteSpace: 'nowrap',
      }}
    >
      <Icon size={9} />
      {label === cfg.labelKey ? event.type : label}
    </span>
  )
}

/* ─── Timeline layout orchestrator ─────────────────────── */

function TracedTimeline({
  groups,
  highlightStep,
  highlightId,
  onStepClick,
}: {
  groups: StepGroup[]
  highlightStep: number | null
  highlightId?: string
  onStepClick: (step: number) => void
}) {
  const preGroup = groups.find(g => g.step === -1)
  const agentGroups = groups.filter(g => g.step >= 0)

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '0.6rem' }}>
      {preGroup && (
        <StepBlock
          group={preGroup}
          highlightId={highlightId}
          highlighted={false}
          onStepClick={onStepClick}
        />
      )}
      {agentGroups.map((g, idx) => {
        const isLast = idx === agentGroups.length - 1
        const isActive = highlightStep === g.step
        return (
          <div key={g.step} style={{ display: 'flex', gap: '0.6rem', alignItems: 'stretch' }}>
            <TimelineNode
              step={g.step}
              totalLatencyMs={g.totalLatencyMs}
              isActive={isActive}
              isLast={isLast}
              onClick={() => onStepClick(g.step)}
            />
            <div style={{ flex: 1, minWidth: 0 }}>
              <StepBlock
                group={g}
                highlightId={highlightId}
                highlighted={isActive}
                onStepClick={onStepClick}
              />
            </div>
          </div>
        )
      })}
    </div>
  )
}

/* ─── Eval Result Panel ────────────────────────────────── */

interface EvalResultPanelProps {
  pred: string
  gold: string
  nScore: number
  score: Record<string, unknown>
  metadata: unknown
  threshold: number
  showPred: boolean
}

function CollapsibleJson({
  label,
  value,
  maxHeight = 200,
  defaultOpen = false,
  icon,
}: {
  label: string
  value: unknown
  maxHeight?: number
  defaultOpen?: boolean
  icon?: React.ReactNode
}) {
  const [open, setOpen] = useState(defaultOpen)
  const isEmpty =
    value == null ||
    (typeof value === 'object' && Object.keys(value as object).length === 0) ||
    value === '{}'
  if (isEmpty) return null
  return (
    <div style={{ marginTop: '0.4rem' }}>
      <button
        onClick={() => setOpen(v => !v)}
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
        }}
      >
        {open ? <ChevronDown size={11} /> : <ChevronRight size={11} />}
        {icon && <span style={{ display: 'inline-flex', alignItems: 'center' }}>{icon}</span>}
        <span className="uppercase tracking-wide font-semibold">{label}</span>
      </button>
      {open && (
        <div
          style={{
            marginTop: '0.3rem',
            borderRadius: '0.4rem',
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

function EvalResultPanel({
  pred,
  gold,
  nScore,
  score,
  metadata,
  threshold,
  showPred,
}: EvalResultPanelProps) {
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
        borderRadius: '0.75rem',
        border: '1px solid var(--border-md)',
        background: 'var(--bg-card2)',
        overflow: 'hidden',
        boxShadow: 'var(--shadow-sm)',
        animation: 'fadeInUp 300ms ease-out 160ms both',
      }}
    >
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

      <div
        style={{
          display: 'grid',
          gridTemplateColumns: showPred ? 'minmax(80px,auto) 1fr minmax(100px,auto)' : '1fr minmax(100px,auto)',
          gap: 0,
          padding: '0.75rem 1rem',
        }}
      >
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
              <span
                style={{
                  fontSize: '0.75rem',
                  color: 'var(--color-ink-muted)',
                  opacity: 0.5,
                  fontStyle: 'italic',
                }}
              >
                = Generated
              </span>
            ) : (
              <div style={{ fontSize: '0.875rem' }}>
                <MarkdownRenderer content={pred} />
              </div>
            )}
          </div>
        )}

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
              defaultOpen
              icon={<FileJson size={11} />}
            />
          )}
          {hasMetadata && (
            <CollapsibleJson
              label={t('prediction.metadata')}
              value={metadata}
              maxHeight={250}
              defaultOpen
              icon={<Database size={11} />}
            />
          )}
        </div>
      )}
    </div>
  )
}

/* ─── Main export ──────────────────────────────────────── */

export default function ChatView({ prediction, threshold = 0.99, highlightMsgId }: Props) {
  const showPred =
    prediction.Pred &&
    prediction.Pred !== '*Same as Generated*' &&
    prediction.Generated &&
    prediction.Pred.trim() !== prediction.Generated.trim()

  const messages = prediction.Messages
  const hasStructured = !!(messages && messages.length > 0)

  const agentTrace = prediction.AgentTrace
  const hasTrace = !!(agentTrace && agentTrace.events && agentTrace.events.length > 0)
  const [highlightedStep, setHighlightedStep] = useState<number | null>(null)

  const stepGroups = useMemo(() => {
    if (!hasTrace || !hasStructured || !messages || !agentTrace) return null
    return buildStepGroups(messages, agentTrace)
  }, [hasTrace, hasStructured, messages, agentTrace])

  const handleStepClick = useCallback((step: number) => {
    setHighlightedStep(prev => (prev === step ? null : step))
  }, [])

  return (
    <div className="flex flex-col gap-4 py-2">
      {hasTrace && stepGroups ? (
        <TracedTimeline
          groups={stepGroups}
          highlightStep={highlightedStep}
          highlightId={highlightMsgId}
          onStepClick={handleStepClick}
        />
      ) : hasStructured ? (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
          <StructuredMessages messages={messages!} highlightId={highlightMsgId} />
        </div>
      ) : (
        (() => {
          const isSystemMsg = hasSystemPrompt(prediction.Input)
          const { system, user } = isSystemMsg
            ? parseSystemUser(prediction.Input)
            : { system: '', user: prediction.Input }
          const headerPerf = prediction.PerfMetrics ? (
            <HeaderPerfChip
              latency={prediction.PerfMetrics.latency != null ? prediction.PerfMetrics.latency * 1000 : null}
              inTok={prediction.PerfMetrics.input_tokens}
              outTok={prediction.PerfMetrics.output_tokens}
            />
          ) : undefined
          return (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
              {system && <MessageRow role="system" content={system} />}
              <MessageRow role="user" content={user || prediction.Input} />
              {prediction.Generated && (
                <MessageRow
                  role="assistant"
                  content={prediction.Generated}
                  headerExtra={headerPerf}
                />
              )}
            </div>
          )
        })()
      )}

      <div style={{ borderTop: '1px solid var(--color-border-subtle)' }} />

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
