import React, { useState, useCallback, useEffect, useRef } from 'react'
import {
  Copy,
  Check,
  ChevronDown,
  ChevronRight,
  Shield,
  Clock,
  Zap,
  Activity,
  ArrowDownToLine,
  ArrowUpFromLine,
} from 'lucide-react'
import type { ContentBlock } from '@/api/types'
import { useLocale } from '@/contexts/LocaleContext'
import MarkdownRenderer from '@/components/common/MarkdownRenderer'
import { fmtMs } from '@/utils/formatUtils'
import { contentToText } from './chatHelpers'
import { type Role, rolePalette } from './roleConfig'
import { renderContentBlocks } from './MediaBlocks'

/* ─── CopyIconButton ───────────────────────────────────────── */

export function CopyIconButton({ text }: { text: string }) {
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
      title={t('prediction.copyContent')}
      style={{
        display: 'inline-flex',
        alignItems: 'center',
        justifyContent: 'center',
        width: 24,
        height: 24,
        borderRadius: 4,
        border: '1px solid var(--color-border-subtle)',
        background: 'transparent',
        color: copied ? 'var(--accent)' : 'var(--text-muted)',
        cursor: 'pointer',
        opacity: 0.6,
        transition: 'opacity 0.15s, color 0.15s',
        flexShrink: 0,
      }}
      onMouseEnter={e => (e.currentTarget.style.opacity = '1')}
      onMouseLeave={e => (e.currentTarget.style.opacity = '0.6')}
    >
      {copied ? <Check size={12} /> : <Copy size={12} />}
    </button>
  )
}

/* ─── MsgIdChip ────────────────────────────────────────────── */

export function MsgIdChip({ msgId }: { msgId: string }) {
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

/* ─── HeaderPerfChip ───────────────────────────────────────── */

/** Compact perf chip rendered inline inside a message header. */
export function HeaderPerfChip({
  latency,
  ttft,
  tpot,
  inTok,
  outTok,
  stopReason,
}: {
  latency?: number | null
  ttft?: number | null
  tpot?: number | null
  inTok?: number | null
  outTok?: number | null
  stopReason?: string
}) {
  const items: React.ReactNode[] = []
  const chipStyle: React.CSSProperties = {
    display: 'inline-flex',
    alignItems: 'center',
    gap: '2px',
  }
  const iconSize = 10

  if (latency != null) {
    items.push(
      <span key="lat" style={chipStyle}>
        <Clock size={iconSize} style={{ opacity: 0.7 }} />
        {fmtMs(latency)}
      </span>
    )
  }
  if (ttft != null) {
    items.push(
      <span key="ttft" style={chipStyle}>
        <Zap size={iconSize} style={{ opacity: 0.7 }} />
        TTFT {fmtMs(ttft * 1000)}
      </span>
    )
  }
  if (tpot != null) {
    items.push(
      <span key="tpot" style={chipStyle}>
        <Activity size={iconSize} style={{ opacity: 0.7 }} />
        TPOT {fmtMs(tpot * 1000)}
      </span>
    )
  }
  if (inTok != null) {
    items.push(
      <span key="in" style={chipStyle}>
        <ArrowDownToLine size={iconSize} style={{ opacity: 0.7 }} />
        in {inTok}
      </span>
    )
  }
  if (outTok != null) {
    items.push(
      <span key="out" style={chipStyle}>
        <ArrowUpFromLine size={iconSize} style={{ opacity: 0.7 }} />
        out {outTok}
      </span>
    )
  }
  if (stopReason) {
    items.push(
      <span key="stop" style={chipStyle}>
        stop:{stopReason}
      </span>
    )
  }

  if (items.length === 0) return null
  return (
    <span
      style={{
        display: 'inline-flex',
        alignItems: 'center',
        gap: '0.5rem',
        fontSize: '0.65rem',
        fontFamily: 'var(--font-mono, monospace)',
        color: 'var(--text-muted)',
        opacity: 0.85,
        whiteSpace: 'nowrap',
        flexWrap: 'wrap',
      }}
    >
      {items}
    </span>
  )
}

/* ─── MessageRow ───────────────────────────────────────────── */

export interface MessageRowProps {
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

export function MessageRow({
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

/* ─── SystemPromptRow ──────────────────────────────────────── */

export function SystemPromptRow({
  content,
  msgId,
}: {
  content: string | ContentBlock[]
  msgId?: string
  highlightId?: string
}) {
  const { t } = useLocale()
  const [open, setOpen] = useState(false)
  const text = contentToText(content)
  const preview = text.replace(/\s+/g, ' ').trim()
  const previewShort = preview.length > 120 ? preview.slice(0, 120) + '…' : preview

  return (
    <div
      style={{
        borderLeft: '3px solid var(--text-muted)',
        borderRadius: '0.5rem',
        background: 'var(--bubble-system-bg)',
        overflow: 'hidden',
      }}
    >
      <button
        onClick={() => setOpen(v => !v)}
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: '0.45rem',
          width: '100%',
          padding: '0.45rem 0.75rem',
          background: 'none',
          border: 'none',
          cursor: 'pointer',
          textAlign: 'left',
        }}
      >
        {open ? <ChevronDown size={12} style={{ color: 'var(--text-muted)', flexShrink: 0 }} /> : <ChevronRight size={12} style={{ color: 'var(--text-muted)', flexShrink: 0 }} />}
        <Shield size={12} style={{ color: 'var(--text-muted)', flexShrink: 0 }} />
        <span style={{ fontSize: '0.72rem', fontWeight: 600, color: 'var(--text-muted)' }}>
          {t('prediction.systemPrompt')}
        </span>
        {!open && (
          <span
            style={{
              fontSize: '0.7rem',
              color: 'var(--text-muted)',
              opacity: 0.6,
              overflow: 'hidden',
              textOverflow: 'ellipsis',
              whiteSpace: 'nowrap',
              flex: 1,
              fontFamily: 'var(--font-mono, monospace)',
            }}
          >
            {previewShort}
          </span>
        )}
        {msgId && <MsgIdChip msgId={msgId} />}
      </button>
      {open && (
        <div style={{ padding: '0 0.75rem 0.6rem 1.6rem' }}>
          <div style={{ fontSize: '0.82rem', lineHeight: 1.55 }}>
            {Array.isArray(content)
              ? renderContentBlocks(content, { includeReasoning: false })
              : <MarkdownRenderer content={content} />}
          </div>
        </div>
      )}
    </div>
  )
}
