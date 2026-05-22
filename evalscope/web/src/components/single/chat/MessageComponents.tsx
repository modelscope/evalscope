import React, { useEffect, useRef, type CSSProperties } from 'react'
import {
  Copy,
  Check,
  Shield,
  Clock,
  Zap,
  Activity,
  ArrowDownToLine,
  ArrowUpFromLine,
} from 'lucide-react'
import type { ContentBlock } from '@/api/types'
import { useLocale } from '@/contexts/LocaleContext'
import { useCopy } from '@/hooks/useCopy'
import MarkdownRenderer from '@/components/common/MarkdownRenderer'
import Collapsible from '@/components/ui/Collapsible'
import ChatBubble from '@/components/ui/ChatBubble'
import { fmtMs } from '@/utils/formatUtils'
import { contentToText } from './chatHelpers'
import { type Role, rolePalette, roleToBubble } from './roleConfig'
import { renderContentBlocks } from './MediaBlocks'

/* ─── CopyIconButton ───────────────────────────────────────── */

const CHIP_BASE_STYLE: CSSProperties = {
  padding: '1px 7px',
  borderRadius: 4,
  background: 'var(--bg-deep)',
  border: '1px solid var(--border)',
  fontSize: '0.65rem',
  fontFamily: 'var(--font-mono, monospace)',
  color: 'var(--text-muted)',
}

export function CopyIconButton({ text }: { text: string }) {
  const { t } = useLocale()
  const { copied, copy } = useCopy()
  return (
    <button
      onClick={(e) => { e.stopPropagation(); copy(text) }}
      title={t('prediction.copyContent')}
      style={{
        display: 'inline-flex',
        alignItems: 'center',
        justifyContent: 'center',
        width: 24,
        height: 24,
        borderRadius: 4,
        border: '1px solid var(--border)',
        background: 'transparent',
        color: copied ? 'var(--accent)' : 'var(--text-muted)',
        cursor: 'pointer',
        opacity: 0.6,
        transition: 'opacity 0.15s, color 0.15s',
        flexShrink: 0,
      }}
      onMouseEnter={(e) => (e.currentTarget.style.opacity = '1')}
      onMouseLeave={(e) => (e.currentTarget.style.opacity = '0.6')}
    >
      {copied ? <Check size={12} /> : <Copy size={12} />}
    </button>
  )
}

/* ─── MsgIdChip ────────────────────────────────────────────── */

export function MsgIdChip({ msgId }: { msgId: string }) {
  const { t } = useLocale()
  const { copied, copy } = useCopy()
  return (
    <button
      onClick={(e) => { e.stopPropagation(); copy(msgId) }}
      title={t('prediction.copyMsgId')}
      style={{
        display: 'inline-flex',
        alignItems: 'center',
        gap: 3,
        padding: '1px 6px',
        borderRadius: 4,
        border: '1px solid var(--border)',
        background: 'var(--bg-deep)',
        color: copied ? 'var(--accent)' : 'var(--text-muted)',
        fontSize: '0.62rem',
        fontFamily: 'var(--font-mono, monospace)',
        cursor: 'pointer',
        opacity: 0.7,
        transition: 'opacity 0.15s, color 0.15s',
      }}
      onMouseEnter={(e) => (e.currentTarget.style.opacity = '1')}
      onMouseLeave={(e) => (e.currentTarget.style.opacity = '0.7')}
    >
      {copied ? <Check size={10} /> : <Copy size={10} />}
      <span>{msgId}</span>
    </button>
  )
}

/* ─── HeaderPerfChip ───────────────────────────────────────── */

const PERF_CHIP_STYLE: CSSProperties = { display: 'inline-flex', alignItems: 'center', gap: '2px' }

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
  const iconSize = 10

  if (latency != null) {
    items.push(<span key="lat" style={PERF_CHIP_STYLE}><Clock size={iconSize} style={{ opacity: 0.7 }} />{fmtMs(latency)}</span>)
  }
  if (ttft != null) {
    items.push(<span key="ttft" style={PERF_CHIP_STYLE}><Zap size={iconSize} style={{ opacity: 0.7 }} />TTFT {fmtMs(ttft * 1000)}</span>)
  }
  if (tpot != null) {
    items.push(<span key="tpot" style={PERF_CHIP_STYLE}><Activity size={iconSize} style={{ opacity: 0.7 }} />TPOT {fmtMs(tpot * 1000)}</span>)
  }
  if (inTok != null) {
    items.push(<span key="in" style={PERF_CHIP_STYLE}><ArrowDownToLine size={iconSize} style={{ opacity: 0.7 }} />in {inTok}</span>)
  }
  if (outTok != null) {
    items.push(<span key="out" style={PERF_CHIP_STYLE}><ArrowUpFromLine size={iconSize} style={{ opacity: 0.7 }} />out {outTok}</span>)
  }
  if (stopReason) {
    items.push(<span key="stop" style={PERF_CHIP_STYLE}>stop:{stopReason}</span>)
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

  return (
    <ChatBubble
      role={roleToBubble(role)}
      highlighted={isHighlighted}
      className="flex w-full px-3.5 py-2.5"
      style={{ animation: 'fadeInUp 240ms ease-out both' }}
    >
      <div ref={ref} style={{ flex: 1, minWidth: 0 }}>
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
            <span style={{ fontSize: '0.75rem', fontWeight: 600, color: palette.labelColor }}>
              {labelOverride ?? palette.label}
            </span>
            {toolFunction && <span style={CHIP_BASE_STYLE}>{toolFunction}</span>}
            {model && <span style={CHIP_BASE_STYLE}>{model}</span>}
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
    </ChatBubble>
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
  const text = contentToText(content)
  const preview = text.replace(/\s+/g, ' ').trim()
  const previewShort = preview.length > 120 ? preview.slice(0, 120) + '…' : preview

  return (
    <ChatBubble role="system" className="overflow-hidden">
      <Collapsible
        header={(open) => (
          <>
            <Shield size={12} style={{ color: 'var(--text-muted)', flexShrink: 0 }} />
            <span style={{ fontSize: '0.72rem', fontWeight: 600, color: 'var(--text-muted)' }}>
              {t('prediction.systemPrompt')}
            </span>
            {!open && (
              <span
                className="font-mono"
                style={{
                  fontSize: '0.7rem',
                  color: 'var(--text-muted)',
                  opacity: 0.6,
                  overflow: 'hidden',
                  textOverflow: 'ellipsis',
                  whiteSpace: 'nowrap',
                  flex: 1,
                }}
              >
                {previewShort}
              </span>
            )}
            {msgId && <MsgIdChip msgId={msgId} />}
          </>
        )}
        headerStyle={{ gap: '0.45rem', padding: '0.45rem 0.75rem' }}
        bodyStyle={{ padding: '0 0.75rem 0.6rem 1.6rem' }}
      >
        <div style={{ fontSize: '0.82rem', lineHeight: 1.55 }}>
          {Array.isArray(content)
            ? renderContentBlocks(content, { includeReasoning: false })
            : <MarkdownRenderer content={content} />}
        </div>
      </Collapsible>
    </ChatBubble>
  )
}
