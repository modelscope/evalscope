import React, { useEffect, useRef } from 'react'
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

const CHIP_CLASS = 'px-[7px] py-[1px] rounded-[4px] bg-[var(--bg-deep)] border border-[var(--border)] text-[0.65rem] font-mono text-[var(--text-muted)]'

export function CopyIconButton({ text }: { text: string }) {
  const { t } = useLocale()
  const { copied, copy } = useCopy()
  return (
    <button
      onClick={(e) => { e.stopPropagation(); copy(text) }}
      title={t('prediction.copyContent')}
      className={[
        'inline-flex items-center justify-center w-6 h-6 rounded-[4px] border border-[var(--border)] bg-transparent cursor-pointer shrink-0',
        'opacity-60 hover:opacity-100 transition-[opacity,color] duration-150',
        copied ? 'text-[var(--accent)]' : 'text-[var(--text-muted)]',
      ].join(' ')}
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
      className={[
        'inline-flex items-center gap-[3px] px-[6px] py-[1px] rounded-[4px] border border-[var(--border)] bg-[var(--bg-deep)]',
        'text-[0.62rem] font-mono cursor-pointer opacity-70 hover:opacity-100 transition-[opacity,color] duration-150',
        copied ? 'text-[var(--accent)]' : 'text-[var(--text-muted)]',
      ].join(' ')}
    >
      {copied ? <Check size={10} /> : <Copy size={10} />}
      <span>{msgId}</span>
    </button>
  )
}

/* ─── HeaderPerfChip ───────────────────────────────────────── */

const PERF_CHIP_CLASS = 'inline-flex items-center gap-[2px]'

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
    items.push(<span key="lat" className={PERF_CHIP_CLASS}><Clock size={iconSize} className="opacity-70" />{fmtMs(latency)}</span>)
  }
  if (ttft != null) {
    items.push(<span key="ttft" className={PERF_CHIP_CLASS}><Zap size={iconSize} className="opacity-70" />TTFT {fmtMs(ttft * 1000)}</span>)
  }
  if (tpot != null) {
    items.push(<span key="tpot" className={PERF_CHIP_CLASS}><Activity size={iconSize} className="opacity-70" />TPOT {fmtMs(tpot * 1000)}</span>)
  }
  if (inTok != null) {
    items.push(<span key="in" className={PERF_CHIP_CLASS}><ArrowDownToLine size={iconSize} className="opacity-70" />in {inTok}</span>)
  }
  if (outTok != null) {
    items.push(<span key="out" className={PERF_CHIP_CLASS}><ArrowUpFromLine size={iconSize} className="opacity-70" />out {outTok}</span>)
  }
  if (stopReason) {
    items.push(<span key="stop" className={PERF_CHIP_CLASS}>stop:{stopReason}</span>)
  }

  if (items.length === 0) return null
  return (
    <span className="inline-flex items-center gap-2 text-[0.65rem] font-mono text-[var(--text-muted)] opacity-85 whitespace-nowrap flex-wrap">
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
      <div ref={ref} className="flex-1 min-w-0">
        {/* Header row */}
        {!compact && (
          <div className="flex items-center gap-2 mb-[0.4rem] flex-wrap">
            <RoleIcon size={13} className="shrink-0" style={{ color: palette.labelColor }} />
            <span className="text-xs font-semibold" style={{ color: palette.labelColor }}>
              {labelOverride ?? palette.label}
            </span>
            {toolFunction && <span className={CHIP_CLASS}>{toolFunction}</span>}
            {model && <span className={CHIP_CLASS}>{model}</span>}
            {headerExtra}
            <span className="flex-1" />
            {msgId && <MsgIdChip msgId={msgId} />}
            {copyText && <CopyIconButton text={copyText} />}
          </div>
        )}

        {/* Tool error banner */}
        {toolError && (
          <div className="mb-[0.4rem] px-[0.6rem] py-[0.4rem] rounded-[0.4rem] bg-[var(--danger-bg)] border border-[var(--danger-border)] text-[var(--danger)] text-[0.72rem] font-mono">
            {toolError.type ? `[${toolError.type}] ` : ''}
            {toolError.message}
          </div>
        )}

        {/* Content */}
        <div className="text-[0.85rem] leading-[1.55]">
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
            <Shield size={12} className="text-[var(--text-muted)] shrink-0" />
            <span className="text-[0.72rem] font-semibold text-[var(--text-muted)]">
              {t('prediction.systemPrompt')}
            </span>
            {!open && (
              <span className="font-mono text-[0.7rem] text-[var(--text-muted)] opacity-60 overflow-hidden text-ellipsis whitespace-nowrap flex-1">
                {previewShort}
              </span>
            )}
            {msgId && <MsgIdChip msgId={msgId} />}
          </>
        )}
        headerStyle={{ gap: '0.45rem', padding: '0.45rem 0.75rem' }}
        bodyStyle={{ padding: '0 0.75rem 0.6rem 1.6rem' }}
      >
        <div className="text-[0.82rem] leading-[1.55]">
          {Array.isArray(content)
            ? renderContentBlocks(content, { includeReasoning: false })
            : <MarkdownRenderer content={content} />}
        </div>
      </Collapsible>
    </ChatBubble>
  )
}
