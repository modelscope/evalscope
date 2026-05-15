import { useState } from 'react'
import { ChevronDown, ChevronRight, Wrench } from 'lucide-react'
import type { ChatMessage } from '@/api/types'
import { useLocale } from '@/contexts/LocaleContext'
import { fmtMs } from '@/utils/formatUtils'
import { contentToText, argsPreview } from './chatHelpers'

/* ─── ToolObservation ──────────────────────────────────────── */

export function ToolObservation({ msg }: { msg: ChatMessage }) {
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

/* ─── ToolCallEntry / ToolCallsGroup / ToolCallEntryRow ─────── */

export interface ToolCallEntry {
  id: string
  function: string
  arguments: unknown
  /** resolved tool/observation message, if any. */
  result?: ChatMessage
  /** latency_ms from trace tool_result event, if any. */
  latencyMs?: number | null
}

export function ToolCallsGroup({ calls }: { calls: ToolCallEntry[] }) {
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

export function ToolCallEntryRow({ entry }: { entry: ToolCallEntry }) {
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
