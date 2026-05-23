import { Wrench } from 'lucide-react'
import type { ChatMessage } from '@/api/types'
import { useLocale } from '@/contexts/LocaleContext'
import Collapsible from '@/components/ui/Collapsible'
import { bubbleAccent, bubbleBorder } from '@/components/ui/ChatBubble'
import { fmtMs } from '@/utils/formatUtils'
import { contentToText, argsPreview } from './chatHelpers'

/** Strip vendor prefixes (e.g. `toolu_` from Anthropic) and keep the unique tail. */
function shortToolId(id: string): string {
  const trimmed = id.replace(/^(toolu_|call_|tool_)/, '')
  return trimmed.slice(0, 8)
}

/* ─── ToolObservation ──────────────────────────────────────── */

export function ToolObservation({ msg }: { msg: ChatMessage }) {
  const { t } = useLocale()
  const text = contentToText(msg.content)
  const preview = text.replace(/\s+/g, ' ').trim()
  const previewShort = preview.length > 140 ? preview.slice(0, 140) + '…' : preview
  const hasError = !!msg.error
  const headerColor = hasError ? 'var(--danger)' : 'var(--text-muted)'

  return (
    <Collapsible
      style={{ borderLeft: `2px solid ${bubbleBorder('tool')}`, paddingLeft: '0.6rem', marginTop: '0.25rem' }}
      headerStyle={{ color: headerColor, fontSize: '0.72rem' }}
      chevronSize={11}
      chevronColor={headerColor}
      header={
        <>
          <span
            className="inline-block w-[6px] h-[6px] rounded-full shrink-0"
            style={{ background: hasError ? 'var(--danger)' : bubbleAccent('bot') }}
          />
          <span className="font-mono text-[0.7rem] overflow-hidden text-ellipsis whitespace-nowrap flex-1">
            {hasError
              ? `${t('trace.error')}: ${msg.error?.message ?? ''}`
              : previewShort || t('trace.stdout')}
          </span>
          {msg.id && (
            <span className="opacity-40 text-[0.6rem] font-mono">
              {msg.id}
            </span>
          )}
        </>
      }
    >
      <pre className="mt-1 mb-[0.4rem] mx-0 px-[0.6rem] py-[0.4rem] bg-[var(--bg-deep)] rounded-[0.35rem] text-[0.7rem] font-mono text-[var(--text-muted)] whitespace-pre-wrap break-all max-h-[260px] overflow-auto">
        {text}
      </pre>
    </Collapsible>
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

  if (calls.length === 0) return null

  const funcNames = Array.from(new Set(calls.map(c => c.function).filter(Boolean)))
  const summaryLabel =
    t('trace.toolCallsCount').replace('${n}', String(calls.length)) +
    (funcNames.length > 0 ? ` (${funcNames.join(', ')})` : '')

  return (
    <Collapsible
      defaultOpen
      style={{ marginTop: '0.6rem' }}
      headerStyle={{
        display: 'inline-flex',
        width: 'auto',
        gap: '0.35rem',
        fontSize: '0.72rem',
        fontFamily: 'var(--font-mono, monospace)',
        color: 'var(--text-muted)',
        opacity: 0.85,
      }}
      bodyStyle={{ marginTop: '0.4rem', display: 'flex', flexDirection: 'column', gap: '0.5rem' }}
      header={<span className="font-semibold">{summaryLabel}</span>}
    >
      {calls.map((call, i) => (
        <ToolCallEntryRow key={call.id || i} entry={call} />
      ))}
    </Collapsible>
  )
}

export function ToolCallEntryRow({ entry }: { entry: ToolCallEntry }) {
  const { t } = useLocale()
  const preview = argsPreview(entry.arguments)
  const toolAccent = bubbleAccent('tool')

  return (
    <div className="pl-[0.7rem] border-l-[3px]" style={{ borderLeftColor: bubbleBorder('tool') }}>
      <Collapsible
        headerStyle={{ gap: '0.45rem' }}
        header={
          <>
            <Wrench size={12} className="shrink-0" style={{ color: toolAccent }} />
            <span className="text-xs font-mono font-semibold" style={{ color: toolAccent }}>
              {entry.function}
            </span>
            {preview && (
              <span className="text-[0.7rem] font-mono text-[var(--text-muted)] opacity-75 overflow-hidden text-ellipsis whitespace-nowrap flex-1">
                {preview}
              </span>
            )}
            {entry.latencyMs != null && (
              <span className="text-[0.65rem] font-mono text-[var(--text-muted)] opacity-60 whitespace-nowrap">
                {fmtMs(entry.latencyMs)}
              </span>
            )}
            {entry.id && (
              <span
                title={entry.id}
                className="text-[0.6rem] font-mono text-[var(--text-dim)] opacity-50 whitespace-nowrap"
              >
                #{shortToolId(entry.id)}
              </span>
            )}
          </>
        }
      >
        {entry.arguments != null && (
          <div className="mt-[0.3rem]">
            <div className="text-[0.62rem] text-[var(--text-muted)] opacity-60 mb-[0.2rem] tracking-[0.04em] uppercase font-semibold">
              {t('trace.arguments')}
            </div>
            <pre className="m-0 px-[0.6rem] py-[0.4rem] bg-[var(--bg-deep)] rounded-[0.35rem] text-[0.7rem] font-mono text-[var(--text-muted)] whitespace-pre-wrap break-all max-h-[200px] overflow-auto">
              {typeof entry.arguments === 'string'
                ? entry.arguments
                : JSON.stringify(entry.arguments, null, 2)}
            </pre>
          </div>
        )}
      </Collapsible>

      {entry.result && <ToolObservation msg={entry.result} />}
    </div>
  )
}
