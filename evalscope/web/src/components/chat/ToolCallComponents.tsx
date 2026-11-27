import { scrollToContent } from './scrollToContent'
import { useContext, useEffect, useRef } from 'react'
import { EvidenceByMessage } from '@/contexts/EvidenceContext'
import { AlertTriangle, Wrench } from 'lucide-react'
import { useLocale } from '@/contexts/LocaleContext'
import Collapsible from '@/components/ui/Collapsible'
import { fmtMs } from '@/utils/formatUtils'
import { contentToText, argsPreview } from '@/domain/chat/messageText'
import type { ToolCallEntry } from '@/domain/trace/stepGroups'

/** Strip vendor prefixes (e.g. `toolu_` from Anthropic) and keep the unique tail. */
function shortToolId(id: string): string {
  const trimmed = id.replace(/^(toolu_|call_|tool_)/, '')
  return trimmed.slice(0, 8)
}

/** Render exact quotations as text nodes; evidence never introduces HTML. */
function EvidenceText({ text, quotes }: { text: string; quotes: string[] }) {
  const quote = quotes.filter((q) => q && text.includes(q)).sort((a, b) => text.indexOf(a) - text.indexOf(b))[0]
  if (!quote) return <>{text}</>
  const at = text.indexOf(quote)
  return (
    <>
      {text.slice(0, at)}
      <mark className="bg-[var(--warning-bg)] text-[var(--text)]">{quote}</mark>
      <EvidenceText text={text.slice(at + quote.length)} quotes={quotes} />
    </>
  )
}

/* ─── ToolCallEntry / ToolCallsGroup / ToolCallEntryRow ─────── */

/** Re-exported so call sites keep importing the entry type next to its renderer. */
export type { ToolCallEntry } from '@/domain/trace/stepGroups'

export function ToolCallsGroup({ calls, failedCallIds = [], highlightId }: { calls: ToolCallEntry[]; failedCallIds?: string[]; highlightId?: string }) {
  if (calls.length === 0) return null

  return (
    <div className="mt-2 space-y-2">
      {calls.map((call, i) => (
        <ToolCallEntryRow key={call.id || i} entry={call} failed={failedCallIds.includes(call.id)} highlightId={highlightId} />
      ))}
    </div>
  )
}

export function ToolCallEntryRow({ entry, failed = false, highlightId }: { entry: ToolCallEntry; failed?: boolean; highlightId?: string }) {
  const { t } = useLocale()
  const preview = argsPreview(entry.arguments)
  const toolAccent = 'var(--text)'
  const evidence = useContext(EvidenceByMessage)
  const hasError = failed || Boolean(entry.result?.error)
  const target = Boolean(highlightId && entry.result?.id?.startsWith(highlightId))
  const ref = useRef<HTMLDivElement>(null)
  useEffect(() => { if (target) scrollToContent(ref.current) }, [target])
  const quotes = entry.result?.id ? (evidence[entry.result.id] ?? []) : []

  return (
    <div ref={ref} className="rounded-md border border-[var(--border)] px-3 py-1">
      <Collapsible
        key={target ? 'message' : quotes.length ? 'evidence' : hasError ? 'error' : 'call'}
        defaultOpen={target || hasError || quotes.length > 0}
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
            {hasError && <span className="text-xs text-[var(--danger)]">{t('diagnosis.finding.tool_failure')}</span>}
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
              {typeof entry.arguments === 'string' ? entry.arguments : JSON.stringify(entry.arguments, null, 2)}
            </pre>
          </div>
        )}
        {entry.result && (
          <div className="mt-3 mb-2">
            <div className="mb-1 text-xs text-[var(--text-muted)]">{t('trace.toolResult')}</div>
            <div
              className={
                hasError ? 'rounded bg-[var(--danger-bg)] p-3 text-[var(--danger)]' : 'rounded bg-[var(--bg-deep)] p-3'
              }
            >
              {hasError && (
                <div className="mb-1 flex items-center gap-1 text-xs">
                  <AlertTriangle size={13} />
                  {t('diagnosis.finding.tool_failure')}
                </div>
              )}
              <pre className="m-0 whitespace-pre-wrap break-words text-sm leading-relaxed">
                <EvidenceText
                  text={contentToText(entry.result.content) || entry.result.error?.message || ''}
                  quotes={quotes}
                />
              </pre>
            </div>
          </div>
        )}
      </Collapsible>
    </div>
  )
}
