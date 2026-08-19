import { useState } from 'react'
import { AlertTriangle, Check, ChevronDown, ChevronRight, RotateCcw, Scale, Timer } from 'lucide-react'
import { useLocale } from '@/contexts/LocaleContext'
import MarkdownRenderer from '@/components/ui/MarkdownRenderer'
import JsonViewer from '@/components/ui/JsonViewer'
import { isUsableStatus, type JudgeCaseView, type JudgeReviewView } from '@/domain/chat/judgeReview'

/** Cases beyond this are collapsed: a rubric benchmark can declare dozens. */
const VISIBLE_CASES = 8

const STATUS_LABEL_KEYS: Record<string, string> = {
  transport_error: 'prediction.statusTransportError',
  parse_error: 'prediction.statusParseError',
  invalid_session: 'prediction.statusInvalidSession',
  excluded: 'prediction.statusExcluded',
}

/**
 * Fields that carry the actual decision, in order of preference.
 *
 * Read as "pick the first one the verdict happens to have": most schemas expose exactly one
 * (`verdict` / `overall_score` / `rating` / `correct`), a rubric schema also has `score` alongside
 * `verdict` -- the row shows the label, the expanded body shows the rest.
 */
const DECISIVE_KEYS = ['verdict', 'overall_score', 'correct', 'awarded', 'rating', 'score'] as const

/** Keys whose value is self-explanatory, so the row shows the value alone without the key. */
const SELF_EVIDENT_KEYS = new Set(['verdict', 'correct'])

/** Render a verdict compactly: the row shows the decision, the expansion shows the evidence. */
function verdictSummary(verdict: unknown): string {
  if (verdict == null) return ''
  if (typeof verdict !== 'object') return String(verdict)
  const entries = Object.entries(verdict as Record<string, unknown>)
  const decisive = DECISIVE_KEYS.map(k => entries.find(([key]) => key === k)).filter(
    (e): e is [string, unknown] => !!e,
  )
  // A schema that carries none of the well-known decisive fields (e.g. mia_bench's component_*)
  // falls back to every non-prose field; the summary row is truncated anyway.
  const shown = decisive.length > 0
    ? decisive
    : entries.filter(([key]) => !/^(reasoning|explanation|rationale|evidence_quotes|missing_elements|notes)$/i.test(key))
  return shown
    .map(([key, value]) => {
      const text = typeof value === 'object' && value !== null ? JSON.stringify(value) : String(value)
      // Keep the key unless the value speaks for itself, so a bare `0` never stands for a score.
      return shown.length === 1 && SELF_EVIDENT_KEYS.has(key) ? text : `${key}: ${text}`
    })
    .join(' · ')
}

/** The row summary for a case, folding both sides of a pairwise verdict when present. */
function caseSummary(view: JudgeCaseView): string {
  if (view.placementVerdicts) {
    return Object.entries(view.placementVerdicts)
      .map(([placement, value]) => `${placement}: ${verdictSummary(value)}`)
      .join('  |  ')
  }
  return verdictSummary(view.verdict)
}

function StatusPill({ status }: { status: string }) {
  const { t } = useLocale()
  const usable = isUsableStatus(status)
  const color = usable
    ? (status === 'fallback' ? 'var(--warning-color)' : 'var(--success)')
    : 'var(--danger)'
  const Icon = usable ? Check : AlertTriangle
  const labelKey = STATUS_LABEL_KEYS[status]
  const label = labelKey ? t(labelKey) : status
  return (
    <span
      className="inline-flex items-center gap-[3px] px-[6px] py-[1px] rounded-[3px] bg-transparent border text-[0.58rem] font-mono font-medium opacity-85 whitespace-nowrap"
      style={{ borderColor: color, color }}
      title={label}
    >
      <Icon size={9} />
      {status}
    </span>
  )
}

/** One case row: collapsed it is a single line; expanded it shows the parsed verdict and raw text. */
function CaseRow({ view, showCaseId }: { view: JudgeCaseView; showCaseId: boolean }) {
  const { t } = useLocale()
  const [open, setOpen] = useState(false)
  const summary = caseSummary(view)

  return (
    <div className="border-t border-[var(--border)] first:border-t-0">
      <button
        onClick={() => setOpen(v => !v)}
        aria-expanded={open}
        className="flex w-full items-center gap-2 bg-transparent px-0 py-1.5 text-left"
      >
        {open ? <ChevronDown size={11} className="shrink-0 text-[var(--text-muted)]" />
          : <ChevronRight size={11} className="shrink-0 text-[var(--text-muted)]" />}
        {showCaseId && (
          <span className="shrink-0 font-mono text-[0.62rem] text-[var(--text-muted)]">{view.caseId}</span>
        )}
        <span className="min-w-0 flex-1 truncate type-body-xs text-[var(--text)]">
          {summary || <span className="opacity-60 italic">{t('prediction.judgeNoVerdict')}</span>}
        </span>
        {view.placements.map(placement => (
          <span
            key={placement}
            className="shrink-0 rounded-[3px] border border-[var(--border-md)] px-[5px] py-[1px] font-mono text-[0.55rem] text-[var(--text-muted)]"
          >
            {placement}
          </span>
        ))}
        {view.retries > 0 && (
          <span
            className="inline-flex shrink-0 items-center gap-[3px] font-mono text-[0.55rem]"
            style={{ color: 'var(--warning-color)' }}
          >
            <RotateCcw size={9} />
            {t('prediction.judgeRetries').replace('${count}', String(view.retries))}
          </span>
        )}
        <StatusPill status={view.status} />
      </button>

      {open && (
        <div className="flex flex-col gap-2 pb-2 pl-[1.15rem]">
          {view.verdict != null && (
            <div>
              <div className="type-label-xs mb-1 text-[var(--text-muted)]">
                {t('prediction.judgeParsedVerdict')}
              </div>
              <div className="rounded-[0.4rem] overflow-hidden border border-[var(--border)]">
                <JsonViewer value={view.verdict} maxHeight={200} />
              </div>
            </div>
          )}
          {view.attempts.map((attempt, index) => (
            <div key={`${attempt.judge_id}-${attempt.repeat_id ?? 0}-${attempt.case_id}-${attempt.placement ?? ''}-${index}`}>
              <div className="type-label-xs mb-1 flex items-center gap-1.5 text-[var(--text-muted)]">
                <span>{t('prediction.judgeRawResponse')}</span>
                <span className="font-mono normal-case opacity-70">
                  {t('prediction.judgeAttemptLabel').replace('${index}', String(index + 1))}
                  {attempt.placement ? ` · ${attempt.placement}` : ''}
                </span>
                <StatusPill status={attempt.status} />
              </div>
              {attempt.error && (
                <div className="mb-1 type-body-xs" style={{ color: 'var(--danger)' }}>{attempt.error}</div>
              )}
              {attempt.raw_response && (
                <pre className="max-h-48 overflow-auto rounded-[0.4rem] border border-[var(--border)] bg-[var(--bg-card2)] p-2 type-body-xs whitespace-pre-wrap break-words text-[var(--text-muted)]">
                  {attempt.raw_response}
                </pre>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

export interface JudgeReviewPanelProps {
  review: JudgeReviewView
}

/**
 * Structured view of the LLM judge's work on one sample.
 *
 * Rendered only for a judge-scored sample; a rule-scored one has no `judge_summary` and
 * `selectJudgeReview` returns `null`.
 */
export function JudgeReviewPanel({ review }: JudgeReviewPanelProps) {
  const { t } = useLocale()
  const [expanded, setExpanded] = useState(false)
  const failureEntries = Object.entries(review.failures)
  const cases = expanded ? review.cases : review.cases.slice(0, VISIBLE_CASES)
  // A single-case benchmark needs no case column: the row is the verdict.
  const showCaseId = review.cases.length > 1

  return (
    <section
      className="overflow-hidden rounded-xl border border-[var(--border-md)] bg-[var(--bg-card)] shadow-[var(--shadow-sm)]"
      style={{ animation: 'fadeInUp 300ms ease-out 200ms both' }}
    >
      <div className="flex flex-wrap items-center gap-x-3 gap-y-1 border-b border-[var(--border)] bg-[var(--bg-card2)] px-4 py-3">
        <span className="flex items-center gap-2">
          <Scale size={15} className="text-[var(--text-muted)]" />
          <span className="type-label-xs">{t('prediction.judgeReview')}</span>
        </span>
        {review.judgeModels.map(model => (
          <span key={model} className="font-mono type-body-xs text-[var(--text)]">{model}</span>
        ))}
        {review.totalObservations != null && (
          <span className="type-body-xs text-[var(--text-muted)]">
            {review.validObservations ?? 0}/{review.totalObservations} {t('prediction.judgeObservations')}
          </span>
        )}
        {failureEntries.map(([status, count]) => (
          <span
            key={status}
            className="inline-flex items-center gap-[3px] font-mono text-[0.58rem]"
            style={{ color: 'var(--warning-color)' }}
          >
            <AlertTriangle size={9} />
            {status} ×{count}
          </span>
        ))}
       {review.totalLatency != null && (
          <span className="inline-flex items-center gap-[3px] font-mono text-[0.58rem] text-[var(--text-muted)]">
            <Timer size={9} />
            {/* A diagnostic wall-clock total, not a report metric; one-decimal seconds is enough. */}
            {Math.round(review.totalLatency * 10) / 10}s
          </span>
        )}
      </div>

      {review.error && (
        <div className="border-b border-[var(--border)] px-4 py-2 type-body-xs" style={{ color: 'var(--danger)' }}>
          {review.error}
        </div>
      )}

      {review.cases.length > 0 && (
        <div className="px-4 py-1">
          {cases.map(view => (
            <CaseRow key={view.caseId} view={view} showCaseId={showCaseId} />
          ))}
          {review.cases.length > VISIBLE_CASES && (
            <button
              onClick={() => setExpanded(v => !v)}
              className="min-h-8 bg-transparent px-0 py-1 text-xs font-medium text-[var(--text-muted)] hover:text-[var(--text)]"
            >
              {expanded
                ? t('prediction.judgeShowFewerCases')
                : t('prediction.judgeShowAllCases').replace('${count}', String(review.cases.length))}
            </button>
          )}
        </div>
      )}

      {review.explanation && (
        <div className="border-t border-[var(--border)] px-4 py-3">
          <div className="type-label-xs mb-2 text-[var(--text-muted)]">{t('prediction.judgeExplanation')}</div>
          <div className="type-body-sm text-[var(--text)]">
            <MarkdownRenderer content={review.explanation} />
          </div>
        </div>
      )}
    </section>
  )
}
