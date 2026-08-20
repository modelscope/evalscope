/**
 * Folds a score's judge diagnostics into the shape the review panel renders.
 *
 * The raw payload is attempt-oriented: one row per request, including the retries that failed.
 * A reader wants it case-oriented -- "what did the judge decide about rubric 3, and how hard was
 * it to get there" -- so attempts are grouped by case and the usable one is promoted to the
 * case's verdict.
 */
import type { JudgeAttempt, PredictionScore } from '@/api/types'

/** Statuses that carry a usable verdict; mirrors `ScoreStatus.is_usable`. */
const USABLE_STATUSES = new Set(['success', 'fallback', 'degraded'])

export interface JudgeCaseView {
  caseId: string
  /** The parsed verdict of the usable attempt, or `undefined` when every attempt failed. */
  verdict?: unknown
  /**
   * Per-placement verdicts, set only when the case was judged in both orders. A pairwise case is
   * one atomic observation, so neither side alone is "the" verdict.
   */
  placementVerdicts?: Record<string, unknown>
  /** Status of the case as a whole: the usable attempt's, else the last attempt's. */
  status: string
  /** Every round trip for this case, in order, including failed retries. */
  attempts: JudgeAttempt[]
  /** Attempts the judge wasted on a reply that broke the contract. */
  retries: number
  /** Distinct placements seen, present only for a pairwise benchmark. */
  placements: string[]
}

export interface JudgeReviewView {
  /** True when a deterministic rule produced the score without calling an LLM judge. */
  skipped: boolean
  /** Machine-readable reason supplied by ``JudgeDefinition.skip``. */
  skipReason?: string
  judgeModels: string[]
  validObservations?: number
  totalObservations?: number
  /** Attempt counts keyed by `ScoreStatus`, e.g. `{ parse_error: 2 }`. */
  failures: Record<string, number>
  /** Why the score is unavailable, when the executor recorded a reason. */
  error?: string
  /** Summed latency across every attempt, in seconds; `undefined` when untimed. */
  totalLatency?: number
  cases: JudgeCaseView[]
  /** The adapter's own narrative, e.g. a per-rubric breakdown. */
  explanation?: string
}

/**
 * Build the view model, or `null` when this sample has neither judge diagnostics nor a rule short-circuit.
 *
 * A rule short-circuit has no judge attempts but remains visible so users can see why an LLM was
 * not called.
 */
export function selectJudgeReview(score: PredictionScore | undefined): JudgeReviewView | null {
  if (!score) return null
  const detail = score.judge_summary
  const attempts = score.metadata?.judge_attempts ?? []
  const skipped = score.metadata?.judge_skipped === true
  if (!detail && attempts.length === 0 && !skipped) return null

  const latencies = attempts.map(a => a.latency).filter((v): v is number => typeof v === 'number')

  return {
    skipped,
    skipReason: score.metadata?.judge_skip_reason,
    judgeModels: detail?.judge_models ?? [],
    validObservations: detail?.valid_observations,
    totalObservations: detail?.total_observations,
    failures: detail?.failures ?? {},
    error: detail?.error,
    totalLatency: latencies.length > 0 ? latencies.reduce((a, b) => a + b, 0) : undefined,
    cases: groupByCase(attempts),
    explanation: score.explanation || undefined,
  }
}

/** Group attempts into cases, preserving first-seen case order. */
function groupByCase(attempts: JudgeAttempt[]): JudgeCaseView[] {
  const byCase = new Map<string, JudgeAttempt[]>()
  for (const attempt of attempts) {
    // Group by case, not by placement: pairwise swaps are two attempts of the same case.
    const key = `${attempt.judge_id}:${attempt.repeat_id ?? 0}:${attempt.case_id}`
    const bucket = byCase.get(key)
    if (bucket) bucket.push(attempt)
    else byCase.set(key, [attempt])
  }

  return [...byCase.values()].map((caseAttempts) => {
    const caseId = caseAttempts[0].case_id
    const usable = caseAttempts.filter(a => USABLE_STATUSES.has(a.status))
    const decisive = usable.length > 0 ? usable[usable.length - 1] : undefined
    const placements = [...new Set(caseAttempts.map(a => a.placement).filter((p): p is string => !!p))]
    // A pairwise case is judged in both orders and the pair is one verdict, so each side is
    // reported separately instead of letting the last one stand for the case.
    const placementVerdicts = placements.length > 1
      ? Object.fromEntries(
        placements
          .map(p => [p, usable.find(a => a.placement === p)?.parsed_value] as const)
          .filter(([, value]) => value !== undefined),
      )
      : undefined
    return {
      caseId,
      verdict: decisive?.parsed_value,
      placementVerdicts,
      status: decisive?.status ?? caseAttempts[caseAttempts.length - 1].status,
      attempts: caseAttempts,
      // Count the attempts that broke the contract. Counting "attempts before the usable one"
      // instead would report a position swap as a retry, since both sides succeed in sequence.
      retries: caseAttempts.filter(a => !USABLE_STATUSES.has(a.status)).length,
      placements: placements.length > 1 ? placements : [],
    }
  })
}

/** Whether a status carries a usable verdict. */
export function isUsableStatus(status: string): boolean {
  return USABLE_STATUSES.has(status)
}

/**
 * The score without its `judge_attempts`, for the raw Score Detail view.
 *
 * The attempts are rendered by the review panel; leaving them in the JSON blob too would repeat
 * every raw judge reply, which for a rubric benchmark is the bulk of a 100 KB payload.
 */
export function scoreWithoutJudgeAttempts(score: PredictionScore): PredictionScore {
  if (!score.metadata?.judge_attempts) return score
  const metadata = { ...score.metadata }
  delete metadata.judge_attempts
  return { ...score, metadata }
}
