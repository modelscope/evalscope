import { describe, expect, it } from 'vitest'
import { isUsableStatus, scoreWithoutJudgeAttempts, selectJudgeReview } from './judgeReview'
import type { PredictionScore } from '@/api/types'

/** A minimal attempt row as the backend serializes one. */
function attempt(overrides: Record<string, unknown>): Record<string, unknown> {
  return {
    status: 'success',
    case_id: 'match',
    judge_id: 'qwen-plus',
    attempt_index: 0,
    ...overrides,
  }
}

describe('selectJudgeReview', () => {
  it('returns null for a rule-scored sample', () => {
    const score = { value: { acc: 1 } } as PredictionScore
    expect(selectJudgeReview(score)).toBeNull()
    expect(selectJudgeReview(undefined)).toBeNull()
  })

  it('surfaces judge_detail even when no attempts were saved', () => {
    const score = {
      value: { acc: 1 },
      judge_detail: { judge_models: ['qwen-plus'], valid_observations: 1, total_observations: 1 },
    } as unknown as PredictionScore

    const review = selectJudgeReview(score)!
    expect(review.judgeModels).toEqual(['qwen-plus'])
    expect(review.cases).toHaveLength(0)
  })

  it('promotes the usable attempt to the case verdict', () => {
    const score = {
      judge_detail: { judge_models: ['qwen-plus'] },
      metadata: {
        judge_attempts: [attempt({ parsed_value: { verdict: 'A' }, latency: 1.5 })],
      },
    } as unknown as PredictionScore

    const review = selectJudgeReview(score)!
    expect(review.cases).toHaveLength(1)
    expect(review.cases[0].verdict).toEqual({ verdict: 'A' })
    expect(review.cases[0].status).toBe('success')
    expect(review.cases[0].retries).toBe(0)
    expect(review.totalLatency).toBe(1.5)
  })

  it('counts a parse-error retry that then succeeds as one retry', () => {
    const score = {
      metadata: {
        judge_attempts: [
          attempt({ status: 'parse_error', attempt_index: 0, error: 'bad json' }),
          attempt({ status: 'success', attempt_index: 1, parsed_value: { verdict: 'A' } }),
        ],
      },
    } as unknown as PredictionScore

    const view = selectJudgeReview(score)!.cases[0]
    expect(view.retries).toBe(1)
    expect(view.status).toBe('success')
    expect(view.attempts).toHaveLength(2)
  })

  it('marks a case excluded when every attempt failed', () => {
    const score = {
      metadata: {
        judge_attempts: [
          attempt({ status: 'parse_error', attempt_index: 0 }),
          attempt({ status: 'parse_error', attempt_index: 1 }),
        ],
      },
    } as unknown as PredictionScore

    const view = selectJudgeReview(score)!.cases[0]
    expect(view.verdict).toBeUndefined()
    expect(view.status).toBe('parse_error')
    expect(view.retries).toBe(2) // both attempts broke the contract
  })

  it('groups attempts by case and preserves first-seen order', () => {
    const score = {
      metadata: {
        judge_attempts: [
          attempt({ case_id: 'rubric_0', parsed_value: { score: 1 } }),
          attempt({ case_id: 'rubric_1', parsed_value: { score: 0 } }),
        ],
      },
    } as unknown as PredictionScore

    const review = selectJudgeReview(score)!
    expect(review.cases.map(c => c.caseId)).toEqual(['rubric_0', 'rubric_1'])
  })

  it('exposes per-placement verdicts for a pairwise case', () => {
    const score = {
      metadata: {
        judge_attempts: [
          attempt({ case_id: 'battle', placement: 'original', parsed_value: { verdict: 'B>>A' } }),
          attempt({ case_id: 'battle', placement: 'swapped', parsed_value: { verdict: 'A>>B' } }),
        ],
      },
    } as unknown as PredictionScore

    const view = selectJudgeReview(score)!.cases[0]
    expect(view.placements).toEqual(['original', 'swapped'])
    // Both sides are kept; neither is silently dropped, and a swap is not a retry.
    expect(view.placementVerdicts).toEqual({ original: { verdict: 'B>>A' }, swapped: { verdict: 'A>>B' } })
    expect(view.retries).toBe(0)
  })
})

describe('scoreWithoutJudgeAttempts', () => {
  it('drops judge_attempts but keeps the rest of metadata', () => {
    const score = {
      value: { acc: 1 },
      metadata: { source: 'llm_judge', judge_attempts: [attempt({})] },
    } as unknown as PredictionScore

    const stripped = scoreWithoutJudgeAttempts(score)
    expect(stripped.metadata?.judge_attempts).toBeUndefined()
    expect((stripped.metadata as Record<string, unknown>).source).toBe('llm_judge')
    // The original is untouched.
    expect(score.metadata?.judge_attempts).toHaveLength(1)
  })

  it('returns the same score when there are no attempts', () => {
    const score = { value: { acc: 1 } } as PredictionScore
    expect(scoreWithoutJudgeAttempts(score)).toBe(score)
  })
})

describe('isUsableStatus', () => {
  it('treats success and fallback as usable, everything else as not', () => {
    expect(isUsableStatus('success')).toBe(true)
    expect(isUsableStatus('fallback')).toBe(true)
    expect(isUsableStatus('parse_error')).toBe(false)
    expect(isUsableStatus('transport_error')).toBe(false)
    expect(isUsableStatus('excluded')).toBe(false)
  })
})
