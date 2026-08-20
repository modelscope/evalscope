import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'
import { ThemeProvider } from '@/contexts/ThemeContext'
import { LocaleProvider } from '@/contexts/LocaleContext'
import { JudgeReviewPanel } from './JudgeReviewPanel'
import { selectJudgeReview } from '@/domain/chat/judgeReview'
import type { PredictionScore } from '@/api/types'

afterEach(cleanup)

/**
 * Build the panel from a score payload shaped exactly as the API delivers it.
 *
 * Flask serializes with sorted keys, so `parsed_value` arrives alphabetically ordered -- the
 * fixtures below preserve that, because it is what made the decisive field fall off the row.
 */
function renderPanel(score: unknown) {
  const review = selectJudgeReview(score as PredictionScore)
  if (!review) throw new Error('expected a judge review')
  return render(
    <ThemeProvider>
      <LocaleProvider>
        <JudgeReviewPanel review={review} />
      </LocaleProvider>
    </ThemeProvider>,
  )
}

describe('JudgeReviewPanel', () => {
  it('shows a rule short-circuit reason without inventing judge activity', () => {
    renderPanel({
      metadata: { judge_skipped: true, judge_skip_reason: 'exact_string_match' },
    })

    expect(screen.getByText('Rule-based score')).toBeTruthy()
    expect(screen.getByText('LLM judge was not called')).toBeTruthy()
    expect(screen.getByText('Reason: exact_string_match')).toBeTruthy()
    expect(screen.queryByText('observations')).toBeNull()
  })

  it('shows the judge model, observation count and a single case row', () => {
    renderPanel({
      judge_summary: { judge_models: ['qwen-plus'], valid_observations: 1, total_observations: 1 },
      metadata: {
        judge_attempts: [{
          status: 'success',
          case_id: 'grade',
          judge_id: 'qwen-plus',
          attempt_index: 0,
          placement: 'original',
          parsed_value: { reasoning: 'because', verdict: 'A' },
          raw_response: '{"verdict": "A"}',
          latency: 1.24,
        }],
      },
    })

    expect(screen.getByText('qwen-plus')).toBeTruthy()
    expect(screen.getByText(/1\/1/)).toBeTruthy()
    expect(screen.getByText('1.2s')).toBeTruthy()
    // The decision is on the row; the prose is not.
    expect(screen.getByText('A')).toBeTruthy()
    expect(screen.queryByText(/because/)).toBeNull()
  })

  it('puts the decisive field on the row even when the API sorts it last', () => {
    // Regression: researchrubrics' BinaryGrade arrives as
    // confidence, evidence_quotes, missing_elements, reasoning, score, verdict.
    // Showing keys in that order pushed `verdict` past the truncation point.
    renderPanel({
      metadata: {
        judge_attempts: [{
          status: 'success',
          case_id: 'rubric_0',
          judge_id: 'qwen-plus',
          attempt_index: 0,
          parsed_value: {
            confidence: 0.99,
            evidence_quotes: ['a very long quote that would otherwise dominate the row'],
            missing_elements: [],
            reasoning: 'long prose',
            score: 1.0,
            verdict: 'Satisfied',
          },
        }],
      },
    })

    const row = screen.getByText(/Satisfied/)
    expect(row).toBeTruthy()
    // Noise must not lead the summary.
    expect(row.textContent?.startsWith('confidence')).toBe(false)
    expect(row.textContent).not.toContain('evidence_quotes')
  })

  it('reveals the parsed verdict and raw reply when a case row is expanded', () => {
    renderPanel({
      metadata: {
        judge_attempts: [{
          status: 'success',
          case_id: 'grade',
          judge_id: 'qwen-plus',
          attempt_index: 0,
          parsed_value: { verdict: 'A' },
          raw_response: '{"verdict": "A", "reasoning": "spelled out"}',
        }],
      },
    })

    expect(screen.queryByText(/Raw judge reply/i)).toBeNull()
    fireEvent.click(screen.getByRole('button', { expanded: false }))
    expect(screen.getByText(/Parsed verdict/i)).toBeTruthy()
    expect(screen.getByText(/Raw judge reply/i)).toBeTruthy()
    expect(screen.getByText(/spelled out/)).toBeTruthy()
  })

  it('shows both sides of a pairwise case and does not call a swap a retry', () => {
    renderPanel({
      metadata: {
        judge_attempts: [
          {
            status: 'success', case_id: 'battle', judge_id: 'j', attempt_index: 0,
            placement: 'original', parsed_value: { reasoning: 'x', verdict: 'B>>A' },
          },
          {
            status: 'success', case_id: 'battle', judge_id: 'j', attempt_index: 0,
            placement: 'swapped', parsed_value: { reasoning: 'y', verdict: 'A>>B' },
          },
        ],
      },
    })

    // Regression: the row used to show only the swapped verdict and label the pair "retried 1x".
    const row = screen.getByText(/original: B>>A/)
    expect(row.textContent).toContain('swapped: A>>B')
    expect(screen.queryByText(/retried/)).toBeNull()
    expect(screen.getByText('original')).toBeTruthy()
    expect(screen.getByText('swapped')).toBeTruthy()
  })

  it('keeps the key for a numeric verdict so a bare 0 never stands for a score', () => {
    renderPanel({
      metadata: {
        judge_attempts: [{
          status: 'success', case_id: 'grade', judge_id: 'j', attempt_index: 0,
          parsed_value: { grading_rationale: 'prose', overall_score: 0, requirement_status: [] },
        }],
      },
    })

    expect(screen.getByText('overall_score: 0')).toBeTruthy()
  })

  it('flags a retry and surfaces the failure count', () => {
    renderPanel({
      judge_summary: { judge_models: ['qwen-plus'], failures: { parse_error: 1 } },
      metadata: {
        judge_attempts: [
          {
            status: 'parse_error', case_id: 'grade', judge_id: 'qwen-plus', attempt_index: 0,
            error: 'reply does not satisfy CLGrade',
          },
          {
            status: 'success', case_id: 'grade', judge_id: 'qwen-plus', attempt_index: 1,
            parsed_value: { overall_score: 0 },
          },
        ],
      },
    })

    expect(screen.getByText(/retried 1/)).toBeTruthy()
    expect(screen.getByText(/parse_error ×1/)).toBeTruthy()
  })

  it('collapses a long case list behind a show-all toggle', () => {
    renderPanel({
      metadata: {
        judge_attempts: Array.from({ length: 28 }, (_, i) => ({
          status: 'success',
          case_id: `rubric_${i}`,
          judge_id: 'qwen-plus',
          attempt_index: 0,
          parsed_value: { verdict: 'Satisfied', score: 1 },
        })),
      },
    })

    expect(screen.queryByText('rubric_27')).toBeNull()
    const toggle = screen.getByText(/Show all 28 cases/)
    fireEvent.click(toggle)
    expect(screen.getByText('rubric_27')).toBeTruthy()
    expect(screen.getByText(/Show fewer/)).toBeTruthy()
  })

  it('marks a case with no usable verdict instead of inventing one', () => {
    renderPanel({
      judge_summary: { judge_models: ['j'], valid_observations: 0, total_observations: 1, error: 'no judge produced a usable verdict' },
      metadata: {
        judge_attempts: [{
          status: 'parse_error', case_id: 'grade', judge_id: 'j', attempt_index: 0, raw_response: 'prose',
        }],
      },
    })

    expect(screen.getByText(/No verdict/i)).toBeTruthy()
    expect(screen.getByText(/no judge produced a usable verdict/)).toBeTruthy()
  })
})
