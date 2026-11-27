import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, expect, it } from 'vitest'
import { LocaleProvider } from '@/contexts/LocaleContext'
import SampleDiagnosisCard from './SampleDiagnosisCard'

afterEach(() => {
  cleanup()
  localStorage.removeItem('evalscope-locale')
})

it('shows the observation and evidence without internal rule names or disclaimers', () => {
  localStorage.setItem('evalscope-locale', 'zh')
  const { container } = render(
    <LocaleProvider>
      <SampleDiagnosisCard
        diagnosis={{
          llm: null, llm_status: 'not_selected', sources: [], context_omitted: false, observed_types: ['truncated'], invalid_evidence: 0, recorded_error: null, evidence_errors: [],
          key: { subset: 'default', index: 7 },
          task_success: false,
          safety_violation: null,
          score_status: 'success',
          stop_reason: 'max_tokens',
          needs_attention: true,
          findings: [
            {
              category: 'tool_failure',
              subtype: 'provider:runtime_error',
              detector: 'tool',
              attribution: 'model',
              confidence: 'medium',
              evidence: [{ excerpt: 'ValueError: invalid date' }],
            },
          ],
        }}
      />
    </LocaleProvider>,
  )
  expect(container.querySelector('details')).not.toHaveAttribute('open')
  fireEvent.click(container.querySelector('summary') as HTMLElement)
  expect(container.querySelector('details')).toHaveAttribute('open')
  expect(screen.getByText(/达到输出长度上限/)).toBeInTheDocument()
  expect(screen.getByText('工具报错')).toBeInTheDocument()
  expect(screen.getByText('ValueError: invalid date')).toBeInTheDocument()
  expect(container.textContent).not.toMatch(/provider:runtime_error|max_tokens|置信度|根因/)
})

it('shows only grounded proposals and preserves Unicode citation boundaries', () => {
  const diagnosis = {
    key: { subset: 'default', index: 1 }, score_status: 'success', task_success: false,
    safety_violation: null, stop_reason: null, needs_attention: true,
    findings: [],
    observed_types: [], invalid_evidence: 0, recorded_error: null, evidence_errors: [], llm_status: 'ok',
    context_omitted: false, sources: [{ id: 'm:1', text: '中文🧪 answer B', message_index: 1, step: 2, excerpted: false, original_start: 0 }],
    llm: { failure_type: 'reasoning_error', secondary_types: [], attribution: 'model', root_cause_step: 2,
      explanation: 'Calculation omits the final factor.', confidence: 'medium', gold_suspect: true,
      evidence: [{ source_id: 'm:1', start: 2, end: 12, quote: '🧪 answer B' }] },
  } as import('@/api/types').SampleDiagnosis
  const { container, rerender } = render(<LocaleProvider><SampleDiagnosisCard diagnosis={diagnosis} /></LocaleProvider>)
  fireEvent.click(container.querySelector('summary') as HTMLElement)
  expect(screen.getByText('Calculation omits the final factor.')).toBeInTheDocument()
  const evidence = screen.getByText(/Evidence 1/)
  fireEvent.click(evidence)
  expect(container.querySelector('mark')).toHaveTextContent('🧪 answer B')
  expect(container.textContent).toContain('References are locatable')
  rerender(<LocaleProvider><SampleDiagnosisCard diagnosis={{ ...diagnosis, llm_status: 'invalid_evidence' }} /></LocaleProvider>)
  expect(screen.queryByText('Calculation omits the final factor.')).not.toBeInTheDocument()
  rerender(<LocaleProvider><SampleDiagnosisCard diagnosis={{ ...diagnosis, llm_status: 'invalid_evidence', context_omitted: true,
    sources: [{ ...diagnosis.sources[0], id: 'omitted-content', text: 'Not analyzed: image.' }],
  }} /></LocaleProvider>)
  expect(screen.getByText(/Media, internal data or protected content was not included/)).toBeInTheDocument()
})
