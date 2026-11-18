import { cleanup, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'
import { LocaleProvider } from '@/contexts/LocaleContext'
import { ThemeProvider } from '@/contexts/ThemeContext'
import { EvalResultPanel } from './EvalResultPanel'
import type { ReportMetric } from '@/domain/report/primaryMetrics'
import { RATIO_PERCENT_SEMANTICS } from '@/domain/report/primaryMetrics'

afterEach(cleanup)

function renderPanel(nScore: number | null, status?: string) {
  return render(
    <ThemeProvider>
      <LocaleProvider>
        <EvalResultPanel
          pred="B"
          gold="B"
          nScore={nScore}
          status={status}
          score={nScore === null ? {} : { acc: nScore }}
          metadata={{ id: '1' }}
          threshold={0.99}
          showPred
        />
      </LocaleProvider>
    </ThemeProvider>,
  )
}

describe('EvalResultPanel', () => {
  it('uses one label hierarchy and keeps the view filter out of score coloring', () => {
    renderPanel(1)

    expect(screen.getByText('Extracted Answer')).toHaveClass('type-label-xs')
    expect(screen.getByText('Expected Answer')).toHaveClass('type-label-xs')
    expect(screen.getByText('100%')).not.toHaveStyle({ color: 'var(--success)' })
  })

  it('renders multiple accepted answers separately', () => {
    return render(
      <ThemeProvider>
        <LocaleProvider>
          <EvalResultPanel
            pred="A"
            gold={['A', 'B']}
            nScore={1}
            score={{ acc: 1 }}
            metadata={{}}
            threshold={0.99}
            showPred
          />
        </LocaleProvider>
      </ThemeProvider>,
    )

    expect(screen.getByText('B')).toBeInTheDocument()
  })

  it('keeps verbose details collapsed without interpreting the filter as a verdict', () => {
    renderPanel(0.2)

    expect(screen.getByRole('button', { name: 'Score Detail' })).toHaveAttribute('aria-expanded', 'false')
    expect(screen.queryByText('"acc"')).not.toBeInTheDocument()
    expect(screen.getByText('20%')).not.toHaveStyle({ color: 'var(--danger)' })
  })

  it('labels lower-is-better attack success and exposes utility without opening raw JSON', () => {
    const metrics: ReportMetric[] = [
      {
        identity: { name: 'targeted_asr', aggregation: 'mean', dimensions: {} },
        semantics: { ...RATIO_PERCENT_SEMANTICS, metric_name: 'Targeted ASR', direction: 'lower_is_better' },
        score: 1, num: 1, categories: [],
      },
      {
        identity: { name: 'utility', aggregation: 'mean', dimensions: {} },
        semantics: { ...RATIO_PERCENT_SEMANTICS, metric_name: 'Utility' },
        score: 0, num: 1, categories: [],
      },
    ]
    render(<LocaleProvider><EvalResultPanel
      pred="" gold="" nScore={1} score={{ main_score_name: 'targeted_asr', value: { utility: 0, targeted_asr: 1 } }}
      metadata={{}} threshold={0.99} showPred={false} metrics={metrics}
    /></LocaleProvider>)
    expect(screen.getByText('Targeted ASR ↓')).toBeInTheDocument()
    expect(screen.getByText('Utility ↑')).toBeInTheDocument()
    expect(screen.getByText('100%')).toHaveAttribute('style', expect.stringContaining('hsl(0 '))
    expect(screen.getByText('0%')).toBeInTheDocument()
  })

  it('shows the judge failure reason instead of a 0% score when there is none', () => {
    renderPanel(null, 'parse_error')

    expect(screen.getByText('Unavailable')).toBeInTheDocument()
    expect(screen.getByText('Judge response could not be parsed')).toBeInTheDocument()
    expect(screen.queryByText('0%')).not.toBeInTheDocument()
    expect(screen.queryByText('Below filter')).not.toBeInTheDocument()
  })

  it('falls back to a generic hint for an unknown status', () => {
    renderPanel(null)

    expect(screen.getByText('Unavailable')).toBeInTheDocument()
    expect(
      screen.getByText('No usable score for this sample; it is excluded from the metric rather than counted as 0.'),
    ).toBeInTheDocument()
  })
})

it('keeps embedded scores compact and hides an absent reference answer', () => {
  const { container } = render(<LocaleProvider><EvalResultPanel compact pred="" gold="*No Gold Provided*" nScore={0}
    score={{ value: { targeted_asr: 0, utility: 0 }, main_score_name: 'targeted_asr' }} metadata={{}} threshold={0.99} showPred={false} /></LocaleProvider>)
  const details = container.querySelector('details')!
  expect(details).not.toHaveAttribute('open')
  expect(details.querySelector('summary')).toHaveTextContent('targeted_asr')
  expect(details.querySelector('summary')).toHaveTextContent('utility')
  expect(screen.queryByText('No Gold Provided')).not.toBeInTheDocument()
  expect(screen.queryByText('Expected Answer')).not.toBeInTheDocument()
})

it('shows the recorded scoring error as plain text outside collapsed details', () => {
  render(<LocaleProvider><EvalResultPanel
    compact pred="" gold="" nScore={null} status="excluded" score={{}} metadata={{}}
    threshold={0.99} showPred={false} recordedError="Distribution exceeds the probability sum tolerance."
  /></LocaleProvider>)
  expect(screen.getByText('Error in the evaluation record')).toBeVisible()
  expect(screen.getByText('Distribution exceeds the probability sum tolerance.')).toBeVisible()
  expect(screen.getByText('Error in the evaluation record').closest('div')?.nextElementSibling?.querySelector('summary'))
    .toHaveTextContent('Unavailable')
})

it('keeps the extracted and reference answers visible in the compact QA score', () => {
  const { container } = render(<LocaleProvider><EvalResultPanel compact pred="12" gold="13" nScore={0}
    score={{ value: { accuracy: 0 }, main_score_name: 'accuracy' }} metadata={{}} threshold={0.99} showPred /></LocaleProvider>)
  const summary = container.querySelector('summary')!
  expect(summary).toHaveTextContent('Extracted Answer: 12')
  expect(summary).toHaveTextContent('Expected Answer: 13')
  expect(container.querySelector('details')).not.toHaveAttribute('open')
})
