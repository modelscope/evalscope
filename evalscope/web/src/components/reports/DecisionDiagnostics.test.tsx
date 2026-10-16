import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { axe } from 'jest-axe'
import type { ReportData } from '@/api/types'
import { LocaleProvider } from '@/contexts/LocaleContext'
import { decisionCurves, decisionObservation } from '@/domain/report/decisionDiagnostics'
import { EvalResultPanel } from '@/components/chat/EvalResultPanel'
import DecisionPanel from '@/components/chat/DecisionPanel'
import DecisionDiagnostics from './DecisionDiagnostics'

beforeEach(() => localStorage.setItem('evalscope-locale', 'en'))
afterEach(cleanup)

const observation = {
  kind: 'score', labels: ['0', '1', '2'], probabilities: [0.2, 0.3, 0.5],
  raw_probabilities: { '0': 0.199, '1': 0.3, '2': 0.5 },
  predicted: '2', gold: 1.5, expected: 1.3, confidence: 0.6, continuous: true,
}

function reportFixture(): ReportData {
  return {
    schema_version: 2, name: 'fixture', dataset_name: 'stsb', model_name: 'fixture', analysis: '',
    metrics: ['top_probability', 'provider_confidence'].map((basis) => ({
      identity: { name: 'aurc', aggregation: 'decision', dimensions: {
        kind: 'score', levels: 3, confidence: basis, risk: 'normalized_mae',
      } },
      num: 3, score: 0.15, semantics: {
        semantic_id: 'quality.loss.unbounded', metric_name: 'AURC', kind: 'quality',
        direction: 'lower_is_better', display_kind: 'number', display_precision: 4,
      },
      categories: [{ name: ['default'], num: 3, score: 0.15, subsets: [{
        name: 'test', score: 0.15, num: 3,
        metadata: {
          [`risk_coverage_${basis}`]: [{ coverage: 1 / 3, risk: 0.1 }, { coverage: 1, risk: 0.2 }],
          unavailable: { correlations: 'Need nonconstant predictions and golds.' },
        },
      }] }],
    })),
  }
}

describe('Decision result interactions', () => {
  it('preserves original probabilities and ordered levels when switching views', () => {
    render(<LocaleProvider><DecisionPanel metadata={{ decision: observation }} /></LocaleProvider>)
    expect(screen.getByText('Probability-weighted score')).toBeInTheDocument()
    expect(screen.getByText('1.3')).toBeInTheDocument()
    expect(screen.getByText('Most likely level')).toBeInTheDocument()
    expect(screen.getByText('20%')).toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: 'Original service probabilities' }))
    expect(screen.getByText('19.9%')).toBeInTheDocument()
    expect(screen.getAllByRole('meter').map((el) => el.getAttribute('aria-label'))).toEqual(['0', '1', '2'])
    expect(screen.getByText(/Provider confidence:/)).toHaveTextContent('Provider confidence: 0.6')
    expect(screen.getByRole('meter', { name: '0' })).toHaveAttribute('aria-valuenow', '0.199')
    expect(screen.getByRole('button', { name: 'Original service probabilities' })).toHaveAttribute('aria-pressed', 'true')
    fireEvent.click(screen.getByRole('button', { name: 'Probabilities used for scoring' }))
    expect(screen.getByRole('meter', { name: '0' })).toHaveAttribute('aria-valuenow', '0.2')
  })

  it('keeps Noul P(yes) and its classification threshold explicit', () => {
    render(<LocaleProvider><DecisionPanel metadata={{ decision: {
      ...observation, kind: 'noul', labels: ['no', 'yes'], probabilities: [0.4, 0.6],
      raw_probabilities: { no: 0.4, yes: 0.6 }, threshold: 0.7, predicted: 'no', expected: null,
    } }} /></LocaleProvider>)
    expect(screen.getByText('P(yes) classification threshold: 0.7')).toBeInTheDocument()
    expect(screen.getByText('no', { selector: 'dd' })).toBeInTheDocument()
    expect(screen.getByRole('meter', { name: 'yes' })).toHaveAttribute('aria-valuenow', '0.6')
  })

  it('does not add decision UI to ordinary or invalid cached outputs', () => {
    const { container } = render(<LocaleProvider><DecisionPanel metadata={{}} /></LocaleProvider>)
    expect(container).toBeEmptyDOMElement()
    for (const probabilities of [[0.2], [NaN, 0.3, 0.5], ['0.2', 0.3, 0.5], [-0.2, 0.3, 0.9]]) {
      expect(decisionObservation({ decision: { ...observation, probabilities } })).toBeNull()
    }
  })

  it('keeps a confident wrong prediction distinct from its reference and handles zero probability', () => {
    render(<LocaleProvider><DecisionPanel metadata={{ decision: {
      kind: 'choice', labels: ['A', 'B', 'C'], probabilities: [0.99, 0.01, 0],
      predicted: 'A', gold: 'B',
    } }} /></LocaleProvider>)
    expect(screen.getByText('A', { selector: 'dd' })).toBeInTheDocument()
    expect(screen.getByText('B', { selector: 'dd' })).toBeInTheDocument()
    expect(screen.getByRole('meter', { name: 'A' })).toHaveAttribute('aria-valuetext', '99%')
    expect(screen.getByRole('meter', { name: 'C' })).toHaveAttribute('aria-valuenow', '0')
    expect(screen.queryByRole('button', { name: 'Original service probabilities' })).not.toBeInTheDocument()
    expect(screen.getByText(/not correctness/)).toBeInTheDocument()
  })

  it('renders continuous decision MAE as an error rather than a percentage score', () => {
    render(<LocaleProvider><EvalResultPanel
      pred="1.3" gold="1.5" nScore={0.2} threshold={0.99} showPred={false}
      score={{ value: { mae: 0.2 }, metadata: { decision: observation } }} metadata={{}}
    /></LocaleProvider>)
    expect(screen.getByText('MAE ↓ · 0.2')).toBeInTheDocument()
    expect(screen.queryByText('20%')).not.toBeInTheDocument()
  })

  it('shows semantic Noul gold instead of the original multiple-choice letter', () => {
    render(<LocaleProvider><EvalResultPanel
      pred="yes" gold="B" nScore={0} threshold={0.99} showPred
      score={{ value: { acc: 0 }, metadata: { decision: {
        ...observation, kind: 'noul', labels: ['no', 'yes'], probabilities: [0.4, 0.6],
        gold: 'no', predicted: 'yes', expected: null, continuous: false,
      } } }} metadata={{}}
    /></LocaleProvider>)
    expect(screen.getByText('no')).toBeInTheDocument()
    expect(screen.queryByText('B')).not.toBeInTheDocument()
  })

  it('switches confidence sources and explores measured coverage without splitting ties', () => {
    render(<LocaleProvider><DecisionDiagnostics report={reportFixture()} /></LocaleProvider>)
    expect(screen.getByRole('status', { name: 'Coverage: 100% · Risk: 20%' })).toBeInTheDocument()
    fireEvent.change(screen.getByRole('slider'), { target: { value: '0' } })
    expect(screen.getByRole('status', { name: 'Coverage: 33.3% · Risk: 10%' })).toBeInTheDocument()
    const selection = screen.getByLabelText('Subset, decision type and confidence source')
    fireEvent.change(selection, { target: { value: decisionCurves(reportFixture())[1].id } })
    expect(screen.getByRole('status', { name: 'Coverage: 100% · Risk: 20%' })).toBeInTheDocument()
    fireEvent.click(screen.getByText('Unavailable metrics'))
    expect(screen.getByText(/Need nonconstant/)).toBeInTheDocument()
    expect(screen.getByText(/MAE divided by the score range/)).toBeInTheDocument()
  })

  it('skips missing or malformed curves instead of inventing risk values', () => {
    expect(decisionCurves()).toEqual([])
    const report = reportFixture()
    report.metrics[0].categories[0].subsets[0].metadata = {
      risk_coverage_top_probability: [{ coverage: 0.5, risk: NaN }],
    }
    expect(decisionCurves(report)).toHaveLength(1)
  })

  it('disables coverage selection when tied confidence leaves only one measured point', () => {
    const report = reportFixture()
    report.metrics[0].categories[0].subsets[0].metadata = {
      risk_coverage_top_probability: [{ coverage: 1, risk: 0.2 }],
    }
    render(<LocaleProvider><DecisionDiagnostics report={report} /></LocaleProvider>)
    expect(screen.getByRole('slider')).toBeDisabled()
    expect(screen.getByRole('status', { name: 'Coverage: 100% · Risk: 20%' })).toBeInTheDocument()
  })

  it('exposes probability values and coverage controls without accessibility violations', async () => {
    const { container } = render(<LocaleProvider>
      <DecisionPanel metadata={{ decision: observation }} />
      <DecisionDiagnostics report={reportFixture()} />
    </LocaleProvider>)
    vi.useRealTimers()
    try {
      expect((await axe(container)).violations).toEqual([])
    } finally {
      vi.useFakeTimers()
      vi.setSystemTime(new Date('2026-07-01T00:00:00.000Z'))
    }
  })
})
