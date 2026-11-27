import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, expect, it, vi } from 'vitest'
import type { SampleDiagnosis, PredictionsResponse } from '@/api/types'
import { LocaleProvider } from '@/contexts/LocaleContext'
import { loadFixture } from '@/test/loadFixture'
import { useState, type ComponentProps } from 'react'
import DiagnosisSamplePanelView from './DiagnosisSamplePanel'

function DiagnosisSamplePanel({ initialPosition = 0, ...props }: Omit<ComponentProps<typeof DiagnosisSamplePanelView>, 'position' | 'onPositionChange'> & { initialPosition?: number }) {
  const [position, setPosition] = useState(initialPosition)
  return <DiagnosisSamplePanelView {...props} position={position} onPositionChange={setPosition} />
}

const { getSampleDiagnoses, getPredictions } = vi.hoisted(() => ({
  getSampleDiagnoses: vi.fn(),
  getPredictions: vi.fn(),
}))
vi.mock('@/api/reports', () => ({ getSampleDiagnoses, getPredictions }))
const base = loadFixture<PredictionsResponse>('predictions-tool-trace').predictions[0]
const sample = (subset: string, index: number): SampleDiagnosis => ({
  key: { subset, index },
  task_success: false,
  safety_violation: null,
  score_status: 'success',
  stop_reason: null,
  needs_attention: true,
  findings: [],
  observed_types: ['tool_error'],
  llm: null,
  llm_status: 'not_selected',
  sources: [], context_omitted: false,
  invalid_evidence: 0,
  recorded_error: null, evidence_errors: [],

})
async function settle() {
  for (let i = 0; i < 8; i++)
    await act(async () => {
      await Promise.resolve()
    })
}
afterEach(() => {
  cleanup()
  vi.resetAllMocks()
})

it('browses the issue across subsets, fetches only the selected prediction, and opens the same sample', async () => {
  getSampleDiagnoses.mockResolvedValue({
    status: 'ready',
    items: [sample('a', 3), sample('b', 8)],
    total: 2,
    page: 1,
    page_size: 200,
  })
  getPredictions.mockImplementation(async (_root, _report, _dataset, subset, _signal, index) => ({
    predictions: [
      {
        ...base,
        Index: String(index),
        Input: `Question ${subset}`,
        Generated: `Answer ${subset}`,
        Messages: null,
        AgentTrace: null,
      },
    ],
  }))
  const onSamples = vi.fn()
  render(
    <LocaleProvider>
      <DiagnosisSamplePanel
        rootPath="/outputs"
        reportName="run/m"
        datasetName="demo"
        filters={{ failure_type: 'tool_error', outcome: 'failed' }}
        onSamples={onSamples}
      />
    </LocaleProvider>,
  )
  await settle()
  expect(screen.getByText('Question a')).toBeInTheDocument()
  expect(screen.getByRole('button', { name: 'Previous sample' })).toBeDisabled()
  fireEvent.click(screen.getByRole('button', { name: 'Next sample' }))
  await settle()
  expect(screen.getByText('Question b')).toBeInTheDocument()
  expect(screen.queryByText('Question a')).not.toBeInTheDocument()
  expect(getPredictions).toHaveBeenLastCalledWith('/outputs', 'run/m', 'demo', 'b', expect.any(AbortSignal), 8)
  expect(screen.getByRole('button', { name: 'Next sample' })).toBeDisabled()
  fireEvent.click(screen.getByRole('button', { name: 'Open in sample details' }))
  expect(onSamples).toHaveBeenCalledWith({ failure_type: 'tool_error', outcome: 'failed' }, { subset: 'b', index: 8 }, undefined, 1)
  expect(screen.getByRole('navigation', { name: 'Sample contents' })).toBeInTheDocument()
})

it('continues to the next diagnostic page after 200 matching samples', async () => {
  getSampleDiagnoses.mockImplementation(async (_root, _report, _dataset, _filters, page) => ({
    status: 'ready',
    total: 201,
    page,
    page_size: 200,
    items: page === 1 ? Array.from({ length: 200 }, (_, index) => sample('a', index)) : [sample('b', 900)],
  }))
  getPredictions.mockImplementation(async (_root, _report, _dataset, _subset, _signal, index) => ({
    predictions: [{ ...base, Index: String(index), Messages: null, AgentTrace: null }],
  }))
  render(
    <LocaleProvider>
      <DiagnosisSamplePanel
        rootPath="/outputs"
        reportName="run/m"
        datasetName="demo"
        filters={{ outcome: 'failed' }}
        initialPosition={199}
        onSamples={vi.fn()}
      />
    </LocaleProvider>,
  )
  await settle()
  const next = screen.getByRole('button', { name: 'Next sample' })
  fireEvent.click(next)
  await settle()
  expect(screen.getByText('201 / 201')).toBeInTheDocument()
  expect(getSampleDiagnoses).toHaveBeenLastCalledWith(
    '/outputs',
    'run/m',
    'demo',
    { outcome: 'failed' },
    2,
    expect.any(AbortSignal),
  )
  expect(getPredictions).toHaveBeenLastCalledWith('/outputs', 'run/m', 'demo', 'b', expect.any(AbortSignal), 900)
})
