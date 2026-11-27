import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { afterEach, describe, expect, it, vi } from 'vitest'

import type { PredictionsResponse } from '@/api/types'
import { DomainError } from '@/api/errors'
import { LocaleProvider } from '@/contexts/LocaleContext'
import { loadFixture } from '@/test/loadFixture'
import PredictionsTab from './PredictionsTab'

const { getDataFrameMock, getPredictionsMock, getDiagnosisMock, getSampleDiagnosesMock } = vi.hoisted(() => ({
  getDiagnosisMock: vi.fn(),
  getSampleDiagnosesMock: vi.fn(),
  getDataFrameMock: vi.fn(),
  getPredictionsMock: vi.fn(),
}))

vi.mock('@/api/reports', () => ({
  getDiagnosis: getDiagnosisMock,
  getSampleDiagnoses: getSampleDiagnosesMock,
  getDataFrame: getDataFrameMock,
  getPredictions: getPredictionsMock,
}))

afterEach(() => {
  cleanup()
  vi.restoreAllMocks()
  getDataFrameMock.mockReset()
  getPredictionsMock.mockReset()
})

describe('PredictionsTab controls', () => {
  it('localizes filter labels and exposes programmatic names for every input and pager button', async () => {
    getDataFrameMock.mockResolvedValue({ columns: ['Subset'], data: [{ Subset: 'example' }] })
    getPredictionsMock.mockResolvedValue(loadFixture<PredictionsResponse>('predictions-tool-trace'))

    render(
      <MemoryRouter>
        <LocaleProvider>
          <PredictionsTab reportName="real-report" datasetName="general_mcq" rootPath="/outputs" />
        </LocaleProvider>
      </MemoryRouter>,
    )

    await act(async () => {
      await Promise.resolve()
      await Promise.resolve()
      await Promise.resolve()
    })

    expect(screen.getByRole('spinbutton', { name: 'Score Threshold' })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /Above filter/ })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /Below filter/ })).toBeInTheDocument()
    expect(screen.queryByText('prediction.aboveFilter')).not.toBeInTheDocument()
    expect(screen.queryByText('prediction.belowFilter')).not.toBeInTheDocument()
    expect(screen.getByRole('textbox', { name: 'Go to index' })).toHaveAttribute('name', 'prediction-index-search')
    expect(screen.getByRole('textbox', { name: 'Find msg id' })).toHaveAttribute('name', 'prediction-message-id-search')
    expect(screen.getByRole('button', { name: 'Previous sample' })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Next sample' })).toBeInTheDocument()
    expect(screen.getByText('You are a helpful math assistant. Use the calculator tool when arithmetic is required.')).toBeInTheDocument()
    expect(screen.getByText('The customer receives 29 dollars in change.')).toBeInTheDocument()
    expect(screen.queryByText('Summary')).not.toBeInTheDocument()
  })

  it('shows a validation error instead of silently replacing the view with an empty state', async () => {
    getDataFrameMock.mockResolvedValue({ columns: ['Subset'], data: [{ Subset: 'example' }] })
    getPredictionsMock.mockRejectedValue(new DomainError('validation', 'Prediction response schema mismatch'))

    render(
      <MemoryRouter>
        <LocaleProvider>
          <PredictionsTab reportName="real-report" datasetName="general_mcq" rootPath="/outputs" />
        </LocaleProvider>
      </MemoryRouter>,
    )

    await act(async () => {
      await Promise.resolve()
      await Promise.resolve()
      await Promise.resolve()
    })

    expect(screen.getByRole('alert')).toHaveTextContent('Prediction response schema mismatch')
    expect(screen.getByRole('button', { name: 'Retry' })).toBeInTheDocument()
  })

  it('silently ignores an intentionally aborted subset request', async () => {
    const consoleError = vi.spyOn(console, 'error').mockImplementation(() => undefined)
    getDataFrameMock.mockRejectedValue(new DomainError('aborted', 'Request was aborted'))

    render(
      <MemoryRouter>
        <LocaleProvider>
          <PredictionsTab reportName="real-report" datasetName="general_mcq" rootPath="/outputs" />
        </LocaleProvider>
      </MemoryRouter>,
    )

    await act(async () => {
      await Promise.resolve()
      await Promise.resolve()
    })

    expect(screen.queryByRole('alert')).not.toBeInTheDocument()
    expect(consoleError).not.toHaveBeenCalled()
    consoleError.mockRestore()
  })
})


it('opens a linked sample after asynchronous prediction loading', async () => {
  getDataFrameMock.mockResolvedValue({columns:['Subset'], data:[{Subset:'example'}]})
  const fixture = loadFixture<PredictionsResponse>('predictions-tool-trace')
  getPredictionsMock.mockResolvedValue({...fixture, predictions: [
    {...fixture.predictions[0], Index:'1'}, {...fixture.predictions[0], Index:'7'},
  ]})
  getDiagnosisMock.mockResolvedValue({status:'ready', report:{patterns:[]}})
  getSampleDiagnosesMock.mockResolvedValue({status:'ready', items:[], total:0})
  render(<MemoryRouter><LocaleProvider><PredictionsTab reportName="run/m" datasetName="demo"
    rootPath="/outputs" initialSubset="example" initialIndex={7} onDiagnosisFiltersChange={vi.fn()} /></LocaleProvider></MemoryRouter>)
  await act(async () => { await Promise.resolve(); await Promise.resolve(); await Promise.resolve() })
  expect(screen.getByText('(index: 7)')).toBeInTheDocument()
  expect(screen.getByText('Sample 2 / 2')).toBeInTheDocument()
})


it.each([false, true])('keeps diagnostic filters secondary and opens them for a linked filter (%s)', async (active) => {
  getDataFrameMock.mockResolvedValue({ columns: ['Subset'], data: [{ Subset: 'example' }] })
  getPredictionsMock.mockResolvedValue(loadFixture<PredictionsResponse>('predictions-tool-trace'))
  getDiagnosisMock.mockResolvedValue({ status: 'ready', report: { patterns: [] } })
  getSampleDiagnosesMock.mockResolvedValue({ status: 'ready', items: [], total: 0 })
  render(<MemoryRouter><LocaleProvider><PredictionsTab reportName="run/m" datasetName="demo"
    rootPath="/outputs" diagnosisFilters={active ? { finding_category: 'tool_failure' } : {}}
    onDiagnosisFiltersChange={vi.fn()} /></LocaleProvider></MemoryRouter>)
  await act(async () => { await Promise.resolve(); await Promise.resolve(); await Promise.resolve() })
  const summary = screen.getByText('Diagnostic filters')
  expect(screen.getByRole('combobox', { name: 'Select Subset' }).compareDocumentPosition(summary) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy()
  if (active) {
    expect(summary.closest('details')).not.toHaveAttribute('open')
    fireEvent.click(summary)
    expect(summary.closest('details')).toHaveAttribute('open')
    expect(screen.getByRole('combobox', { name: 'Observation type' })).toBeInTheDocument()
  } else {
    expect(summary.closest('details')).not.toHaveAttribute('open')
    fireEvent.click(summary)
    expect(screen.getByRole('combobox', { name: 'Observation type' })).toBeInTheDocument()
  }
})

it('keeps the sample link aligned after manual navigation', async () => {
  const fixture = loadFixture<PredictionsResponse>('predictions-tool-trace')
  getDataFrameMock.mockResolvedValue({ columns: ['Subset'], data: [{ Subset: 'example' }] })
  getPredictionsMock.mockResolvedValue({ ...fixture, predictions: [
    { ...fixture.predictions[0], Index: '7' }, { ...fixture.predictions[0], Index: '42' },
  ] })
  getDiagnosisMock.mockResolvedValue({ status: 'missing', report: null })
  getSampleDiagnosesMock.mockResolvedValue({ status: 'missing', items: [], total: 0, page: 1, page_size: 200 })
  const navigate = vi.fn()
  render(<MemoryRouter><LocaleProvider><PredictionsTab reportName="run/m" datasetName="demo" rootPath="/outputs"
    onDiagnosisFiltersChange={navigate} /></LocaleProvider></MemoryRouter>)
  for (let i = 0; i < 8; i++) await act(async () => { await Promise.resolve() })
  fireEvent.click(screen.getByRole('button', { name: 'Next sample' }))
  expect(navigate).toHaveBeenLastCalledWith({ subset_name: 'example' }, { subset: 'example', index: 42 })
})

it('opens the exact subset from a diagnosis link', async () => {
  getDataFrameMock.mockResolvedValue({ columns: ['Subset'], data: [{ Subset: 'empty' }, { Subset: 'example' }] })
  getPredictionsMock.mockResolvedValue(loadFixture<PredictionsResponse>('predictions-tool-trace'))
  getDiagnosisMock.mockResolvedValue({
    status: 'ready',
    report: { patterns: [{ failure_type: 'tool_error' }] },
  })
  getSampleDiagnosesMock.mockResolvedValue({ status: 'ready', items: [], total: 0 })
  render(
    <MemoryRouter>
      <LocaleProvider>
        <PredictionsTab
          reportName="run/m"
          datasetName="demo"
          rootPath="/outputs"
          initialSubset="example"
          diagnosisFilters={{ failure_type: 'tool_error', outcome: 'failed' }}
          onDiagnosisFiltersChange={vi.fn()}
        />
      </LocaleProvider>
    </MemoryRouter>,
  )
  for (let i = 0; i < 8; i++)
    await act(async () => {
      await Promise.resolve()
    })
  expect(screen.getByRole('combobox', { name: 'Select Subset' })).toHaveValue('example')
  expect(getPredictionsMock).toHaveBeenLastCalledWith('/outputs', 'run/m', 'demo', 'example', expect.anything())
  expect(screen.getByRole('button', { name: 'Clear filters' })).toBeInTheDocument()
})
