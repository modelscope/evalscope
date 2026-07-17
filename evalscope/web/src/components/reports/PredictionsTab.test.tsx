import { act, cleanup, render, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { afterEach, describe, expect, it, vi } from 'vitest'

import type { PredictionsResponse } from '@/api/types'
import { DomainError } from '@/api/errors'
import { LocaleProvider } from '@/contexts/LocaleContext'
import { loadFixture } from '@/test/loadFixture'
import PredictionsTab from './PredictionsTab'

const { getDataFrameMock, getPredictionsMock } = vi.hoisted(() => ({
  getDataFrameMock: vi.fn(),
  getPredictionsMock: vi.fn(),
}))

vi.mock('@/api/reports', () => ({
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
