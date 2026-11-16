import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, expect, it, vi } from 'vitest'
import { MemoryRouter } from 'react-router-dom'
import { LocaleProvider } from '@/contexts/LocaleContext'
import ReportHeader from './ReportHeader'

afterEach(cleanup)

it('marks the active diagnostic benchmark and retains dataset switching', () => {
  const onDatasetClick = vi.fn()
  const view = (selectedDataset: string) => (
    <MemoryRouter>
      <LocaleProvider>
        <ReportHeader
          modelName="m"
          datasetName="AgentHarm"
          datasets={['agentharm', 'agentharm_benign']}
          datasetLabels={{ agentharm: 'AgentHarm', agentharm_benign: 'AgentHarm Benign' }}
          selectedDataset={selectedDataset}
          score={null}
          totalSamples={160}
          htmlReportUrl="/report.html"
          onDatasetClick={onDatasetClick}
        />
      </LocaleProvider>
    </MemoryRouter>
  )
  const { rerender } = render(view('agentharm'))
  expect(screen.getByRole('button', { name: 'AgentHarm', pressed: true })).toHaveClass('underline')
  fireEvent.click(screen.getByRole('button', { name: 'AgentHarm Benign', pressed: false }))
  expect(onDatasetClick).toHaveBeenCalledWith('agentharm_benign')
  rerender(view('agentharm_benign'))
  expect(screen.getByRole('button', { name: 'AgentHarm Benign', pressed: true })).toHaveClass('underline')
  expect(screen.getByRole('button', { name: 'AgentHarm', pressed: false })).toBeInTheDocument()
})
