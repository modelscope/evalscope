import { cleanup, render, screen } from '@testing-library/react'
import { afterEach, expect, it } from 'vitest'
import { MemoryRouter } from 'react-router-dom'
import { LocaleProvider } from '@/contexts/LocaleContext'
import ReportHeader from './ReportHeader'

afterEach(cleanup)

it('keeps the model and task totals without duplicate benchmark links', () => {
  render(
    <MemoryRouter>
      <LocaleProvider>
        <ReportHeader
          modelName="m"
          datasetCount={3}
          totalSamples={160}
          htmlReportUrl="/report.html"
          reportsUrl="/reports"
        />
      </LocaleProvider>
    </MemoryRouter>,
  )
  expect(screen.getByRole('heading', { level: 1, name: 'm' })).toBeInTheDocument()
  expect(screen.getByText('3 datasets · 160 samples')).toBeInTheDocument()
  expect(screen.getByRole('button', { name: 'View HTML Report' })).toBeInTheDocument()
  expect(screen.queryByRole('button', { pressed: true })).not.toBeInTheDocument()
  expect(screen.queryByText('Average Score')).not.toBeInTheDocument()
})
