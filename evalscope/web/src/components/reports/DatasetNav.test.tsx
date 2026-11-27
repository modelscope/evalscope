import { cleanup, fireEvent, render, screen, within } from '@testing-library/react'
import { afterEach, expect, it, vi } from 'vitest'
import type { LoadReportResponse } from '@/api/types'
import { LocaleProvider } from '@/contexts/LocaleContext'
import { loadFixture } from '@/test/loadFixture'
import DatasetNav from './DatasetNav'

afterEach(cleanup)
const reports = loadFixture<LoadReportResponse>('report-multi-dataset').report_list

it('keeps each dataset metric scale and unit, and switches the selected dataset', () => {
  const onChange = vi.fn()
  render(
    <LocaleProvider>
      <DatasetNav reports={reports} active="gsm8k" onChange={onChange} />
    </LocaleProvider>,
  )
  const navigation = screen.getByRole('navigation', { name: 'Datasets' })
  expect(within(navigation).getByText('Accuracy ↑')).toBeInTheDocument()
  expect(within(navigation).getByText('81.5%')).toBeInTheDocument()
  expect(within(navigation).getByText('512 tok/s')).toBeInTheDocument()
  expect(within(navigation).queryByText('8150%')).not.toBeInTheDocument()
  expect(within(navigation).getByRole('button', { name: /gsm8k/, pressed: true })).toBeInTheDocument()
  fireEvent.click(within(navigation).getByRole('button', { name: /arc_challenge/, pressed: false }))
  expect(onChange).toHaveBeenCalledWith('arc_challenge')
  expect(screen.queryByText('Average Score')).not.toBeInTheDocument()
})

it('keeps low-is-better and unavailable primary scores honest', () => {
  const primary = reports[0].metrics[0]
  const speech = {
    ...reports[0],
    dataset_name: 'speech',
    metrics: [
      {
        ...primary,
        score: 0.07,
        semantics: { ...primary.semantics, metric_name: 'WER', direction: 'lower_is_better' as const },
      },
    ],
  }
  const unavailable = { ...reports[0], dataset_name: 'unavailable', primary_metric_identity: null }
  render(
    <LocaleProvider>
      <DatasetNav reports={[speech, unavailable]} active="speech" onChange={vi.fn()} />
    </LocaleProvider>,
  )
  expect(screen.getByText('WER ↓')).toBeInTheDocument()
  expect(screen.getByText('7%')).toBeInTheDocument()
  expect(screen.getByText('—')).toBeInTheDocument()
})

it('includes an unscored sample in the dataset count', () => {
  const report = { ...reports[0], num: 79, execution_summary: { requested: 80 } }
  render(<LocaleProvider><DatasetNav reports={[report]} active={report.dataset_name} onChange={vi.fn()} /></LocaleProvider>)
  expect(screen.getByRole('button')).toHaveTextContent('80 samples')
})
