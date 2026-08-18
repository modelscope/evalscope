import { cleanup, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'
import { LocaleProvider } from '@/contexts/LocaleContext'
import { ThemeProvider } from '@/contexts/ThemeContext'
import { EvalResultPanel } from './EvalResultPanel'

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
  it('uses one neutral label hierarchy and a restrained success score', () => {
    renderPanel(1)

    expect(screen.getByText('Extracted Answer')).toHaveClass('type-label-xs')
    expect(screen.getByText('Expected Answer')).toHaveClass('type-label-xs')
    expect(screen.getByText('100%')).toHaveStyle({ color: 'var(--success)' })
  })

  it('keeps verbose details collapsed and uses danger styling below the filter', () => {
    renderPanel(0.2)

    expect(screen.getByRole('button', { name: 'Score Detail' })).toHaveAttribute('aria-expanded', 'false')
    expect(screen.queryByText('"acc"')).not.toBeInTheDocument()
    expect(screen.getByText('20%')).toHaveStyle({ color: 'var(--danger)' })
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
