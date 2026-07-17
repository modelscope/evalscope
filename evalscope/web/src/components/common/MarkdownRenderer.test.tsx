import { cleanup, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'

import { LocaleProvider } from '@/contexts/LocaleContext'
import { ThemeProvider } from '@/contexts/ThemeContext'
import MarkdownRenderer from './MarkdownRenderer'

afterEach(cleanup)

function renderMarkdown(content: string, collapsed = false) {
  return render(
    <LocaleProvider>
      <ThemeProvider>
        <MarkdownRenderer content={content} collapsed={collapsed} />
      </ThemeProvider>
    </LocaleProvider>,
  )
}

describe('MarkdownRenderer heavy-content gating', () => {
  it('keeps ordinary markdown visible while a collapsed fenced block is not mounted', () => {
    renderMarkdown('Visible paragraph\n\n```python\nprint("heavy fixture")\n```', true)

    expect(screen.getByText('Visible paragraph')).toBeInTheDocument()
    expect(screen.getByRole('status')).toHaveTextContent('Content collapsed')
    expect(screen.queryByText(/heavy fixture/)).not.toBeInTheDocument()
  })

  it('renders ordinary markdown without loading a heavy-content placeholder', () => {
    renderMarkdown('A **deterministic** paragraph.')

    expect(screen.getByText('deterministic')).toBeInTheDocument()
    expect(screen.queryByRole('status')).not.toBeInTheDocument()
  })
})
