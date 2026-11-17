import { cleanup, render, screen } from '@testing-library/react'
import { afterEach, expect, it, vi } from 'vitest'
import { LocaleProvider } from '@/contexts/LocaleContext'
import MarkdownRenderer from './MarkdownRenderer'

vi.mock('rehype-katex', () => { throw new Error('A previous math bundle is no longer available') })
afterEach(cleanup)

it('keeps the original answer readable and offers page recovery after a math resource fails', async () => {
  vi.useRealTimers()
  render(<LocaleProvider><MarkdownRenderer content="Answer: $x^2$" /></LocaleProvider>)
  expect(await screen.findByRole('status')).toHaveTextContent('Formulas could not be loaded')
  expect(screen.getByText(/Answer:/)).toBeInTheDocument()
  expect(screen.getByRole('button', { name: 'Reload page' })).toBeInTheDocument()
})
