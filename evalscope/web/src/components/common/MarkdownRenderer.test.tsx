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

describe('MarkdownRenderer image sources', () => {
  it('proxies a local absolute image path through the media file endpoint', () => {
    // The markdown chain (e.g. the `Input` column built by `messages_to_markdown`)
    // must resolve server-side paths the same way structured messages do.
    const { container } = renderMarkdown('![shot](/tmp/miniwob/step-000.png)')

    expect(container.querySelector('img')).toHaveAttribute(
      'src',
      '/api/v1/reports/media/file?path=%2Ftmp%2Fminiwob%2Fstep-000.png',
    )
  })

  it('decodes a percent-encoded non-ascii path back to the real filesystem path', () => {
    // The markdown parser encodes non-ascii destinations, so without decoding
    // the proxy would receive the literal escape sequences and return 404.
    const { container } = renderMarkdown('![shot](</tmp/\u6570\u636e\u96c6/a.png>)')

    expect(container.querySelector('img')).toHaveAttribute(
      'src',
      `/api/v1/reports/media/file?path=${encodeURIComponent('/tmp/\u6570\u636e\u96c6/a.png')}`,
    )
  })

  it('renders a local path containing spaces via the angle-bracket destination', () => {
    const { container } = renderMarkdown('![shot](</tmp/my dir/a.png>)')

    expect(container.querySelector('img')).toHaveAttribute(
      'src',
      `/api/v1/reports/media/file?path=${encodeURIComponent('/tmp/my dir/a.png')}`,
    )
  })

  it('keeps http and data URI image sources untouched', () => {
    const { container } = renderMarkdown(
      '![remote](https://example.com/a%20b.png)\n\n![inline](data:image/png;base64,aGVsbG8=)',
    )

    const images = container.querySelectorAll('img')
    expect(images[0]).toHaveAttribute('src', 'https://example.com/a%20b.png')
    expect(images[1]).toHaveAttribute('src', 'data:image/png;base64,aGVsbG8=')
  })

  it('keeps an uppercase scheme and a protocol-relative url untouched', () => {
    // These are valid remote sources; treating them as local paths or base64
    // payloads would break images that rendered fine before.
    const { container } = renderMarkdown('![up](HTTPS://example.com/a.png)\n\n![rel](//cdn.example.com/a.png)')

    const images = container.querySelectorAll('img')
    expect(images[0]).toHaveAttribute('src', 'HTTPS://example.com/a.png')
    expect(images[1]).toHaveAttribute('src', '//cdn.example.com/a.png')
  })
})
