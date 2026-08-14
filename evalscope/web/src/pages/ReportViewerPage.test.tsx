// Report Viewer page — the only surface that embeds an external report URL.
//
// The point of interest is the URL guard: `?url=` is placed straight into an
// <iframe src>, so anything but a same-origin report path must be refused. These
// pin the three branches (missing / rejected / accepted) so the guard cannot be
// loosened without a test noticing.

import { afterEach, describe, expect, it } from 'vitest'
import { cleanup, render, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'

import { LocaleProvider } from '@/contexts/LocaleContext'
import ReportViewerPage from './ReportViewerPage'

afterEach(cleanup)

function renderAt(search: string) {
  render(
    <LocaleProvider>
      <MemoryRouter initialEntries={[`/viewer${search}`]}>
        <ReportViewerPage />
      </MemoryRouter>
    </LocaleProvider>,
  )
}

describe('ReportViewerPage', () => {
  it('asks for a URL when none is given', () => {
    renderAt('')
    expect(screen.getByText(/No report URL specified/i)).toBeInTheDocument()
    expect(document.querySelector('iframe')).toBeNull()
  })

  it('refuses a cross-origin or non-report URL', () => {
    renderAt('?url=https://evil.example.com/steal')
    expect(screen.getByText(/Only same-origin report paths are allowed/i)).toBeInTheDocument()
    expect(document.querySelector('iframe')).toBeNull()
  })

  it('refuses a javascript: URL', () => {
    renderAt('?url=javascript:alert(1)')
    expect(screen.getByText(/Only same-origin report paths are allowed/i)).toBeInTheDocument()
  })

  it('embeds a same-origin report path in a sandboxed iframe', () => {
    renderAt('?url=/api/v1/reports/run/model/report.html')
    const iframe = document.querySelector('iframe')
    expect(iframe).not.toBeNull()
    expect(iframe!.getAttribute('src')).toBe('/api/v1/reports/run/model/report.html')
    // The sandbox attribute is what keeps an embedded report from acting as the app.
    expect(iframe!.getAttribute('sandbox')).toBe('allow-scripts allow-same-origin')
  })
})
