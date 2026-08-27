import { render as rtlRender, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'

import { NudgeRow, TraceEventPill } from './AgentTraceView'
import { LocaleProvider } from '@/contexts/LocaleContext'
import type { AgentTraceEvent, ChatMessage } from '@/api/types'

function render(ui: React.ReactElement) {
  return rtlRender(<LocaleProvider>{ui}</LocaleProvider>)
}

function nudgeMessage(text: string): ChatMessage {
  return { id: 'm1', role: 'user', content: text } as ChatMessage
}

function event(partial: Partial<AgentTraceEvent>): AgentTraceEvent {
  return { step: 0, timestamp: 0, type: 'submit', payload: {}, ...partial } as AgentTraceEvent
}

describe('NudgeRow', () => {
  it('labels a malformed-output nudge', () => {
    render(<NudgeRow msg={nudgeMessage('Call at most 1 tool call per turn.')} outcome="malformed" />)
    expect(screen.getByText('malformed output')).toBeInTheDocument()
    expect(screen.getByText('Call at most 1 tool call per turn.')).toBeInTheDocument()
  })

  it('labels an idle nudge', () => {
    render(<NudgeRow msg={nudgeMessage('No tool was called.')} outcome="idle" />)
    expect(screen.getByText('no tool call')).toBeInTheDocument()
  })

  it('renders without a badge when the outcome is absent', () => {
    render(<NudgeRow msg={nudgeMessage('reminder')} />)
    expect(screen.queryByText('no tool call')).not.toBeInTheDocument()
    expect(screen.queryByText('malformed output')).not.toBeInTheDocument()
  })

  it('falls back to the raw outcome when untranslated', () => {
    render(<NudgeRow msg={nudgeMessage('reminder')} outcome="brand_new" />)
    expect(screen.getByText('brand_new')).toBeInTheDocument()
  })
})

describe('TraceEventPill submit source', () => {
  it('distinguishes a malformed-output exit from a no-tool-call exit', () => {
    const malformed = render(
      <TraceEventPill event={event({ payload: { source: 'parse_error_exhausted' } })} />
    )
    expect(malformed.container.textContent).toContain('malformed output')
    malformed.unmount()

    const idle = render(<TraceEventPill event={event({ payload: { source: 'implicit_no_nudge' } })} />)
    expect(idle.container.textContent).toContain('no tool call')
  })

  it('falls back to the raw source when untranslated', () => {
    const { container } = render(<TraceEventPill event={event({ payload: { source: 'brand_new' } })} />)
    expect(container.textContent).toContain('brand_new')
  })

  it('omits the suffix when a submit carries no source', () => {
    const { container } = render(<TraceEventPill event={event({ payload: { final_answer: '42' } })} />)
    expect(container.textContent).not.toContain('·')
  })

  it('does not add a source suffix to non-submit events', () => {
    const { container } = render(
      <TraceEventPill event={event({ type: 'nudge', payload: { source: 'nudge' } })} />
    )
    expect(container.textContent).not.toContain('·')
  })
})
