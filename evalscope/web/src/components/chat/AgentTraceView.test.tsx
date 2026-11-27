import { fireEvent, render as rtlRender, screen } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'

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

it('places a finding badge only on the step containing its recorded call ID', async () => {
  const { buildStepGroups } = await import('@/domain/trace/stepGroups')
  const { TracedTimeline } = await import('./AgentTraceView')
  const messages = [
    {role:'assistant', content:'first', tool_calls:[{id:'a',function:'read',arguments:{}}]},
    {role:'assistant', content:'second', tool_calls:[{id:'b',function:'write',arguments:{}}]},
  ] as ChatMessage[]
  const trace = {events:[
    event({step:0,type:'tool_call',payload:{id:'a',name:'read',arguments:{}}}),
    event({step:1,type:'tool_call',payload:{id:'b',name:'write',arguments:{}}}),
  ]} as import('@/api/types').AgentTrace
  const {container} = render(<TracedTimeline messages={messages} trace={trace}
    groups={buildStepGroups(messages,trace)} highlightStep={null} onStepClick={() => {}}
    findings={[{detector:'risk',category:'risky_command',subtype:'risk',attribution:'model',confidence:'low',call_ids:['b']},
      {detector:'repeat',category:'repetition',subtype:'repeat',attribution:'model',confidence:'low',call_ids:['absent']}]} />)
  const badge = screen.getByText('Potentially risky command')
  expect(badge.closest('button')).toHaveTextContent('Step 1')
  expect(container.textContent).not.toContain('Repetition')
})

it('shows a provider-confirmed tool error only on its matching result, including old message formats', async () => {
  const { buildStepGroups } = await import('@/domain/trace/stepGroups')
  const { TracedTimeline } = await import('./AgentTraceView')
  const messages = [
    { role: 'assistant', id: 'm0', content: '', tool_calls: [{ id: 'bad', function: 'create_event', arguments: {} }] },
    { role: 'tool', id: 'r1', tool_call_id: 'bad', content: 'ValueError: invalid date' },
    { role: 'assistant', id: 'm1', content: '', tool_calls: [{ id: 'ok', function: 'create_event', arguments: {} }] },
    { role: 'tool', id: 'r2', tool_call_id: 'ok', content: 'Event created' },
  ] as ChatMessage[]
  const trace = { events: [
    event({ step: 0, message_id: 'm0', type: 'tool_call', payload: { id: 'bad', name: 'create_event', arguments: {} } }),
    event({ step: 1, message_id: 'm1', type: 'tool_call', payload: { id: 'ok', name: 'create_event', arguments: {} } }),
  ] } as import('@/api/types').AgentTrace
  render(<TracedTimeline messages={messages} trace={trace}
    groups={buildStepGroups(messages, trace)} highlightStep={null} onStepClick={() => {}}
    findings={[{ detector: 'tool_failure', category: 'tool_failure', subtype: 'provider:runtime_error', attribution: 'model', confidence: 'medium', call_ids: ['bad'] }]} />)
  fireEvent.click(screen.getByRole('button', { name: /Step 0 create_event/ }))
  expect(screen.getByText('ValueError: invalid date')).toBeInTheDocument()
  fireEvent.click(screen.getByRole('button', { name: /Step 1 create_event/ }))
  fireEvent.click(screen.getAllByRole('button', { name: /^create_event/ })[1])
  expect(screen.getByText('Event created')).toBeInTheDocument()
  expect(screen.queryByText('Error: Event created')).not.toBeInTheDocument()
})


it('highlights tool evidence in text order even when citation order is reversed', async () => {
  const { EvidenceByMessage } = await import('@/contexts/EvidenceContext')
  const { ToolCallEntryRow } = await import('./ToolCallComponents')
  const { container } = render(<EvidenceByMessage.Provider value={{ result: ['later', 'earlier'] }}>
    <ToolCallEntryRow entry={{ id: 'call', function: 'read', arguments: {}, result: { role: 'tool', id: 'result', content: 'earlier then later' } as ChatMessage }} />
  </EvidenceByMessage.Provider>)
  expect(Array.from(container.querySelectorAll('mark'), node => node.textContent)).toEqual(['earlier', 'later'])
})

it('can reveal an agent step again after clearing selection and scrolling away', async () => {
  const { default: VirtualList } = await import('@/components/ui/VirtualList')
  const previousObserver = globalThis.ResizeObserver
  vi.stubGlobal('ResizeObserver', class { observe() {} disconnect() {} })
  const props = { items: Array.from({ length: 60 }, (_, i) => i), getKey: (i: number) => i,
    renderItem: (i: number) => <p>Step content {i}</p> }
  const view = render(<VirtualList {...props} revealIndex={40} />)
  expect(screen.getByText('Step content 40')).toBeInTheDocument()
  view.rerender(<LocaleProvider><VirtualList {...props} revealIndex={-1} /></LocaleProvider>)
  const scroller = view.container.querySelector('[style*="overflow"]')!
  fireEvent.scroll(scroller, { target: { scrollTop: 0 } })
  expect(screen.queryByText('Step content 40')).not.toBeInTheDocument()
  view.rerender(<LocaleProvider><VirtualList {...props} revealIndex={40} /></LocaleProvider>)
  expect(screen.getByText('Step content 40')).toBeInTheDocument()
  vi.stubGlobal('ResizeObserver', previousObserver)
})

it('reveals a searched result under its original call even when recorded on the following step', async () => {
  const { buildStepGroups } = await import('@/domain/trace/stepGroups')
  const { TracedTimeline } = await import('./AgentTraceView')
  const messages = [
    { role: 'assistant', id: 'm0', content: '', tool_calls: [{ id: 'call', function: 'read', arguments: {} }] },
    { role: 'tool', id: 'result-0', tool_call_id: 'call', content: 'Complete searched result' },
    { role: 'assistant', id: 'm1', content: 'Finished.' },
  ] as ChatMessage[]
  const trace = { events: [
    event({ step: 0, message_id: 'm0', type: 'tool_call', payload: { id: 'call', name: 'read', arguments: {} } }),
    event({ step: 1, message_id: 'result-0', type: 'tool_result', payload: { id: 'call', name: 'read' } }),
    event({ step: 1, message_id: 'm1', type: 'model_generate' }),
  ] } as import('@/api/types').AgentTrace
  render(<TracedTimeline groups={buildStepGroups(messages, trace)} messages={messages} trace={trace}
    highlightStep={null} highlightId="result-0" onStepClick={() => {}} />)
  expect(screen.getByRole('button', { name: 'Step 0 read' })).toHaveAttribute('aria-expanded', 'true')
  expect(screen.getByText('Complete searched result')).toBeInTheDocument()
})
