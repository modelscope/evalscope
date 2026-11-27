import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { LocaleProvider } from '@/contexts/LocaleContext'
import { listPlugins } from '@/api/plugins'
import { EvalFormHarness, PerfFormHarness } from '@/test/TaskFormHarness'
import { pluginApiOptions } from './taskConfig'

vi.mock('@/api/eval', () => ({ listBenchmarks: vi.fn(() => new Promise(() => {})) }))
vi.mock('@/api/plugins', () => ({ listPlugins: vi.fn() }))

const ENABLED = {
  plugins: [
    {
      name: 'example',
      module: 'evalscope_example_plugin',
      distribution: 'evalscope-example-plugin',
      version: '0.1.0',
      registrations: { model_api: ['example_arith_api'], perf_api: ['example_arith'], benchmark: ['example_arith'] },
    },
  ],
}

beforeEach(() => {
  // The shared setup installs fake timers; the async plugin read needs real ones.
  vi.useRealTimers()
  vi.mocked(listPlugins).mockResolvedValue(ENABLED)
})

afterEach(() => {
  cleanup()
  localStorage.clear()
  vi.mocked(listPlugins).mockReset()
})

function change(id: string, value: string) {
  const element = document.getElementById(id)
  if (!element) throw new Error(`Missing field ${id}`)
  fireEvent.change(element, { target: { value } })
}

describe('pluginApiOptions', () => {
  it('lists one registry kind with the plugin that registered each name', () => {
    expect(pluginApiOptions(ENABLED.plugins, 'model_api')).toEqual([{ value: 'example_arith_api', plugin: 'example' }])
    expect(pluginApiOptions(ENABLED.plugins, 'perf_api')).toEqual([{ value: 'example_arith', plugin: 'example' }])
    expect(pluginApiOptions([], 'perf_api')).toEqual([])
  })
})

describe('Plugin API field', () => {
  it('sends the selected plugin model API as eval_type', async () => {
    const submit = vi.fn()
    render(<LocaleProvider><EvalFormHarness onSubmit={submit} /></LocaleProvider>)
    const select = await screen.findByLabelText('Plugin model API')
    expect([...select.querySelectorAll('option')].map((option) => option.value)).toEqual(['', 'example_arith_api'])
    change('connection-model', 'arith-demo')
    change('connection-url', 'http://127.0.0.1:8901/v1')
    change('eval-datasets', 'example_arith')
    change('eval-pluginEvalType', 'example_arith_api')
    fireEvent.click(screen.getByRole('button', { name: 'Start Evaluation' }))
    expect(submit).toHaveBeenCalledWith(expect.objectContaining({ eval_type: 'example_arith_api', datasets: ['example_arith'] }))
  })

  it('sends the selected plugin protocol as the perf api', async () => {
    const submit = vi.fn()
    render(<LocaleProvider><PerfFormHarness onSubmit={submit} /></LocaleProvider>)
    await screen.findByLabelText('Plugin API protocol')
    change('connection-model', 'arith-demo')
    change('connection-url', 'http://127.0.0.1:8901/v1/arith')
    change('perf-pluginApi', 'example_arith')
    fireEvent.click(screen.getByRole('button', { name: 'Start Performance Test' }))
    expect(submit).toHaveBeenCalledWith(expect.objectContaining({ api: 'example_arith', model: 'arith-demo' }))
  })

  it('stays hidden and keeps the connection protocol when no plugin is enabled', async () => {
    vi.mocked(listPlugins).mockResolvedValue({ plugins: [] })
    const submit = vi.fn()
    render(<LocaleProvider><PerfFormHarness onSubmit={submit} /></LocaleProvider>)
    await vi.waitFor(() => expect(listPlugins).toHaveBeenCalled())
    expect(screen.queryByLabelText('Plugin API protocol')).not.toBeInTheDocument()
    change('connection-model', 'same-model')
    change('connection-url', 'http://localhost:18101/v1')
    fireEvent.click(screen.getByRole('button', { name: 'Start Performance Test' }))
    expect(submit.mock.calls[0][0].api).toBe('openai')
  })
})
