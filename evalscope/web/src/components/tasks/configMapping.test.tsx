import { afterEach, describe, expect, it, vi } from 'vitest'
import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { LocaleProvider } from '@/contexts/LocaleContext'
import { EvalFormHarness as EvalConfigForm, PerfFormHarness as PerfConfigForm } from '@/test/TaskFormHarness'

vi.mock('@/api/eval', () => ({ listBenchmarks: vi.fn(() => new Promise(() => {})) }))

afterEach(() => { cleanup(); localStorage.clear() })

function change(id: string, value: string) {
  const element = document.getElementById(id)
  if (!element) throw new Error(`Missing field ${id}`)
  fireEvent.change(element, { target: { value } })
}

function evalForm() {
  const submit = vi.fn()
  render(<LocaleProvider><EvalConfigForm onSubmit={submit} /></LocaleProvider>)
  change('connection-model', ' same-model ')
  change('eval-datasets', ' general_qa, gsm8k ')
  change('connection-url', ' http://localhost:18101/v1 ')
  fireEvent.click(screen.getByRole('button', { name: 'More Parameters' }))
  return submit
}

function perfForm() {
  const submit = vi.fn()
  render(<LocaleProvider><PerfConfigForm onSubmit={submit} /></LocaleProvider>)
  change('connection-model', ' same-model ')
  change('connection-url', ' http://localhost:18101/v1 ')
  return submit
}

function submitEval() { fireEvent.click(screen.getByRole('button', { name: 'Start Evaluation' })) }
function submitPerf() { fireEvent.click(screen.getByRole('button', { name: 'Start Performance Test' })) }

describe('Evaluation form → TaskConfig payload', () => {
  it('maps every displayed field, preserves false and uses generation_config', () => {
    const submit = evalForm()
    change('connection-apiKey', 'fake-key')
    change('connection-alias', ' server-A ')
    change('eval-limit', '.25')
    change('eval-batchSize', '2')
    change('eval-repeats', '3')
    change('eval-timeout', '2.5')
    change('eval-temperature', '.85')
    change('eval-topP', '.95')
    change('eval-maxTokens', '32')
    change('eval-topK', '-1')
    change('eval-datasetArgs', '{"general_qa":{"local_path":"/tmp/qa.jsonl"}}')
    submitEval()
    expect(submit).toHaveBeenCalledWith({
      model: 'same-model', datasets: ['general_qa', 'gsm8k'], api_url: 'http://localhost:18101/v1',
      api_key: 'fake-key', alias: 'server-A', limit: .25, eval_batch_size: 2, repeats: 3,
      generation_config: { stream: false, timeout: 2.5, temperature: .85, top_p: .95, max_tokens: 32, top_k: -1 },
      dataset_args: { general_qa: { local_path: '/tmp/qa.jsonl' } },
    })
    expect(submit.mock.calls[0][0]).not.toHaveProperty('timeout')
    expect(submit.mock.calls[0][0]).not.toHaveProperty('stream')
  })
  it('maps both states of the stream switch', () => {
    const submit = evalForm()
    const toggle = screen.getByRole('switch', { name: 'Stream Output' })
    expect(toggle).toHaveAttribute('aria-checked', 'false')
    fireEvent.click(toggle)
    expect(toggle).toHaveAttribute('aria-checked', 'true')
    submitEval()
    expect(submit.mock.calls[0][0].generation_config.stream).toBe(true)
    fireEvent.click(screen.getByLabelText('Stream Output'))
    submitEval()
    expect(submit.mock.calls[1][0].generation_config.stream).toBe(false)
  })
  it.each(['', '0', '.01', '.25', '1', '20'])('accepts the supported limit %j', (value) => {
    const submit = evalForm()
    change('eval-limit', value)
    submitEval()
    expect(submit).toHaveBeenCalledOnce()
    expect(submit.mock.calls[0][0].limit).toBe(value ? Number(value) : undefined)
  })
  it.each(['-1', '1.5', '2.5'])('rejects the unsupported limit %s', (value) => {
    const submit = evalForm()
    change('eval-limit', value)
    submitEval()
    expect(submit).not.toHaveBeenCalled()
    expect(document.getElementById('eval-limit')).toHaveAttribute('aria-invalid', 'true')
  })
  it.each(['eval-batchSize', 'eval-repeats', 'eval-maxTokens', 'eval-topK'])('rejects fractional integer field %s', (id) => {
    const submit = evalForm()
    change(id, '1.5')
    submitEval()
    expect(submit).not.toHaveBeenCalled()
    expect(document.getElementById(id)).toHaveAttribute('aria-invalid', 'true')
  })
  it('does not treat comma-only dataset input as a valid selection', () => {
    const submit = evalForm()
    change('eval-datasets', ', ,')
    submitEval()
    expect(submit).not.toHaveBeenCalled()
  })
  it('requires a positive timeout', () => {
    const submit = evalForm()
    change('eval-timeout', '0')
    submitEval()
    expect(submit).not.toHaveBeenCalled()
  })
  it.each(['{"general_qa":null}', '{"general_qa":[]}', '{"general_qa":2}'])('rejects non-object dataset configuration %s', (value) => {
    const submit = evalForm()
    change('eval-datasetArgs', value)
    submitEval()
    expect(submit).not.toHaveBeenCalled()
  })
  it('maps Docker configuration and omits it when disabled', () => {
    const submit = evalForm()
    fireEvent.click(screen.getByLabelText('Enable Sandbox'))
    change('eval-sandboxUrl', 'http://sandbox:8000')
    change('eval-sandboxImage', 'python:3.12')
    change('eval-sandboxPool', '2')
    submitEval()
    expect(submit.mock.calls[0][0].sandbox).toEqual({
      enabled: true, engine: 'docker', manager_config: { base_url: 'http://sandbox:8000' },
      default_config: { image: 'python:3.12' }, pool_size: 2,
    })
    fireEvent.click(screen.getByLabelText('Enable Sandbox'))
    submitEval()
    expect(submit.mock.calls[1][0]).not.toHaveProperty('sandbox')
  })
  it('does not forward Docker-only fields after selecting Volcengine', () => {
    const submit = evalForm()
    fireEvent.click(screen.getByLabelText('Enable Sandbox'))
    change('eval-sandboxUrl', 'http://sandbox:8000')
    change('eval-sandboxImage', 'python:3.12')
    change('eval-sandboxEngine', 'volcengine')
    expect(screen.queryByLabelText('Manager URL')).not.toBeInTheDocument()
    expect(screen.queryByLabelText('Docker Image')).not.toBeInTheDocument()
    submitEval()
    expect(submit.mock.calls[0][0].sandbox).toEqual({ enabled: true, engine: 'volcengine' })
  })
})

describe('Performance form → Arguments payload', () => {
  it('maps every remote field and trims optional text', () => {
    const submit = perfForm()
    change('connection-apiKey', 'fake-key')
    change('connection-alias', ' server-A ')
    change('perf-parallel', '1, 2')
    change('perf-number', '10, 20')
    change('perf-rate', '.5')
    change('perf-maxTokens', '32')
    change('perf-minTokens', '0')
    change('perf-dataset', ' openqa ')
    change('perf-maxPromptLen', '100')
    change('perf-minPromptLen', '0')
    fireEvent.click(screen.getByRole('button', { name: 'More Parameters' }))
    change('perf-tokenizerPath', ' /tmp/tokenizer ')
    submitPerf()
    expect(submit).toHaveBeenCalledWith({
      model: 'same-model', api: 'openai', url: 'http://localhost:18101/v1', api_key: 'fake-key', alias: 'server-A',
      parallel: [1, 2], number: [10, 20], rate: .5, max_tokens: 32, min_tokens: 0, dataset: 'openqa',
      max_prompt_length: 100, min_prompt_length: 0, tokenizer_path: '/tmp/tokenizer',
    })
  })
  it('offers only the protocols the form can execute and omits unused local connection fields', () => {
    const submit = perfForm()
    expect([...document.querySelectorAll('#connection-api option')].map((item) => (item as HTMLOptionElement).value)).toEqual(['openai', 'local'])
    change('connection-apiKey', 'must-not-leak')
    change('connection-api', 'local')
    expect(screen.queryByRole('textbox', { name: /Model API URL/ })).not.toBeInTheDocument()
    expect(screen.queryByLabelText('Model API Key')).not.toBeInTheDocument()
    submitPerf()
    expect(submit.mock.calls[0][0].api).toBe('local')
    expect(submit.mock.calls[0][0]).not.toHaveProperty('url')
    expect(submit.mock.calls[0][0]).not.toHaveProperty('api_key')
  })
  it.each(['', '-1', '.1', '2.5'])('accepts supported rate %j', (value) => {
    const submit = perfForm()
    change('perf-rate', value)
    submitPerf()
    expect(submit).toHaveBeenCalledOnce()
  })
  it.each(['0', '-2'])('rejects unsafe rate %j', (value) => {
    const submit = perfForm()
    change('perf-rate', value)
    submitPerf()
    expect(submit).not.toHaveBeenCalled()
  })
  it('rejects unequal sweep lengths', () => {
    const submit = perfForm()
    change('perf-parallel', '1, 2')
    submitPerf()
    expect(submit).not.toHaveBeenCalled()
  })
  it.each(['perf-maxTokens', 'perf-minTokens', 'perf-maxPromptLen', 'perf-minPromptLen'])('rejects fractional integer %s', (id) => {
    const submit = perfForm()
    change(id, '1.5')
    submitPerf()
    expect(submit).not.toHaveBeenCalled()
  })
  it('rejects inverted token and prompt-length ranges', () => {
    const submit = perfForm()
    change('perf-maxTokens', '8'); change('perf-minTokens', '10')
    submitPerf()
    expect(submit).not.toHaveBeenCalled()
    change('perf-minTokens', '0')
    change('perf-maxPromptLen', '10'); change('perf-minPromptLen', '20')
    submitPerf()
    expect(submit).not.toHaveBeenCalled()
  })
  it('checks minimums against backend defaults when maximums are blank', () => {
    const submit = perfForm()
    change('perf-maxTokens', '')
    change('perf-minTokens', '3000')
    submitPerf()
    expect(submit).not.toHaveBeenCalled()
    change('perf-minTokens', '0')
    change('perf-maxPromptLen', '')
    change('perf-minPromptLen', '200000')
    submitPerf()
    expect(submit).not.toHaveBeenCalled()
  })
  it('requires and reveals tokenizer_path for the random dataset', () => {
    const submit = perfForm()
    change('perf-dataset', 'random')
    submitPerf()
    expect(submit).not.toHaveBeenCalled()
    expect(screen.getByLabelText('Tokenizer Path')).toHaveAttribute('aria-invalid', 'true')
  })
})
