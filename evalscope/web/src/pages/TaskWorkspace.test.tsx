import { afterEach, describe, expect, it, vi } from 'vitest'
import { act, cleanup, fireEvent, render, screen, within } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { LocaleProvider } from '@/contexts/LocaleContext'
import {
  CONFIG_STORAGE_KEY,
  emptyWorkspace,
  exportLibrary,
  readWorkspace,
  writeWorkspace,
} from '@/components/tasks/configStorage'
import TasksPage from './TasksPage'

vi.mock('@/api/eval', () => ({
  listBenchmarks: vi.fn(() => new Promise(() => {})),
  submitEvalTask: vi.fn(() => new Promise(() => {})),
  stopEvalTask: vi.fn(),
  getEvalLog: vi.fn(() => new Promise(() => {})),
}))
vi.mock('@/api/perf', () => ({
  submitPerfTask: vi.fn(() => new Promise(() => {})),
  stopPerfTask: vi.fn(),
  getPerfLog: vi.fn(() => new Promise(() => {})),
}))
vi.mock('@/api/tasks', () => ({
  listActiveTasks: vi.fn(() => new Promise(() => {})),
  getTask: vi.fn(() => new Promise(() => {})),
}))

import { submitEvalTask } from '@/api/eval'
import { submitPerfTask } from '@/api/perf'

afterEach(async () => {
  await act(async () => {
    await Promise.resolve()
  })
  cleanup()
  localStorage.clear()
  sessionStorage.clear()
  vi.clearAllMocks()
  vi.restoreAllMocks()
})

function mount(search = '') {
  return render(
    <LocaleProvider>
      <MemoryRouter initialEntries={[`/tasks${search}`]}>
        <TasksPage />
      </MemoryRouter>
    </LocaleProvider>,
  )
}
function change(id: string, value: string) {
  fireEvent.change(document.getElementById(id)!, { target: { value } })
}
function tab(kind: 'eval' | 'perf') {
  fireEvent.click(screen.getByRole('tab', { name: kind === 'eval' ? 'Evaluation' : 'Performance' }))
}
function fillConnection() {
  change('connection-model', 'same-model')
  change('connection-url', 'http://localhost:18101/v1')
  change('connection-apiKey', 'fake-key-for-tests')
}
function save(name: string, group: string) {
  const controls = within(screen.getByRole('group', { name: group }))
  fireEvent.change(controls.getByRole('textbox'), { target: { value: name } })
  fireEvent.click(controls.getByRole('button', { name: 'Save' }))
}
async function flushDraft() {
  await act(async () => {
    vi.advanceTimersByTime(400)
    await Promise.resolve()
  })
}

describe('Task workspace', () => {
  it('submits evaluation and performance with one shared connection', () => {
    mount()
    fillConnection()
    change('eval-datasets', 'gsm8k')
    fireEvent.click(screen.getByRole('button', { name: 'Start Evaluation' }))
    expect(submitEvalTask).toHaveBeenCalledWith(
      expect.objectContaining({
        model: 'same-model',
        api_url: 'http://localhost:18101/v1',
        api_key: 'fake-key-for-tests',
      }),
      expect.any(String),
    )
    tab('perf')
    change('perf-parallel', '1,2')
    change('perf-number', '4,8')
    fireEvent.click(screen.getByRole('button', { name: 'Start Performance Test' }))
    expect(submitPerfTask).toHaveBeenCalledWith(
      expect.objectContaining({
        model: 'same-model',
        url: 'http://localhost:18101/v1',
        api_key: 'fake-key-for-tests',
        parallel: [1, 2],
        number: [4, 8],
      }),
      expect.any(String),
    )
    expect(screen.getAllByRole('textbox', { name: /Model Name/ })).toHaveLength(1)
  })
  it('preserves independent task drafts and restores them after reload', async () => {
    const view = mount()
    fillConnection()
    change('eval-datasets', 'gsm8k')
    change('eval-limit', '.25')
    tab('perf')
    change('perf-number', '16')
    tab('eval')
    expect(document.getElementById('eval-limit')).toHaveValue(0.25)
    await flushDraft()
    view.unmount()
    mount()
    tab('perf')
    expect(document.getElementById('perf-number')).toHaveValue('16')
    expect(document.getElementById('connection-apiKey')).toHaveValue('')
    expect(document.getElementById('connection-model')).toHaveValue('same-model')
  })
  it('saves credentials only after opting in and confirming, then loads them', () => {
    const view = mount()
    fillConnection()
    fireEvent.click(screen.getByRole('switch', { name: 'Save API Key on this browser' }))
    save('Server A', 'Model connection')
    expect(screen.getByRole('alertdialog')).toHaveTextContent('plaintext')
    expect(localStorage.getItem(CONFIG_STORAGE_KEY)).toBeNull()
    fireEvent.click(screen.getByRole('button', { name: 'Confirm' }))
    expect(localStorage.getItem(CONFIG_STORAGE_KEY)).toContain('fake-key-for-tests')
    view.unmount()
    mount()
    expect(document.getElementById('connection-apiKey')).toHaveValue('fake-key-for-tests')
  })
  it('clears credentials on service changes without changing a saved profile', () => {
    mount()
    fillConnection()
    change('connection-url', 'http://localhost:18102/v1')
    expect(document.getElementById('connection-apiKey')).toHaveValue('')
    expect(screen.getByText(/previous API Key was cleared/)).toBeInTheDocument()
  })
  it('saves and loads templates without changing the model connection', () => {
    mount()
    fillConnection()
    change('eval-datasets', 'gsm8k')
    change('eval-limit', '.25')
    save('QA', 'Evaluation template')
    const selected = readWorkspace()
    if (!selected.ok) throw new Error('Expected saved workspace')
    const id = selected.value.selectedEval
    change('eval-limit', '5')
    const controls = within(screen.getByRole('group', { name: 'Evaluation template' }))
    expect(controls.getByRole('combobox')).toHaveValue(id)
    fireEvent.click(controls.getByRole('button', { name: 'Load' }))
    expect(screen.getByRole('alertdialog')).toHaveTextContent('current draft will be replaced')
    fireEvent.click(screen.getByRole('button', { name: 'Cancel' }))
    expect(document.getElementById('eval-limit')).toHaveValue(5)
    fireEvent.click(controls.getByRole('button', { name: 'Load' }))
    fireEvent.click(screen.getByRole('button', { name: 'Confirm' }))
    expect(document.getElementById('eval-limit')).toHaveValue(0.25)
    expect(document.getElementById('connection-apiKey')).toHaveValue('fake-key-for-tests')
  })
  it('keeps Local performance settings from being used for Evaluation', () => {
    mount()
    fillConnection()
    change('connection-api', 'local')
    expect(screen.getByRole('button', { name: 'Start Evaluation' })).toBeDisabled()
    expect(screen.queryByRole('textbox', { name: /API URL/ })).not.toBeInTheDocument()
    tab('perf')
    fireEvent.click(screen.getByRole('button', { name: 'Start Performance Test' }))
    expect(submitPerfTask).toHaveBeenCalledWith(expect.objectContaining({ api: 'local' }), expect.any(String))
    expect(vi.mocked(submitPerfTask).mock.calls[0][0]).not.toHaveProperty('api_key')
  })
  it('uses accessible help icons that do not submit or toggle fields', () => {
    mount()
    fillConnection()
    const help = screen.getByRole('button', { name: 'Alias (optional) help' })
    fireEvent.mouseEnter(help)
    expect(screen.getByRole('tooltip')).toHaveTextContent('Optional run name')
    fireEvent.keyDown(help, { key: 'Escape' })
    expect(screen.queryByRole('tooltip')).not.toBeInTheDocument()
    fireEvent.click(help)
    expect(screen.getByRole('tooltip')).toBeInTheDocument()
    expect(submitEvalTask).not.toHaveBeenCalled()
    expect(document.getElementById('connection-alias')).toHaveValue('')
  })
  it('does not overwrite damaged stored data during editing', async () => {
    localStorage.setItem(CONFIG_STORAGE_KEY, '{"version":99}')
    mount()
    change('connection-model', 'new-model')
    await flushDraft()
    expect(localStorage.getItem(CONFIG_STORAGE_KEY)).toBe('{"version":99}')
    expect(screen.getByText(/Stored configuration is damaged/)).toHaveAttribute('role', 'alert')
  })
  it('reports a write failure instead of claiming a configuration was saved', () => {
    mount()
    fillConnection()
    vi.spyOn(Storage.prototype, 'setItem').mockImplementation(() => {
      throw new Error('Quota')
    })
    save('Server', 'Model connection')
    expect(screen.getByText(/Could not save to browser storage/)).toHaveAttribute('role', 'alert')
    expect(screen.queryByText('Configuration saved on this browser.')).not.toBeInTheDocument()
  })
  it('clears local keys only after explicit confirmation', () => {
    const state = emptyWorkspace()
    state.connection = { ...state.connection, model: 'm', url: 'http://localhost/v1', apiKey: 'fake-saved' }
    state.connections = [{ id: 'a', name: 'A', value: { ...state.connection }, rememberKey: true }]
    state.selectedConnection = 'a'
    writeWorkspace(state)
    mount()
    fireEvent.click(screen.getByRole('button', { name: 'Remove saved API Key' }))
    fireEvent.click(screen.getByRole('button', { name: 'Cancel' }))
    expect(document.getElementById('connection-apiKey')).toHaveValue('fake-saved')
    fireEvent.click(screen.getByRole('button', { name: 'Remove saved API Key' }))
    fireEvent.click(screen.getByRole('button', { name: 'Confirm' }))
    expect(document.getElementById('connection-apiKey')).toHaveValue('')
    expect(localStorage.getItem(CONFIG_STORAGE_KEY)).not.toContain('fake-saved')
  })
  it('previews imports without exposing their keys and adds only after confirmation', async () => {
    mount()
    const state = emptyWorkspace()
    state.connections = [
      {
        id: 'a',
        name: 'Imported',
        value: { ...state.connection, model: 'm', url: 'http://localhost/v1', apiKey: 'fake-imported-key' },
        rememberKey: true,
      },
    ]
    const file = { size: 500, text: () => Promise.resolve(exportLibrary(state, true)) }
    await act(async () => {
      fireEvent.change(screen.getByLabelText('Import', { selector: 'input' }), { target: { files: [file] } })
      await Promise.resolve()
    })
    expect(screen.getByRole('alertdialog')).toHaveTextContent('1 model connections')
    expect(screen.getByRole('alertdialog')).not.toHaveTextContent('fake-imported-key')
    fireEvent.click(screen.getByRole('button', { name: 'Confirm' }))
    expect(readWorkspace()).toMatchObject({ ok: true, value: { connections: [{ name: 'Imported' }] } })
  })
})
