import { useCallback, useState } from 'react'
import { useLocale } from '@/contexts/LocaleContext'
import { useQueryParams } from '@/hooks/useQueryParams'
import Tabs from '@/components/ui/Tabs'
import ActiveTasksPanel from '@/components/tasks/ActiveTasksPanel'
import EvalTaskPanel from '@/components/tasks/EvalTaskPanel'
import PerfTaskPanel from '@/components/tasks/PerfTaskPanel'
import ModelConnectionEditor from '@/components/tasks/ModelConnectionEditor'
import PresetToolbar from '@/components/tasks/PresetToolbar'
import ConfigLibraryTools from '@/components/tasks/ConfigLibraryTools'
import { useTaskWorkspace } from '@/components/tasks/useTaskWorkspace'
import { createPresetId, mergeLibrary, readWorkspace, savePreset } from '@/components/tasks/configStorage'
import {
  DEFAULT_CONNECTION,
  DEFAULT_EVAL,
  DEFAULT_PERF,
  displayEndpoint,
  editConnection,
  validateConnection,
  validateEvalDraft,
  validatePerfDraft,
  type FieldErrors,
  type ModelConnection,
} from '@/components/tasks/taskConfig'

/**
 * Unified "run a task" page — hosts the Evaluation and Performance task
 * runners as sub-tabs. The active sub-tab is reflected in the `?tab=` query
 * param so it survives refresh / sharing.
 */
export default function TasksPage() {
  const { t } = useLocale()
  const { get, set } = useQueryParams()
  const tab = get('tab') === 'perf' ? 'perf' : 'eval'
  const { workspace, setWorkspace, message, setMessage, commit, clear } = useTaskWorkspace(get('model'), get('dataset'))
  const [connectionErrors, setConnectionErrors] = useState<FieldErrors>({})
  const [rememberKey, setRememberKey] = useState(
    () => workspace.connections.find((entry) => entry.id === workspace.selectedConnection)?.rememberKey ?? false,
  )

  function resetConnection() {
    setWorkspace({ ...workspace, selectedConnection: '', connection: { ...DEFAULT_CONNECTION } })
    setRememberKey(false)
    setConnectionErrors({})
  }

  function clearWorkspace() {
    if (clear()) {
      setRememberKey(false)
      setConnectionErrors({})
    }
  }

  function checkConnection() {
    const errors = validateConnection(workspace.connection)
    setConnectionErrors(errors)
    return Object.fromEntries(Object.entries(errors).map(([field, error]) => [`connection-${field}`, error]))
  }

  function changeConnection(patch: Partial<ModelConnection>) {
    const next = editConnection(workspace.connection, patch)
    if (workspace.connection.apiKey && !next.apiKey) setMessage('keyCleared')
    else if (message === 'saved' || message === 'cleared') setMessage('')
    setWorkspace({ ...workspace, connection: next })
    setConnectionErrors({})
  }

  function loadConnection(id: string) {
    const fresh = readWorkspace()
    if (!fresh.ok) {
      setMessage(fresh.error)
      return
    }
    const entries = fresh.value.connections
    const entry = entries.find((item) => item.id === id)
    setWorkspace({
      ...workspace,
      connections: entries,
      selectedConnection: id,
      connection: { ...(entry?.value ?? DEFAULT_CONNECTION) },
    })
    setRememberKey(entry?.rememberKey ?? false)
    setConnectionErrors({})
  }

  function saveConnection(name: string, asNew: boolean, rememberKey: boolean): boolean {
    const errors = checkConnection()
    if (Object.keys(errors).length) {
      document.getElementById(Object.keys(errors)[0])?.focus()
      return false
    }
    const id =
      !asNew && workspace.connections.some((entry) => entry.id === workspace.selectedConnection)
        ? workspace.selectedConnection
        : createPresetId()
    const value = { ...workspace.connection, apiKey: rememberKey ? workspace.connection.apiKey : '' }
    return commit({
      ...workspace,
      selectedConnection: id,
      connections: [...workspace.connections.filter((entry) => entry.id !== id), { id, name, value, rememberKey }],
    })
  }

  function saveEval(name: string, asNew: boolean): boolean {
    if (Object.keys(validateEvalDraft(workspace.eval)).length) {
      setMessage('invalidTemplate')
      return false
    }
    const saved = savePreset(workspace.evalTemplates, workspace.selectedEval, name, workspace.eval, asNew)
    return commit({ ...workspace, evalTemplates: saved.entries, selectedEval: saved.selectedId })
  }

  function savePerf(name: string, asNew: boolean): boolean {
    if (Object.keys(validatePerfDraft(workspace.perf)).length) {
      setMessage('invalidTemplate')
      return false
    }
    const saved = savePreset(workspace.perfTemplates, workspace.selectedPerf, name, workspace.perf, asNew)
    return commit({ ...workspace, perfTemplates: saved.entries, selectedPerf: saved.selectedId })
  }

  function loadTemplate(kind: 'eval' | 'perf', id: string) {
    const fresh = readWorkspace()
    if (!fresh.ok) {
      setMessage(fresh.error)
      return
    }
    if (kind === 'eval') {
      const entry = fresh.value.evalTemplates.find((item) => item.id === id)
      setWorkspace({
        ...workspace,
        evalTemplates: fresh.value.evalTemplates,
        selectedEval: id,
        eval: { ...(entry?.value ?? DEFAULT_EVAL) },
      })
    } else {
      const entry = fresh.value.perfTemplates.find((item) => item.id === id)
      setWorkspace({
        ...workspace,
        perfTemplates: fresh.value.perfTemplates,
        selectedPerf: id,
        perf: { ...(entry?.value ?? DEFAULT_PERF) },
      })
    }
  }

  const selectedEval = workspace.evalTemplates.find((entry) => entry.id === workspace.selectedEval)
  const selectedPerf = workspace.perfTemplates.find((entry) => entry.id === workspace.selectedPerf)
  const evalToolbar = (
    <PresetToolbar
      label={t('tasks.library.evalTemplate')}
      entries={workspace.evalTemplates}
      selectedId={workspace.selectedEval}
      dirty={JSON.stringify(workspace.eval) !== JSON.stringify(selectedEval?.value ?? DEFAULT_EVAL)}
      onLoad={(id) => loadTemplate('eval', id)}
      onSave={saveEval}
      onReset={() => setWorkspace({ ...workspace, selectedEval: '', eval: { ...DEFAULT_EVAL } })}
      onDelete={() =>
        commit({
          ...workspace,
          selectedEval: '',
          evalTemplates: workspace.evalTemplates.filter((entry) => entry.id !== workspace.selectedEval),
        })
      }
    />
  )
  const perfToolbar = (
    <PresetToolbar
      label={t('tasks.library.perfTemplate')}
      entries={workspace.perfTemplates}
      selectedId={workspace.selectedPerf}
      dirty={JSON.stringify(workspace.perf) !== JSON.stringify(selectedPerf?.value ?? DEFAULT_PERF)}
      onLoad={(id) => loadTemplate('perf', id)}
      onSave={savePerf}
      onReset={() => setWorkspace({ ...workspace, selectedPerf: '', perf: { ...DEFAULT_PERF } })}
      onDelete={() =>
        commit({
          ...workspace,
          selectedPerf: '',
          perfTemplates: workspace.perfTemplates.filter((entry) => entry.id !== workspace.selectedPerf),
        })
      }
    />
  )

  const [selected, setSelected] = useState<Record<'eval' | 'perf', string | null>>(() => {
    const read = (kind: 'eval' | 'perf'): string | null => {
      try {
        return sessionStorage.getItem(`evalscope-active-task-${kind}`)
      } catch {
        return null
      }
    }
    return { eval: read('eval'), perf: read('perf') }
  })
  const handleAdopt = useCallback(
    (kind: 'eval' | 'perf', taskId: string) => {
      setSelected((previous) => ({ ...previous, [kind]: taskId }))
      set('tab', kind)
    },
    [set],
  )

  const tabs = [
    { key: 'eval', label: t('tasks.evalTab'), panelId: 'tasks-eval-panel' },
    { key: 'perf', label: t('tasks.perfTab'), panelId: 'tasks-perf-panel' },
  ]

  return (
    <div className="page-enter flex min-w-0 flex-col gap-4">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <h1 className="text-xl font-semibold">{t('tasks.library.title')}</h1>
          <p className="mt-1 text-sm text-[var(--text-muted)]">{t('tasks.library.subtitle')}</p>
        </div>
        <ConfigLibraryTools
          library={workspace}
          onImport={(incoming) => commit(mergeLibrary(workspace, incoming))}
          onClear={clearWorkspace}
        />
      </div>
      {message && (
        <p
          role={message === 'storageFailed' || message === 'invalidStorage' ? 'alert' : 'status'}
          className="text-sm text-[var(--text-muted)]"
        >
          {t(`tasks.library.${message}`)}
        </p>
      )}
      <ModelConnectionEditor
        connection={workspace.connection}
        entries={workspace.connections}
        selectedId={workspace.selectedConnection}
        errors={connectionErrors}
        rememberKey={rememberKey}
        onRememberKeyChange={setRememberKey}
        onChange={changeConnection}
        onLoad={loadConnection}
        onSave={saveConnection}
        onReset={resetConnection}
        onDelete={() =>
          commit({
            ...workspace,
            selectedConnection: '',
            connection: { ...DEFAULT_CONNECTION },
            connections: workspace.connections.filter((entry) => entry.id !== workspace.selectedConnection),
          })
        }
        onForgetKey={() =>
          commit({
            ...workspace,
            connection: { ...workspace.connection, apiKey: '' },
            connections: workspace.connections.map((entry) =>
              entry.id === workspace.selectedConnection
                ? { ...entry, rememberKey: false, value: { ...entry.value, apiKey: '' } }
                : entry,
            ),
          })
        }
      />
      <div
        className="flex flex-wrap gap-x-4 gap-y-1 text-xs text-[var(--text-muted)]"
        aria-label={t('tasks.library.summary')}
      >
        <span>
          {t('eval.modelName')}: {workspace.connection.model || '—'}
        </span>
        <span>
          {t('perf.archive.endpoint')}:{' '}
          {workspace.connection.api === 'openai' ? displayEndpoint(workspace.connection.url) || '—' : '—'}
        </span>
        <span>
          {tab === 'eval'
            ? `${t('eval.datasets')}: ${workspace.eval.datasets || '—'}`
            : `${t('perf.task.parallel')}: ${workspace.perf.parallel} · ${t('perf.task.number')}: ${workspace.perf.number}`}
        </span>
      </div>
      <ActiveTasksPanel onAdopt={handleAdopt} selectedTaskId={selected[tab]} />
      <Tabs
        keepMounted
        tabs={tabs}
        activeKey={tab}
        onChange={(k) => set('tab', k)}
        panels={{
          'tasks-eval-panel': (
            <EvalTaskPanel
              connection={workspace.connection}
              draft={workspace.eval}
              toolbar={evalToolbar}
              onValidateConnection={checkConnection}
              onChange={(evalDraft) => setWorkspace({ ...workspace, eval: evalDraft })}
              active={tab === 'eval'}
              selectedTaskId={selected.eval}
              onSelect={(id) => handleAdopt('eval', id)}
            />
          ),
          'tasks-perf-panel': (
            <PerfTaskPanel
              connection={workspace.connection}
              draft={workspace.perf}
              toolbar={perfToolbar}
              onValidateConnection={checkConnection}
              onChange={(perfDraft) => setWorkspace({ ...workspace, perf: perfDraft })}
              active={tab === 'perf'}
              selectedTaskId={selected.perf}
              onSelect={(id) => handleAdopt('perf', id)}
            />
          ),
        }}
      />
    </div>
  )
}
