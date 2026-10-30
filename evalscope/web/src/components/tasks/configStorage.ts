import {
  DEFAULT_CONNECTION,
  DEFAULT_EVAL,
  DEFAULT_PERF,
  endpointOrigin,
  validateConnection,
  validateEvalDraft,
  validatePerfDraft,
  type EvalDraft,
  type ModelConnection,
  type PerfDraft,
} from './taskConfig'

export const CONFIG_STORAGE_KEY = 'evalscope-task-workspace-v1'
export const MAX_IMPORT_BYTES = 1024 * 1024
export interface Preset<T> {
  id: string
  name: string
  value: T
}
export interface ConnectionPreset extends Preset<ModelConnection> {
  rememberKey: boolean
}
export interface ConfigLibrary {
  version: 1
  connections: ConnectionPreset[]
  evalTemplates: Preset<EvalDraft>[]
  perfTemplates: Preset<PerfDraft>[]
}
export interface Workspace extends ConfigLibrary {
  connection: ModelConnection
  eval: EvalDraft
  perf: PerfDraft
  selectedConnection: string
  selectedEval: string
  selectedPerf: string
  restoreSavedKey: boolean
}
export type StorageResult<T> = { ok: true; value: T } | { ok: false; error: 'invalidStorage' | 'storageFailed' }

export function emptyWorkspace(): Workspace {
  return {
    version: 1,
    connections: [],
    evalTemplates: [],
    perfTemplates: [],
    connection: { ...DEFAULT_CONNECTION },
    eval: { ...DEFAULT_EVAL },
    perf: { ...DEFAULT_PERF },
    selectedConnection: '',
    selectedEval: '',
    selectedPerf: '',
    restoreSavedKey: false,
  }
}

function object(value: unknown): Record<string, unknown> {
  if (!value || typeof value !== 'object' || Array.isArray(value)) throw new Error('Invalid object')
  return value as Record<string, unknown>
}

function exactKeys(value: Record<string, unknown>, keys: string[]): void {
  if (Object.keys(value).some((key) => !keys.includes(key))) throw new Error('Unknown field')
}

/** Local storage and imported JSON are untrusted; reconstruct only known fields. */
function parseDraft<T extends object>(value: unknown, defaults: T): T {
  const raw = object(value)
  exactKeys(raw, Object.keys(defaults))
  const result = { ...defaults }
  for (const key of Object.keys(defaults) as (keyof T)[]) {
    if (!(key in raw)) continue
    if (typeof raw[String(key)] !== typeof defaults[key]) throw new Error('Invalid field')
    if (typeof raw[String(key)] === 'string' && String(raw[String(key)]).length > MAX_IMPORT_BYTES)
      throw new Error('Field too large')
    result[key] = raw[String(key)] as T[keyof T]
  }
  return result
}

function parseConnection(value: unknown): ModelConnection {
  const connection = parseDraft(value, DEFAULT_CONNECTION)
  if (!['openai', 'local'].includes(connection.api)) throw new Error('Invalid protocol')
  return connection
}

function parseEval(value: unknown): EvalDraft {
  const draft = parseDraft(value, DEFAULT_EVAL)
  if (!['docker', 'volcengine'].includes(draft.sandboxEngine)) throw new Error('Invalid engine')
  return draft
}

function parsePresets<T>(value: unknown, parse: (value: unknown) => T, allowCredentialChoice = false): Preset<T>[] {
  if (!Array.isArray(value) || value.length > 200) throw new Error('Invalid presets')
  const ids = new Set<string>()
  return value.map((entry: unknown) => {
    const raw = object(entry)
    exactKeys(raw, allowCredentialChoice ? ['id', 'name', 'value', 'rememberKey'] : ['id', 'name', 'value'])
    if (typeof raw.id !== 'string' || !raw.id || ids.has(raw.id)) throw new Error('Invalid id')
    if (typeof raw.name !== 'string' || !raw.name.trim() || raw.name.length > 120) throw new Error('Invalid name')
    ids.add(raw.id)
    return { id: raw.id, name: raw.name.trim(), value: parse(raw.value) }
  })
}

function parseLibrary(raw: Record<string, unknown>): ConfigLibrary {
  if (raw.version !== 1) throw new Error('Unsupported version')
  const connections = parsePresets(raw.connections, parseConnection, true).map((entry, index) => {
    const source = object((raw.connections as unknown[])[index])
    if (typeof source.rememberKey !== 'boolean') throw new Error('Invalid credential choice')
    return {
      ...entry,
      rememberKey: source.rememberKey,
      value: { ...entry.value, apiKey: source.rememberKey ? entry.value.apiKey : '' },
    }
  })
  return {
    version: 1,
    connections,
    evalTemplates: parsePresets(raw.evalTemplates, parseEval),
    perfTemplates: parsePresets(raw.perfTemplates, (value) => parseDraft(value, DEFAULT_PERF)),
  }
}

export function parseImport(text: string): ConfigLibrary {
  if (new TextEncoder().encode(text).length > MAX_IMPORT_BYTES) throw new Error('Import too large')
  const raw = object(
    JSON.parse(text, (key, value: unknown) => {
      if (['__proto__', 'prototype', 'constructor'].includes(key)) throw new Error('Unsafe field')
      return value
    }),
  )
  exactKeys(raw, ['version', 'connections', 'evalTemplates', 'perfTemplates'])
  const library = parseLibrary(raw)
  if (
    library.connections.some((entry) => Object.keys(validateConnection(entry.value)).length > 0) ||
    library.evalTemplates.some((entry) => Object.keys(validateEvalDraft(entry.value)).length > 0) ||
    library.perfTemplates.some((entry) => Object.keys(validatePerfDraft(entry.value)).length > 0)
  ) {
    throw new Error('Invalid configuration values')
  }
  return library
}

export function readWorkspace(): StorageResult<Workspace> {
  try {
    const text = localStorage.getItem(CONFIG_STORAGE_KEY)
    if (!text) return { ok: true, value: emptyWorkspace() }
    try {
      const raw = object(JSON.parse(text))
      exactKeys(raw, Object.keys(emptyWorkspace()))
      const library = parseLibrary(raw)
      const state: Workspace = {
        ...library,
        connection: parseConnection(raw.connection),
        eval: parseEval(raw.eval),
        perf: parseDraft(raw.perf, DEFAULT_PERF),
        selectedConnection: '',
        selectedEval: '',
        selectedPerf: '',
        restoreSavedKey: false,
      }
      for (const key of ['selectedConnection', 'selectedEval', 'selectedPerf'] as const) {
        if (typeof raw[key] !== 'string') throw new Error('Invalid selection')
        state[key] = raw[key]
      }
      if (typeof raw.restoreSavedKey !== 'boolean') throw new Error('Invalid credential restore')
      state.connection.apiKey = ''
      const selected = state.connections.find((entry) => entry.id === state.selectedConnection)
      if (
        raw.restoreSavedKey &&
        selected?.rememberKey &&
        selected.value.api === state.connection.api &&
        endpointOrigin(selected.value.url) === endpointOrigin(state.connection.url)
      ) {
        state.connection.apiKey = selected.value.apiKey
      }
      return { ok: true, value: state }
    } catch {
      return { ok: false, error: 'invalidStorage' }
    }
  } catch {
    return { ok: false, error: 'storageFailed' }
  }
}

export function storageSnapshot(workspace: Workspace): Workspace {
  const selected = workspace.connections.find((entry) => entry.id === workspace.selectedConnection)
  return {
    ...workspace,
    connections: workspace.connections.map((entry) => ({
      ...entry,
      value: { ...entry.value, apiKey: entry.rememberKey ? entry.value.apiKey : '' },
    })),
    connection: { ...workspace.connection, apiKey: '' },
    restoreSavedKey: Boolean(
      selected?.rememberKey &&
        selected.value.apiKey &&
        selected.value.apiKey === workspace.connection.apiKey &&
        selected.value.api === workspace.connection.api &&
        endpointOrigin(selected.value.url) === endpointOrigin(workspace.connection.url),
    ),
  }
}

export function writeWorkspace(workspace: Workspace): StorageResult<null> {
  try {
    if (
      [workspace.connections, workspace.evalTemplates, workspace.perfTemplates].some((entries) => entries.length > 200)
    ) {
      return { ok: false, error: 'invalidStorage' }
    }
    localStorage.setItem(CONFIG_STORAGE_KEY, JSON.stringify(storageSnapshot(workspace)))
    return { ok: true, value: null }
  } catch {
    return { ok: false, error: 'storageFailed' }
  }
}

export function exportLibrary(workspace: ConfigLibrary, includeKeys: boolean): string {
  return JSON.stringify(
    {
      version: 1,
      connections: workspace.connections.map((entry) => ({
        ...entry,
        rememberKey: includeKeys && entry.rememberKey,
        value: { ...entry.value, apiKey: includeKeys && entry.rememberKey ? entry.value.apiKey : '' },
      })),
      evalTemplates: workspace.evalTemplates,
      perfTemplates: workspace.perfTemplates,
    },
    null,
    2,
  )
}

function uniqueName(name: string, existing: string[]): string {
  const base = name.slice(0, 110)
  let result = name
  let index = 2
  while (existing.includes(result)) result = `${base} (${index++})`
  return result
}

export function createPresetId(): string {
  return Array.from(crypto.getRandomValues(new Uint32Array(4)), (part) => part.toString(16).padStart(8, '0')).join('')
}

export function savePreset<T extends object>(
  entries: Preset<T>[],
  selectedId: string,
  name: string,
  value: T,
  asNew: boolean,
): { entries: Preset<T>[]; selectedId: string } {
  const id = !asNew && entries.some((entry) => entry.id === selectedId) ? selectedId : createPresetId()
  const entry = { id, name, value: { ...value } }
  return { entries: [...entries.filter((item) => item.id !== id), entry], selectedId: id }
}

export function mergeLibrary(workspace: Workspace, imported: ConfigLibrary): Workspace {
  function append<T>(existing: Preset<T>[], incoming: Preset<T>[]): Preset<T>[] {
    const result = [...existing]
    for (const entry of incoming) {
      result.push({
        ...entry,
        id: createPresetId(),
        name: uniqueName(
          entry.name,
          result.map((item) => item.name),
        ),
      })
    }
    return result
  }
  const connections = [...workspace.connections]
  for (const entry of imported.connections) {
    connections.push({
      ...entry,
      id: createPresetId(),
      name: uniqueName(
        entry.name,
        connections.map((item) => item.name),
      ),
    })
  }
  return {
    ...workspace,
    connections,
    evalTemplates: append(workspace.evalTemplates, imported.evalTemplates),
    perfTemplates: append(workspace.perfTemplates, imported.perfTemplates),
  }
}
