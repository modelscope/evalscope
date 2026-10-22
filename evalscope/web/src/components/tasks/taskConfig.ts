import {
  collectNumericErrors,
  FORM_MESSAGE_KEYS,
  parseNumericList,
  validateApiUrl,
  validateDatasetArgs,
  validateDatasetLimit,
  validatePositiveIntegerList,
  validateRequestRate,
} from '@/domain/form/validation'

export interface ModelConnection {
  model: string
  url: string
  apiKey: string
  alias: string
  api: 'openai' | 'local'
}

export interface EvalDraft {
  datasets: string
  limit: string
  batchSize: string
  repeats: string
  timeout: string
  stream: boolean
  temperature: string
  topP: string
  maxTokens: string
  topK: string
  datasetArgs: string
  sandboxEnabled: boolean
  sandboxEngine: 'docker' | 'volcengine'
  sandboxUrl: string
  sandboxImage: string
  sandboxPool: string
}

export interface PerfDraft {
  parallel: string
  number: string
  rate: string
  maxTokens: string
  minTokens: string
  dataset: string
  maxPromptLen: string
  minPromptLen: string
  tokenizerPath: string
}

export const DEFAULT_CONNECTION: ModelConnection = { model: '', url: '', apiKey: '', alias: '', api: 'openai' }
export const DEFAULT_EVAL: EvalDraft = {
  datasets: '',
  limit: '5',
  batchSize: '8',
  repeats: '1',
  timeout: '60',
  stream: false,
  temperature: '',
  topP: '',
  maxTokens: '',
  topK: '',
  datasetArgs: '',
  sandboxEnabled: false,
  sandboxEngine: 'docker',
  sandboxUrl: '',
  sandboxImage: '',
  sandboxPool: '',
}
export const DEFAULT_PERF: PerfDraft = {
  parallel: '1',
  number: '10',
  rate: '',
  maxTokens: '512',
  minTokens: '',
  dataset: '',
  maxPromptLen: '',
  minPromptLen: '',
  tokenizerPath: '',
}

export type FieldErrors = Record<string, string>

export function validateConnection(connection: ModelConnection): FieldErrors {
  const errors: FieldErrors = {}
  if (!connection.model.trim()) errors.model = FORM_MESSAGE_KEYS.required
  if (connection.api === 'openai') {
    const error = validateApiUrl(connection.url, true)
    if (error) errors.url = error
  }
  return errors
}

export function validateEvalDraft(draft: EvalDraft): FieldErrors {
  const errors = collectNumericErrors([
    { id: 'batchSize', value: draft.batchSize, min: 1, step: 1 },
    { id: 'repeats', value: draft.repeats, min: 1, step: 1 },
    { id: 'timeout', value: draft.timeout, min: 0 },
    { id: 'temperature', value: draft.temperature, min: 0, max: 2 },
    { id: 'topP', value: draft.topP, min: 0, max: 1 },
    { id: 'maxTokens', value: draft.maxTokens, min: 1, step: 1 },
    { id: 'topK', value: draft.topK, step: 1 },
    ...(draft.sandboxEnabled ? [{ id: 'sandboxPool', value: draft.sandboxPool, min: 1, step: 1 }] : []),
  ])
  if (!draft.datasets.split(',').some((name) => name.trim())) errors.datasets = FORM_MESSAGE_KEYS.required
  const limitError = validateDatasetLimit(draft.limit)
  if (limitError) errors.limit = limitError
  if (draft.timeout.trim() && Number(draft.timeout) <= 0) errors.timeout = FORM_MESSAGE_KEYS.numericBelowMin
  if (draft.sandboxEnabled && draft.sandboxEngine === 'docker' && draft.sandboxUrl.trim()) {
    const error = validateApiUrl(draft.sandboxUrl, false)
    if (error) errors.sandboxUrl = error
  }
  if (draft.datasetArgs.trim()) {
    const result = validateDatasetArgs(draft.datasetArgs)
    if (!result.ok) errors.datasetArgs = result.messageKey
  }
  return errors
}

export function validatePerfDraft(draft: PerfDraft): FieldErrors {
  const errors = collectNumericErrors([
    { id: 'maxTokens', value: draft.maxTokens, min: 1, step: 1 },
    { id: 'minTokens', value: draft.minTokens, min: 0, step: 1 },
    { id: 'maxPromptLen', value: draft.maxPromptLen, min: 0, step: 1 },
    { id: 'minPromptLen', value: draft.minPromptLen, min: 0, step: 1 },
  ])
  for (const field of ['parallel', 'number'] as const) {
    const error = validatePositiveIntegerList(draft[field])
    if (error) errors[field] = error
  }
  const rateError = validateRequestRate(draft.rate)
  if (rateError) errors.rate = rateError
  if (parseNumericList(draft.parallel).length !== parseNumericList(draft.number).length) {
    errors.number = 'perf.task.sweepLengthMismatch'
  }
  const maxTokens = draft.maxTokens.trim() ? Number(draft.maxTokens) : 2048
  const maxPrompt = draft.maxPromptLen.trim() ? Number(draft.maxPromptLen) : 131072
  if (draft.minTokens.trim() && Number(draft.minTokens) > maxTokens) errors.minTokens = 'perf.task.invalidTokenRange'
  if (draft.minPromptLen.trim() && Number(draft.minPromptLen) > maxPrompt)
    errors.minPromptLen = 'perf.task.invalidPromptRange'
  if (draft.dataset.trim() === 'random' && !draft.tokenizerPath.trim())
    errors.tokenizerPath = FORM_MESSAGE_KEYS.required
  return errors
}

/** Only explicit connection fields can become transport parameters. */
export function connectionPayload(connection: ModelConnection, kind: 'eval' | 'perf'): Record<string, unknown> {
  const payload: Record<string, unknown> = { model: connection.model.trim() }
  if (kind === 'perf') payload.api = connection.api
  if (connection.api === 'openai') {
    payload[kind === 'eval' ? 'api_url' : 'url'] = connection.url.trim()
    if (connection.apiKey) payload.api_key = connection.apiKey
  }
  if (connection.alias.trim()) payload.alias = connection.alias.trim()
  return payload
}

export function buildEvalPayload(draft: EvalDraft): Record<string, unknown> {
  const generation: Record<string, unknown> = { stream: draft.stream }
  for (const [field, key] of [
    ['timeout', 'timeout'],
    ['temperature', 'temperature'],
    ['topP', 'top_p'],
    ['maxTokens', 'max_tokens'],
    ['topK', 'top_k'],
  ] as const) {
    if (draft[field].trim()) generation[key] = Number(draft[field])
  }
  const payload: Record<string, unknown> = {
    datasets: draft.datasets
      .split(',')
      .map((name) => name.trim())
      .filter(Boolean),
    limit: draft.limit.trim() ? Number(draft.limit) : undefined,
    eval_batch_size: draft.batchSize.trim() ? Number(draft.batchSize) : undefined,
    generation_config: generation,
  }
  if (draft.repeats.trim() && Number(draft.repeats) > 1) payload.repeats = Number(draft.repeats)
  if (draft.datasetArgs.trim()) {
    const parsed = validateDatasetArgs(draft.datasetArgs)
    if (parsed.ok) payload.dataset_args = parsed.value
  }
  if (draft.sandboxEnabled) {
    const sandbox: Record<string, unknown> = { enabled: true, engine: draft.sandboxEngine }
    if (draft.sandboxEngine === 'docker') {
      if (draft.sandboxUrl.trim()) sandbox.manager_config = { base_url: draft.sandboxUrl.trim() }
      if (draft.sandboxImage.trim()) sandbox.default_config = { image: draft.sandboxImage.trim() }
    }
    if (draft.sandboxPool.trim()) sandbox.pool_size = Number(draft.sandboxPool)
    payload.sandbox = sandbox
  }
  return payload
}

export function buildPerfPayload(draft: PerfDraft): Record<string, unknown> {
  const payload: Record<string, unknown> = {
    parallel: parseNumericList(draft.parallel),
    number: parseNumericList(draft.number),
  }
  for (const [field, key] of [
    ['rate', 'rate'],
    ['maxTokens', 'max_tokens'],
    ['minTokens', 'min_tokens'],
    ['maxPromptLen', 'max_prompt_length'],
    ['minPromptLen', 'min_prompt_length'],
  ] as const) {
    if (draft[field].trim()) payload[key] = Number(draft[field])
  }
  if (draft.dataset.trim()) payload.dataset = draft.dataset.trim()
  if (draft.tokenizerPath.trim()) payload.tokenizer_path = draft.tokenizerPath.trim()
  return payload
}

export function endpointOrigin(url: string): string | null {
  try {
    return new URL(url.trim()).origin
  } catch {
    return null
  }
}

export function displayEndpoint(url: string): string {
  try {
    return new URL(url.trim()).host
  } catch {
    return ''
  }
}

/** Moving to another service must not carry credentials from the previous one. */
export function editConnection(connection: ModelConnection, patch: Partial<ModelConnection>): ModelConnection {
  const next = { ...connection, ...patch }
  const changedOrigin = patch.url !== undefined && endpointOrigin(connection.url) !== endpointOrigin(next.url)
  if (changedOrigin || (patch.api !== undefined && patch.api !== connection.api)) next.apiKey = ''
  return next
}
