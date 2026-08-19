import { useEffect, useId, useRef, useState, type KeyboardEvent } from 'react'
import { useLocale } from '@/contexts/LocaleContext'
import { useAsyncResource } from '@/hooks/useAsyncResource'
import { listBenchmarks } from '@/api/eval'
import Field from '@/components/ui/Field'
import { inputClass } from '@/components/ui/formStyles'
import {
  collectNumericErrors,
  validateDatasetArgs,
  FORM_MESSAGE_KEYS,
} from '@/domain/form/validation'
import { ApiKeyField, ApiUrlField, ModelField, OutputDirField } from '@/components/tasks/TaskFormFields'
import TaskFormShell from '@/components/tasks/TaskFormShell'
import { useTaskForm } from '@/components/tasks/useTaskForm'

interface Props {
  onSubmit: (config: Record<string, unknown>, outputDirName?: string) => void
  disabled?: boolean
  initialDataset?: string
  /**
   * Model to prefill, so a past run can be repeated without retyping it.
   *
   * Only non-secret fields are prefillable this way: the value arrives through the URL, and an API
   * key in a URL ends up in history and in server logs.
   */
  initialModel?: string
}

/** Stable placeholder so an unresolved suggestion list keeps a single identity. */
const EMPTY_NAMES: string[] = []

/** Stable field ids, reused as label/error association targets and focus targets. */
const IDS = {
  model: 'eval-model',
  datasets: 'eval-datasets',
  apiUrl: 'eval-apiUrl',
  apiKey: 'eval-apiKey',
  limit: 'eval-limit',
  batchSize: 'eval-batchSize',
  repeats: 'eval-repeats',
  timeout: 'eval-timeout',
  stream: 'eval-stream',
  temperature: 'eval-temperature',
  topP: 'eval-topP',
  maxTokens: 'eval-maxTokens',
  topK: 'eval-topK',
  datasetArgs: 'eval-datasetArgs',
  sandboxEngine: 'eval-sandboxEngine',
  sandboxUrl: 'eval-sandboxUrl',
  sandboxImage: 'eval-sandboxImage',
  sandboxPool: 'eval-sandboxPool',
  sandboxEnable: 'eval-sandboxEnable',
  outputDir: 'eval-outputDir',
} as const

/** DOM order of focusable fields, drives first-invalid focus on submit. */
const DOM_ORDER: string[] = [
  IDS.model,
  IDS.datasets,
  IDS.apiUrl,
  IDS.apiKey,
  IDS.limit,
  IDS.batchSize,
  IDS.repeats,
  IDS.timeout,
  IDS.temperature,
  IDS.topP,
  IDS.maxTokens,
  IDS.topK,
  IDS.datasetArgs,
  IDS.sandboxPool,
  IDS.outputDir,
]

/** Fields that live inside the collapsible "More Parameters" section. */
const MORE_PARAMS_IDS: string[] = [
  IDS.repeats,
  IDS.timeout,
  IDS.temperature,
  IDS.topP,
  IDS.maxTokens,
  IDS.topK,
  IDS.datasetArgs,
  IDS.sandboxPool,
  IDS.outputDir,
]

export default function EvalConfigForm({ onSubmit, disabled, initialDataset, initialModel }: Props) {
  const { t } = useLocale()
  const [model, setModel] = useState(initialModel ?? '')
  const [apiUrl, setApiUrl] = useState('')
  const [apiKey, setApiKey] = useState('')
  const [datasets, setDatasets] = useState(initialDataset ?? '')
  const [limit, setLimit] = useState('5')
  const [evalBatchSize, setEvalBatchSize] = useState('16')
  const [repeats, setRepeats] = useState('1')
  const [timeout, setTimeout_] = useState('60')
  const [stream, setStream] = useState(false)
  const [temperature, setTemperature] = useState('')
  const [topP, setTopP] = useState('')
  const [maxTokens, setMaxTokens] = useState('')
  const [topK, setTopK] = useState('')
  const [datasetArgs, setDatasetArgs] = useState('')
  const [sandboxEnabled, setSandboxEnabled] = useState(false)
  const [sandboxEngine, setSandboxEngine] = useState('docker')
  const [sandboxUrl, setSandboxUrl] = useState('')
  const [sandboxImage, setSandboxImage] = useState('')
  const [sandboxPoolSize, setSandboxPoolSize] = useState('')
  const [outputDirName, setOutputDirName] = useState('')

  const { errorFor: errMsg, clearError: clearErr, showMore, toggleMore, submitHandler } = useTaskForm({
    domOrder: DOM_ORDER,
    moreParamsIds: MORE_PARAMS_IDS,
  })

  const validate = (): Record<string, string> => {
    const errors: Record<string, string> = {}

    if (!model.trim()) errors[IDS.model] = FORM_MESSAGE_KEYS.required
    if (!datasets.trim()) errors[IDS.datasets] = FORM_MESSAGE_KEYS.required

    Object.assign(errors, collectNumericErrors([
      { id: IDS.limit, value: limit, min: 1 },
      { id: IDS.batchSize, value: evalBatchSize, min: 1 },
      { id: IDS.repeats, value: repeats, min: 1 },
      { id: IDS.timeout, value: timeout, min: 0 },
      { id: IDS.temperature, value: temperature, min: 0, max: 2, step: 0.1 },
      { id: IDS.topP, value: topP, min: 0, max: 1, step: 0.1 },
      { id: IDS.maxTokens, value: maxTokens, min: 1 },
      { id: IDS.topK, value: topK, min: 1 },
      ...(sandboxEnabled
        ? [{ id: IDS.sandboxPool, value: sandboxPoolSize, min: 1, step: 1 }]
        : []),
    ]))

    // Dataset_Args JSON validation without mutating the raw input.
    if (datasetArgs.trim()) {
      const result = validateDatasetArgs(datasetArgs)
      if (!result.ok) errors[IDS.datasetArgs] = result.messageKey
    }

    return errors
  }

  const buildConfig = () => {
    const config: Record<string, unknown> = {
      model,
      datasets: datasets.split(',').map((s) => s.trim()).filter(Boolean),
      limit: limit ? Number(limit) : undefined,
      eval_batch_size: evalBatchSize ? Number(evalBatchSize) : undefined,
    }
    if (apiUrl) config.api_url = apiUrl
    if (apiKey) config.api_key = apiKey
    if (repeats && Number(repeats) > 1) config.repeats = Number(repeats)
    if (timeout) config.timeout = Number(timeout)
    if (stream) config.stream = true
    // Wrap generation params into generation_config dict
    const genConfig: Record<string, unknown> = {}
    if (temperature) genConfig.temperature = Number(temperature)
    if (topP) genConfig.top_p = Number(topP)
    if (maxTokens) genConfig.max_tokens = Number(maxTokens)
    if (topK) genConfig.top_k = Number(topK)
    if (Object.keys(genConfig).length > 0) config.generation_config = genConfig
    // Re-parsed rather than carried over from validation: submission only runs
    // once validation passed, so this cannot fail here.
    if (datasetArgs.trim()) {
      const parsed = validateDatasetArgs(datasetArgs)
      if (parsed.ok) config.dataset_args = parsed.value
    }
    if (sandboxEnabled) {
      const sandbox: Record<string, unknown> = {
        enabled: true,
        engine: sandboxEngine,
      }

      if (sandboxUrl.trim()) {
        sandbox.manager_config = {
          base_url: sandboxUrl.trim(),
        }
      }

      if (sandboxImage.trim()) {
        sandbox.default_config = {
          image: sandboxImage.trim(),
        }
      }

      if (sandboxPoolSize.trim()) {
        sandbox.pool_size = Number(sandboxPoolSize)
      }

      config.sandbox = sandbox
    }
    onSubmit(config, outputDirName.trim() || undefined)
  }

  // Dataset autocomplete
  const [showSuggestions, setShowSuggestions] = useState(false)
  const [filteredSuggestions, setFilteredSuggestions] = useState<string[]>([])
  const [activeSuggestion, setActiveSuggestion] = useState(-1)
  const datasetInputRef = useRef<HTMLDivElement>(null)
  const datasetListboxId = `${useId()}-dataset-listbox`

  useEffect(() => {
    const applyInitial = () => {
      if (initialDataset) setDatasets(initialDataset)
      if (initialModel) setModel(initialModel)
    }
    applyInitial()
  }, [initialDataset, initialModel])

  // Dataset autocomplete suggestions. A failure here only costs the suggestions,
  // so the field stays usable and no error is surfaced.
  const benchmarks = useAsyncResource(
    async (signal) => {
      const res = await listBenchmarks(undefined, undefined, signal)
      return [
        ...(res.text ?? []).map((b) => b.name),
        ...(res.multimodal ?? []).map((b) => b.name),
      ]
    },
    [],
  )
  const benchmarkNames = benchmarks.data ?? EMPTY_NAMES

  // Close suggestions on outside click
  useEffect(() => {
    const handler = (e: MouseEvent) => {
      if (datasetInputRef.current && !datasetInputRef.current.contains(e.target as Node)) {
        setShowSuggestions(false)
      }
    }
    document.addEventListener('mousedown', handler)
    return () => document.removeEventListener('mousedown', handler)
  }, [])

  const handleDatasetChange = (val: string) => {
    setDatasets(val)
    // Filter based on last token after comma
    const parts = val.split(',')
    const current = parts[parts.length - 1].trim().toLowerCase()
    if (current) {
      const matches = benchmarkNames.filter((n) => n.toLowerCase().includes(current))
      setFilteredSuggestions(matches.slice(0, 8))
      setShowSuggestions(matches.length > 0)
      setActiveSuggestion(matches.length > 0 ? 0 : -1)
    } else {
      setShowSuggestions(false)
      setActiveSuggestion(-1)
    }
    clearErr(IDS.datasets)
  }

  const selectSuggestion = (name: string) => {
    const parts = datasets.split(',').map((s) => s.trim())
    parts[parts.length - 1] = name
    setDatasets(parts.join(', '))
    setShowSuggestions(false)
    setActiveSuggestion(-1)
    clearErr(IDS.datasets)
  }

  const handleDatasetKeyDown = (event: KeyboardEvent<HTMLInputElement>) => {
    if (event.key === 'Escape' && showSuggestions) {
      event.preventDefault()
      setShowSuggestions(false)
      setActiveSuggestion(-1)
      return
    }
    if (event.key === 'ArrowDown' || event.key === 'ArrowUp') {
      if (filteredSuggestions.length === 0) return
      event.preventDefault()
      setShowSuggestions(true)
      setActiveSuggestion((current) => {
        if (event.key === 'ArrowDown') return Math.min(Math.max(current, -1) + 1, filteredSuggestions.length - 1)
        return Math.max(current <= 0 ? 0 : current - 1, 0)
      })
      return
    }
    if (event.key === 'Enter' && showSuggestions && activeSuggestion >= 0) {
      event.preventDefault()
      selectSuggestion(filteredSuggestions[activeSuggestion])
    }
  }

  /** Fields behind the "more parameters" disclosure. */
  const MORE_PARAMS = (
    <>
        <Field id={IDS.repeats} name="repeats" labelKey="eval.repeats" error={errMsg(IDS.repeats)}>
          {(aria) => (
            <input {...aria} type="number" min={1} value={repeats} onChange={(e) => { setRepeats(e.target.value); clearErr(IDS.repeats) }} className={inputClass(errMsg(IDS.repeats))} />
          )}
        </Field>
        <Field id={IDS.timeout} name="timeout" labelKey="eval.timeout" error={errMsg(IDS.timeout)}>
          {(aria) => (
            <input {...aria} type="number" min={0} value={timeout} onChange={(e) => { setTimeout_(e.target.value); clearErr(IDS.timeout) }} className={inputClass(errMsg(IDS.timeout))} />
          )}
        </Field>
        <Field id={IDS.stream} name="stream" labelKey="eval.stream">
          {(aria) => (
            <input
              {...aria}
              type="checkbox"
              checked={stream}
              onChange={(e) => setStream(e.target.checked)}
              className="size-11 accent-[var(--accent)] cursor-pointer"
            />
          )}
        </Field>
        <Field id={IDS.temperature} name="temperature" labelKey="eval.temperature" error={errMsg(IDS.temperature)}>
          {(aria) => (
            <input {...aria} type="number" min={0} max={2} step="0.1" value={temperature} onChange={(e) => { setTemperature(e.target.value); clearErr(IDS.temperature) }} className={inputClass(errMsg(IDS.temperature))} />
          )}
        </Field>
        <Field id={IDS.topP} name="top_p" labelKey="eval.topP" error={errMsg(IDS.topP)}>
          {(aria) => (
            <input {...aria} type="number" min={0} max={1} step="0.1" value={topP} onChange={(e) => { setTopP(e.target.value); clearErr(IDS.topP) }} className={inputClass(errMsg(IDS.topP))} />
          )}
        </Field>
        <Field id={IDS.maxTokens} name="max_tokens" labelKey="eval.maxTokens" error={errMsg(IDS.maxTokens)}>
          {(aria) => (
            <input {...aria} type="number" min={1} value={maxTokens} onChange={(e) => { setMaxTokens(e.target.value); clearErr(IDS.maxTokens) }} className={inputClass(errMsg(IDS.maxTokens))} />
          )}
        </Field>
        <Field id={IDS.topK} name="top_k" labelKey="eval.topK" error={errMsg(IDS.topK)}>
          {(aria) => (
            <input {...aria} type="number" min={1} value={topK} onChange={(e) => { setTopK(e.target.value); clearErr(IDS.topK) }} className={inputClass(errMsg(IDS.topK))} />
          )}
        </Field>
        <Field id={IDS.datasetArgs} name="dataset_args" labelKey="eval.datasetArgs" error={errMsg(IDS.datasetArgs)} className="md:col-span-2">
          {(aria) => (
            <textarea
              {...aria}
              value={datasetArgs}
              onChange={(e) => { setDatasetArgs(e.target.value); clearErr(IDS.datasetArgs) }}
              className={inputClass(errMsg(IDS.datasetArgs), 'h-20 resize-y')}
              style={{ fontFamily: 'var(--font-mono)' }}
              placeholder='{"gsm8k": {"few_shot_num": 4}}'
            />
          )}
        </Field>
        <OutputDirField id={IDS.outputDir} value={outputDirName} onChange={setOutputDirName} />
        <hr className="md:col-span-3 my-2" />
        <h3 className="md:col-span-3 font-semibold">{t('eval.sandbox')}</h3>
        <Field id={IDS.sandboxEnable} name="sandbox_enabled" labelKey="eval.sandboxEnable">
          {(aria) => (
            <input
              {...aria}
              type="checkbox"
              checked={sandboxEnabled}
              onChange={(e) => setSandboxEnabled(e.target.checked)}
              className="size-11 accent-[var(--accent)] cursor-pointer"
            />
          )}
        </Field>
        {sandboxEnabled && (
          <>
            <Field id={IDS.sandboxEngine} name="sandbox_engine" labelKey="eval.sandboxEngine">
              {(aria) => (
                <select {...aria} value={sandboxEngine} onChange={(e) => setSandboxEngine(e.target.value)} className={inputClass()}>
                  <option value="docker">docker</option>
                  <option value="volcengine">volcengine</option>
                </select>
              )}
            </Field>
            <Field id={IDS.sandboxUrl} name="sandbox_url" labelKey="eval.sandboxUrl">
              {(aria) => (
                <input {...aria} value={sandboxUrl} onChange={(e) => setSandboxUrl(e.target.value)} className={inputClass()} />
              )}
            </Field>
            <Field id={IDS.sandboxImage} name="sandbox_image" labelKey="eval.sandboxImage">
              {(aria) => (
                <input {...aria} value={sandboxImage} onChange={(e) => setSandboxImage(e.target.value)} className={inputClass()} />
              )}
            </Field>
            <Field id={IDS.sandboxPool} name="sandbox_pool" labelKey="eval.sandboxPool" error={errMsg(IDS.sandboxPool)}>
              {(aria) => (
                <input {...aria} type="number" min={1} step={1} value={sandboxPoolSize} onChange={(e) => { setSandboxPoolSize(e.target.value); clearErr(IDS.sandboxPool) }} className={inputClass(errMsg(IDS.sandboxPool))} />
              )}
            </Field>
          </>
        )}
    </>
  )

  return (
    <TaskFormShell
      onSubmit={submitHandler(validate, buildConfig)}
      moreParamsLabel={t('eval.moreParams')}
      showMore={showMore}
      onToggleMore={toggleMore}
      submitLabel={t('eval.startEval')}
      disabled={disabled}
      moreParams={MORE_PARAMS}
    >
      <ModelField
          id={IDS.model}
          value={model}
          error={errMsg(IDS.model)}
          onChange={(value) => { setModel(value); clearErr(IDS.model) }}
        />

        {/* Datasets with autocomplete */}
        <Field id={IDS.datasets} name="datasets" labelKey="eval.datasets" required error={errMsg(IDS.datasets)} autoComplete="off" className="relative">
          {(aria) => (
            <div ref={datasetInputRef}>
              <input
                {...aria}
                type="text"
                value={datasets}
                onChange={(e) => handleDatasetChange(e.target.value)}
                onFocus={() => { if (filteredSuggestions.length) setShowSuggestions(true) }}
                onKeyDown={handleDatasetKeyDown}
                role="combobox"
                aria-autocomplete="list"
                aria-expanded={showSuggestions}
                aria-controls={datasetListboxId}
                aria-activedescendant={showSuggestions && activeSuggestion >= 0 ? `${datasetListboxId}-option-${activeSuggestion}` : undefined}
                className={inputClass(errMsg(IDS.datasets))}
                placeholder="gsm8k, arc"
              />
              {showSuggestions && (
                <div id={datasetListboxId} role="listbox" className="absolute z-50 left-0 right-0 mt-1 rounded-[var(--radius-sm)] border border-[var(--border-md)] bg-[var(--bg-card)] shadow-[var(--shadow)] overflow-hidden max-h-48 overflow-y-auto">
                  {filteredSuggestions.map((name, index) => (
                    <button
                      key={name}
                      id={`${datasetListboxId}-option-${index}`}
                      type="button"
                      role="option"
                      aria-selected={index === activeSuggestion}
                      onMouseEnter={() => setActiveSuggestion(index)}
                      onClick={() => selectSuggestion(name)}
                      className={`w-full min-h-11 text-left px-3 py-2 text-sm text-[var(--text)] transition-colors cursor-pointer ${index === activeSuggestion ? 'bg-[var(--bg-card2)]' : 'hover:bg-[var(--bg-card2)]'}`}
                    >
                      {name}
                    </button>
                  ))}
                </div>
              )}
            </div>
          )}
        </Field>

        <ApiUrlField id={IDS.apiUrl} name="api_url" value={apiUrl} onChange={setApiUrl} />
        <ApiKeyField id={IDS.apiKey} value={apiKey} onChange={setApiKey} />

        <Field id={IDS.limit} name="limit" labelKey="eval.limit" error={errMsg(IDS.limit)}>
          {(aria) => (
            <input {...aria} type="number" min={1} value={limit} onChange={(e) => { setLimit(e.target.value); clearErr(IDS.limit) }} className={inputClass(errMsg(IDS.limit))} />
          )}
        </Field>

        <Field id={IDS.batchSize} name="eval_batch_size" labelKey="eval.batchSize" error={errMsg(IDS.batchSize)}>
          {(aria) => (
            <input {...aria} type="number" min={1} value={evalBatchSize} onChange={(e) => { setEvalBatchSize(e.target.value); clearErr(IDS.batchSize) }} className={inputClass(errMsg(IDS.batchSize))} />
          )}
        </Field>
    </TaskFormShell>
  )
}
