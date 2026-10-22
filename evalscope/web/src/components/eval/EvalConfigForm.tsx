import { useEffect, useId, useRef, useState, type KeyboardEvent, type ReactNode } from 'react'
import { useLocale } from '@/contexts/LocaleContext'
import { useAsyncResource } from '@/hooks/useAsyncResource'
import { listBenchmarks } from '@/api/eval'
import Field from '@/components/ui/Field'
import Switch from '@/components/ui/Switch'
import { inputClass } from '@/components/ui/formStyles'
import { buildEvalPayload, validateEvalDraft, type EvalDraft } from '@/components/tasks/taskConfig'
import TaskFormShell from '@/components/tasks/TaskFormShell'
import { useTaskForm } from '@/components/tasks/useTaskForm'

interface Props {
  onSubmit: (config: Record<string, unknown>) => void
  disabled?: boolean
  draft: EvalDraft
  onChange: (draft: EvalDraft) => void
  toolbar?: ReactNode
  onValidateConnection: () => Record<string, string>
}

/** Stable placeholder so an unresolved suggestion list keeps a single identity. */
const EMPTY_NAMES: string[] = []

/** Stable field ids, reused as label/error association targets and focus targets. */
const IDS = {
  datasets: 'eval-datasets',
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
} as const

/** DOM order of focusable fields, drives first-invalid focus on submit. */
const DOM_ORDER: string[] = [
  'connection-model',
  'connection-url',
  IDS.datasets,
  IDS.limit,
  IDS.batchSize,
  IDS.repeats,
  IDS.timeout,
  IDS.temperature,
  IDS.topP,
  IDS.maxTokens,
  IDS.topK,
  IDS.datasetArgs,
  IDS.sandboxUrl,
  IDS.sandboxPool,
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
  IDS.sandboxUrl,
  IDS.sandboxPool,
]

export default function EvalConfigForm({ onSubmit, disabled, draft, onChange, toolbar, onValidateConnection }: Props) {
  const { t } = useLocale()
  const {
    datasets,
    limit,
    batchSize: evalBatchSize,
    repeats,
    timeout,
    stream,
    temperature,
    topP,
    maxTokens,
    topK,
    datasetArgs,
    sandboxEnabled,
    sandboxEngine,
    sandboxUrl,
    sandboxImage,
    sandboxPool: sandboxPoolSize,
  } = draft
  const changeField = <K extends keyof EvalDraft>(field: K, value: EvalDraft[K]) => {
    onChange({ ...draft, [field]: value })
    clearErr(`eval-${field}`)
  }
  const setDatasets = (value: string) => changeField('datasets', value)
  const setLimit = (value: string) => changeField('limit', value)
  const setEvalBatchSize = (value: string) => changeField('batchSize', value)
  const setRepeats = (value: string) => changeField('repeats', value)
  const setTimeout_ = (value: string) => changeField('timeout', value)
  const setStream = (value: boolean) => changeField('stream', value)
  const setTemperature = (value: string) => changeField('temperature', value)
  const setTopP = (value: string) => changeField('topP', value)
  const setMaxTokens = (value: string) => changeField('maxTokens', value)
  const setTopK = (value: string) => changeField('topK', value)
  const setDatasetArgs = (value: string) => changeField('datasetArgs', value)
  const setSandboxEnabled = (value: boolean) => changeField('sandboxEnabled', value)
  const setSandboxEngine = (value: string) => {
    if (value === 'docker' || value === 'volcengine') changeField('sandboxEngine', value)
  }
  const setSandboxUrl = (value: string) => changeField('sandboxUrl', value)
  const setSandboxImage = (value: string) => changeField('sandboxImage', value)
  const setSandboxPoolSize = (value: string) => changeField('sandboxPool', value)

  const {
    errorFor: errMsg,
    clearError: clearErr,
    showMore,
    toggleMore,
    submitHandler,
  } = useTaskForm({
    domOrder: DOM_ORDER,
    moreParamsIds: MORE_PARAMS_IDS,
    values: Object.fromEntries(Object.entries(draft).map(([field, value]) => [`eval-${field}`, value])),
  })

  const validate = () => ({
    ...onValidateConnection(),
    ...Object.fromEntries(Object.entries(validateEvalDraft(draft)).map(([field, error]) => [`eval-${field}`, error])),
  })
  const buildConfig = () => onSubmit(buildEvalPayload(draft))

  // Dataset autocomplete
  const [showSuggestions, setShowSuggestions] = useState(false)
  const [filteredSuggestions, setFilteredSuggestions] = useState<string[]>([])
  const [activeSuggestion, setActiveSuggestion] = useState(-1)
  const datasetInputRef = useRef<HTMLDivElement>(null)
  const datasetListboxId = `${useId()}-dataset-listbox`

  // Dataset autocomplete suggestions. A failure here only costs the suggestions,
  // so the field stays usable and no error is surfaced.
  const benchmarks = useAsyncResource(async (signal) => {
    const res = await listBenchmarks(undefined, undefined, signal)
    return [...(res.text ?? []).map((b) => b.name), ...(res.multimodal ?? []).map((b) => b.name)]
  }, [])
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
          <input
            {...aria}
            type="number"
            min={1}
            value={repeats}
            onChange={(e) => {
              setRepeats(e.target.value)
              clearErr(IDS.repeats)
            }}
            className={inputClass(errMsg(IDS.repeats))}
          />
        )}
      </Field>
      <Field id={IDS.timeout} name="timeout" labelKey="eval.timeout" error={errMsg(IDS.timeout)}>
        {(aria) => (
          <input
            {...aria}
            type="number"
            min={0}
            step="any"
            value={timeout}
            onChange={(e) => {
              setTimeout_(e.target.value)
              clearErr(IDS.timeout)
            }}
            className={inputClass(errMsg(IDS.timeout))}
          />
        )}
      </Field>
      <Field id={IDS.stream} name="stream" labelKey="eval.stream">
        {(aria) => <Switch {...aria} checked={stream} onCheckedChange={setStream} />}
      </Field>
      <Field id={IDS.temperature} name="temperature" labelKey="eval.temperature" error={errMsg(IDS.temperature)}>
        {(aria) => (
          <input
            {...aria}
            type="number"
            min={0}
            max={2}
            step="any"
            value={temperature}
            onChange={(e) => {
              setTemperature(e.target.value)
              clearErr(IDS.temperature)
            }}
            className={inputClass(errMsg(IDS.temperature))}
          />
        )}
      </Field>
      <Field id={IDS.topP} name="top_p" labelKey="eval.topP" error={errMsg(IDS.topP)}>
        {(aria) => (
          <input
            {...aria}
            type="number"
            min={0}
            max={1}
            step="any"
            value={topP}
            onChange={(e) => {
              setTopP(e.target.value)
              clearErr(IDS.topP)
            }}
            className={inputClass(errMsg(IDS.topP))}
          />
        )}
      </Field>
      <Field id={IDS.maxTokens} name="max_tokens" labelKey="eval.maxTokens" error={errMsg(IDS.maxTokens)}>
        {(aria) => (
          <input
            {...aria}
            type="number"
            min={1}
            value={maxTokens}
            onChange={(e) => {
              setMaxTokens(e.target.value)
              clearErr(IDS.maxTokens)
            }}
            className={inputClass(errMsg(IDS.maxTokens))}
          />
        )}
      </Field>
      <Field id={IDS.topK} name="top_k" labelKey="eval.topK" error={errMsg(IDS.topK)} hintKey="eval.topKHint">
        {(aria) => (
          <input
            {...aria}
            type="number"
            step={1}
            value={topK}
            onChange={(e) => {
              setTopK(e.target.value)
              clearErr(IDS.topK)
            }}
            className={inputClass(errMsg(IDS.topK))}
          />
        )}
      </Field>
      <Field
        id={IDS.datasetArgs}
        name="dataset_args"
        labelKey="eval.datasetArgs"
        error={errMsg(IDS.datasetArgs)}
        className="md:col-span-2"
      >
        {(aria) => (
          <textarea
            {...aria}
            value={datasetArgs}
            onChange={(e) => {
              setDatasetArgs(e.target.value)
              clearErr(IDS.datasetArgs)
            }}
            className={inputClass(errMsg(IDS.datasetArgs), 'h-20 resize-y')}
            style={{ fontFamily: 'var(--font-mono)' }}
            placeholder='{"gsm8k": {"few_shot_num": 4}}'
          />
        )}
      </Field>
      <hr className="md:col-span-2 my-2" />
      <h3 className="md:col-span-2 font-semibold">{t('eval.sandbox')}</h3>
      <Field id={IDS.sandboxEnable} name="sandbox_enabled" labelKey="eval.sandboxEnable">
        {(aria) => <Switch {...aria} checked={sandboxEnabled} onCheckedChange={setSandboxEnabled} />}
      </Field>
      {sandboxEnabled && (
        <>
          <Field id={IDS.sandboxEngine} name="sandbox_engine" labelKey="eval.sandboxEngine" hintKey="eval.sandboxHint">
            {(aria) => (
              <select
                {...aria}
                value={sandboxEngine}
                onChange={(e) => setSandboxEngine(e.target.value)}
                className={inputClass()}
              >
                <option value="docker">docker</option>
                <option value="volcengine">volcengine</option>
              </select>
            )}
          </Field>
          {sandboxEngine === 'docker' && (
            <>
              <Field id={IDS.sandboxUrl} name="sandbox_url" labelKey="eval.sandboxUrl" error={errMsg(IDS.sandboxUrl)}>
                {(aria) => (
                  <input
                    {...aria}
                    value={sandboxUrl}
                    onChange={(e) => {
                      setSandboxUrl(e.target.value)
                      clearErr(IDS.sandboxUrl)
                    }}
                    className={inputClass(errMsg(IDS.sandboxUrl))}
                  />
                )}
              </Field>
              <Field id={IDS.sandboxImage} name="sandbox_image" labelKey="eval.sandboxImage">
                {(aria) => (
                  <input
                    {...aria}
                    value={sandboxImage}
                    onChange={(e) => setSandboxImage(e.target.value)}
                    className={inputClass()}
                  />
                )}
              </Field>
            </>
          )}
          <Field id={IDS.sandboxPool} name="sandbox_pool" labelKey="eval.sandboxPool" error={errMsg(IDS.sandboxPool)}>
            {(aria) => (
              <input
                {...aria}
                type="number"
                min={1}
                step={1}
                value={sandboxPoolSize}
                onChange={(e) => {
                  setSandboxPoolSize(e.target.value)
                  clearErr(IDS.sandboxPool)
                }}
                className={inputClass(errMsg(IDS.sandboxPool))}
              />
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
      moreParamsColumns={2}
      toolbar={toolbar}
    >
      {/* Datasets with autocomplete */}
      <Field
        id={IDS.datasets}
        name="datasets"
        labelKey="eval.datasets"
        required
        error={errMsg(IDS.datasets)}
        autoComplete="off"
        className="relative"
      >
        {(aria) => (
          <div ref={datasetInputRef}>
            <input
              {...aria}
              type="text"
              value={datasets}
              onChange={(e) => handleDatasetChange(e.target.value)}
              onFocus={() => {
                if (filteredSuggestions.length) setShowSuggestions(true)
              }}
              onKeyDown={handleDatasetKeyDown}
              role="combobox"
              aria-autocomplete="list"
              aria-expanded={showSuggestions}
              aria-controls={datasetListboxId}
              aria-activedescendant={
                showSuggestions && activeSuggestion >= 0 ? `${datasetListboxId}-option-${activeSuggestion}` : undefined
              }
              className={inputClass(errMsg(IDS.datasets))}
              placeholder="gsm8k, arc"
            />
            {showSuggestions && (
              <div
                id={datasetListboxId}
                role="listbox"
                className="absolute z-50 left-0 right-0 mt-1 rounded-[var(--radius-sm)] border border-[var(--border-md)] bg-[var(--bg-card)] shadow-[var(--shadow)] overflow-hidden max-h-48 overflow-y-auto"
              >
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

      <Field id={IDS.limit} name="limit" labelKey="eval.limit" error={errMsg(IDS.limit)} hintKey="eval.limitHint">
        {(aria) => (
          <input
            {...aria}
            type="number"
            min={0}
            step="any"
            value={limit}
            onChange={(e) => {
              setLimit(e.target.value)
              clearErr(IDS.limit)
            }}
            className={inputClass(errMsg(IDS.limit))}
          />
        )}
      </Field>

      <Field id={IDS.batchSize} name="eval_batch_size" labelKey="eval.batchSize" error={errMsg(IDS.batchSize)}>
        {(aria) => (
          <input
            {...aria}
            type="number"
            min={1}
            value={evalBatchSize}
            onChange={(e) => {
              setEvalBatchSize(e.target.value)
              clearErr(IDS.batchSize)
            }}
            className={inputClass(errMsg(IDS.batchSize))}
          />
        )}
      </Field>
    </TaskFormShell>
  )
}
