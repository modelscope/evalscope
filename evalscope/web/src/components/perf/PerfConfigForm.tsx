import { useState } from 'react'
import { useLocale } from '@/contexts/LocaleContext'
import Field from '@/components/ui/Field'
import { FORM_INPUT_CLASS, inputClass } from '@/components/ui/formStyles'
import {
  collectNumericErrors,
  parseNumericList,
  validatePositiveIntegerList,
  FORM_MESSAGE_KEYS,
} from '@/domain/form/validation'
import { ApiKeyField, ApiUrlField, ModelField } from '@/components/tasks/TaskFormFields'
import TaskFormShell from '@/components/tasks/TaskFormShell'
import { useTaskForm } from '@/components/tasks/useTaskForm'

interface Props {
  onSubmit: (config: Record<string, unknown>) => void
  disabled?: boolean
}

/** Stable field ids, reused as label/error association targets and focus targets. */
const IDS = {
  model: 'perf-model',
  api: 'perf-api',
  url: 'perf-url',
  apiKey: 'perf-apiKey',
  parallel: 'perf-parallel',
  number: 'perf-number',
  rate: 'perf-rate',
  maxTokens: 'perf-maxTokens',
  minTokens: 'perf-minTokens',
  dataset: 'perf-dataset',
  maxPromptLen: 'perf-maxPromptLen',
  minPromptLen: 'perf-minPromptLen',
  tokenizerPath: 'perf-tokenizerPath',
} as const

/** DOM order of focusable fields, drives first-invalid focus on submit. */
const DOM_ORDER: string[] = [
  IDS.model,
  IDS.api,
  IDS.url,
  IDS.apiKey,
  IDS.parallel,
  IDS.number,
  IDS.rate,
  IDS.maxTokens,
  IDS.minTokens,
  IDS.dataset,
  IDS.maxPromptLen,
  IDS.minPromptLen,
  IDS.tokenizerPath,
]

/** Fields that live inside the collapsible "More Parameters" section. */
const MORE_PARAMS_IDS: string[] = [IDS.tokenizerPath]

export default function PerfConfigForm({ onSubmit, disabled }: Props) {
  const { t } = useLocale()
  const [model, setModel] = useState('')
  const [url, setUrl] = useState('')
  const [apiKey, setApiKey] = useState('')
  const [api, setApi] = useState('openai')
  const [parallel, setParallel] = useState('1')
  const [number, setNumber] = useState('10')
  const [rate, setRate] = useState('')
  const [maxTokens, setMaxTokens] = useState('512')
  const [minTokens, setMinTokens] = useState('')
  const [dataset, setDataset] = useState('')
  const [maxPromptLen, setMaxPromptLen] = useState('')
  const [minPromptLen, setMinPromptLen] = useState('')
  const [tokenizerPath, setTokenizerPath] = useState('')

  const { errorFor: errMsg, clearError: clearErr, showMore, toggleMore, submitHandler } = useTaskForm({
    domOrder: DOM_ORDER,
    moreParamsIds: MORE_PARAMS_IDS,
  })

  const validate = (): Record<string, string> => {
    const errors: Record<string, string> = {}

    if (!model.trim()) errors[IDS.model] = FORM_MESSAGE_KEYS.required

    // Sweep fields: each is a comma-separated list of positive integers.
    for (const check of [{ id: IDS.parallel, value: parallel }, { id: IDS.number, value: number }]) {
      const error = validatePositiveIntegerList(check.value)
      if (error) errors[check.id] = error
    }

    return {
      ...errors,
      ...collectNumericErrors([
        { id: IDS.rate, value: rate, min: 0 },
        { id: IDS.maxTokens, value: maxTokens, min: 1 },
        { id: IDS.minTokens, value: minTokens, min: 0 },
        { id: IDS.maxPromptLen, value: maxPromptLen, min: 0 },
        { id: IDS.minPromptLen, value: minPromptLen, min: 0 },
      ]),
    }
  }

  const buildConfig = () => {
    const config: Record<string, unknown> = {
      model,
      api,
      parallel: parseNumericList(parallel),
      number: parseNumericList(number),
    }
    if (url) config.url = url
    if (apiKey) config.api_key = apiKey
    if (rate) config.rate = Number(rate)
    if (maxTokens) config.max_tokens = Number(maxTokens)
    if (minTokens) config.min_tokens = Number(minTokens)
    if (dataset) config.dataset = dataset
    if (maxPromptLen) config.max_prompt_length = Number(maxPromptLen)
    if (minPromptLen) config.min_prompt_length = Number(minPromptLen)
    if (tokenizerPath) config.tokenizer_path = tokenizerPath
    onSubmit(config)
  }

  return (
    <TaskFormShell
      onSubmit={submitHandler(validate, buildConfig)}
      moreParamsColumns={2}
      moreParamsLabel={t('perf.task.moreParams')}
      showMore={showMore}
      onToggleMore={toggleMore}
      submitLabel={t('perf.task.startPerf')}
      disabled={disabled}
      moreParams={(
        <Field id={IDS.tokenizerPath} name="tokenizer_path" labelKey="perf.task.tokenizerPath">
          {(aria) => (
            <input {...aria} type="text" value={tokenizerPath} onChange={(e) => setTokenizerPath(e.target.value)} className={FORM_INPUT_CLASS} placeholder="/path/to/tokenizer" />
          )}
        </Field>
      )}
    >
      <ModelField
        id={IDS.model}
        value={model}
        error={errMsg(IDS.model)}
        onChange={(value) => { setModel(value); clearErr(IDS.model) }}
      />

      <Field id={IDS.api} name="api" labelKey="perf.task.apiType">
        {(aria) => (
          <select {...aria} value={api} onChange={(e) => setApi(e.target.value)} className={FORM_INPUT_CLASS}>
            <option value="openai">OpenAI</option>
            <option value="dashscope">DashScope</option>
            <option value="local">Local</option>
          </select>
        )}
      </Field>

      <ApiUrlField id={IDS.url} name="url" value={url} onChange={setUrl} />
      <ApiKeyField id={IDS.apiKey} value={apiKey} onChange={setApiKey} />

      <Field id={IDS.parallel} name="parallel" labelKey="perf.task.parallel" error={errMsg(IDS.parallel)}>
        {(aria) => (
          <input {...aria} type="text" inputMode="numeric" value={parallel} onChange={(e) => { setParallel(e.target.value); clearErr(IDS.parallel) }} className={inputClass(errMsg(IDS.parallel))} placeholder="1, 4, 8" />
        )}
      </Field>

      <Field id={IDS.number} name="number" labelKey="perf.task.number" error={errMsg(IDS.number)}>
        {(aria) => (
          <input {...aria} type="text" inputMode="numeric" value={number} onChange={(e) => { setNumber(e.target.value); clearErr(IDS.number) }} className={inputClass(errMsg(IDS.number))} placeholder="10, 100" />
        )}
      </Field>

      <Field id={IDS.rate} name="rate" labelKey="perf.task.rate" error={errMsg(IDS.rate)}>
        {(aria) => (
          <input {...aria} type="number" min={0} value={rate} onChange={(e) => { setRate(e.target.value); clearErr(IDS.rate) }} className={inputClass(errMsg(IDS.rate))} />
        )}
      </Field>

      <Field id={IDS.maxTokens} name="max_tokens" labelKey="perf.task.maxTokens" error={errMsg(IDS.maxTokens)}>
        {(aria) => (
          <input {...aria} type="number" min={1} value={maxTokens} onChange={(e) => { setMaxTokens(e.target.value); clearErr(IDS.maxTokens) }} className={inputClass(errMsg(IDS.maxTokens))} />
        )}
      </Field>

      <Field id={IDS.minTokens} name="min_tokens" labelKey="perf.task.minTokens" error={errMsg(IDS.minTokens)}>
        {(aria) => (
          <input {...aria} type="number" min={0} value={minTokens} onChange={(e) => { setMinTokens(e.target.value); clearErr(IDS.minTokens) }} className={inputClass(errMsg(IDS.minTokens))} />
        )}
      </Field>

      <Field id={IDS.dataset} name="dataset" labelKey="perf.task.dataset">
        {(aria) => (
          <input {...aria} type="text" value={dataset} onChange={(e) => setDataset(e.target.value)} className={FORM_INPUT_CLASS} placeholder="openqa" />
        )}
      </Field>

      <Field id={IDS.maxPromptLen} name="max_prompt_length" labelKey="perf.task.maxPromptLen" error={errMsg(IDS.maxPromptLen)}>
        {(aria) => (
          <input {...aria} type="number" min={0} value={maxPromptLen} onChange={(e) => { setMaxPromptLen(e.target.value); clearErr(IDS.maxPromptLen) }} className={inputClass(errMsg(IDS.maxPromptLen))} />
        )}
      </Field>

      <Field id={IDS.minPromptLen} name="min_prompt_length" labelKey="perf.task.minPromptLen" error={errMsg(IDS.minPromptLen)}>
        {(aria) => (
          <input {...aria} type="number" min={0} value={minPromptLen} onChange={(e) => { setMinPromptLen(e.target.value); clearErr(IDS.minPromptLen) }} className={inputClass(errMsg(IDS.minPromptLen))} />
        )}
      </Field>
    </TaskFormShell>
  )
}
