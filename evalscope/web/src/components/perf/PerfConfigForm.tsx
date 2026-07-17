import { useState, type SyntheticEvent } from 'react'
import { useLocale } from '@/contexts/LocaleContext'
import Button from '@/components/ui/Button'
import Field from '@/components/ui/Field'
import { FORM_INPUT_CLASS, inputClass } from '@/components/ui/formStyles'
import { computeFirstInvalid, validateNumeric, FORM_MESSAGE_KEYS } from '@/domain/form/validation'
import { useFormErrors } from '@/hooks/useFormErrors'
import { ApiKeyField, ApiUrlField, ModelField } from '@/components/tasks/TaskFormFields'

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
]

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

  const { setErrors, errorFor: errMsg, clearError: clearErr } = useFormErrors()

  const handleSubmit = (e: SyntheticEvent<HTMLFormElement>) => {
    e.preventDefault()
    const newErrors: Record<string, string> = {}

    // Required text fields.
    if (!model.trim()) newErrors[IDS.model] = FORM_MESSAGE_KEYS.required

    const positiveIntegerListChecks = [
      { id: IDS.parallel, value: parallel },
      { id: IDS.number, value: number },
    ]
    for (const check of positiveIntegerListChecks) {
      const values = check.value.split(',').map((part) => part.trim())
      if (values.length === 0 || values.some((value) => !/^\d+$/.test(value) || Number(value) < 1)) {
        newErrors[check.id] = FORM_MESSAGE_KEYS.numericBelowMin
      }
    }

    // Numeric fields with min constraints. Empty optional fields are
    // skipped; non-empty values are validated.
    const numericChecks: Array<{ id: string; value: string; min?: number; max?: number; step?: number }> = [
      { id: IDS.rate, value: rate, min: 0 },
      { id: IDS.maxTokens, value: maxTokens, min: 1 },
      { id: IDS.minTokens, value: minTokens, min: 0 },
      { id: IDS.maxPromptLen, value: maxPromptLen, min: 0 },
      { id: IDS.minPromptLen, value: minPromptLen, min: 0 },
    ]
    for (const check of numericChecks) {
      if (check.value.trim() === '') continue
      const err = validateNumeric(Number(check.value), check.min, check.max, check.step)
      if (err) newErrors[check.id] = err.messageKey
    }

    if (Object.keys(newErrors).length > 0) {
      setErrors(newErrors)
      // Move focus to the first invalid field in DOM order.
      const firstInvalid = computeFirstInvalid(DOM_ORDER, Object.keys(newErrors))
      if (firstInvalid) {
        requestAnimationFrame(() => document.getElementById(firstInvalid)?.focus())
      }
      return
    }
    setErrors({})

    const config: Record<string, unknown> = {
      model,
      api,
      parallel: parallel.split(',').map((s) => Number(s.trim())).filter(Boolean),
      number: number.split(',').map((s) => Number(s.trim())).filter(Boolean),
    }
    if (url) config.url = url
    if (apiKey) config.api_key = apiKey
    if (rate) config.rate = Number(rate)
    if (maxTokens) config.max_tokens = Number(maxTokens)
    if (minTokens) config.min_tokens = Number(minTokens)
    if (dataset) config.dataset = dataset
    if (maxPromptLen) config.max_prompt_length = Number(maxPromptLen)
    if (minPromptLen) config.min_prompt_length = Number(minPromptLen)
    onSubmit(config)
  }

  return (
    <form onSubmit={handleSubmit} className="space-y-4" noValidate>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <ModelField
          id={IDS.model}
          value={model}
          error={errMsg(IDS.model)}
          onChange={(value) => { setModel(value); clearErr(IDS.model) }}
        />

        <Field id={IDS.api} name="api" labelKey="perf.apiType">
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

        <Field id={IDS.parallel} name="parallel" labelKey="perf.parallel" error={errMsg(IDS.parallel)}>
          {(aria) => (
            <input {...aria} type="text" inputMode="numeric" value={parallel} onChange={(e) => { setParallel(e.target.value); clearErr(IDS.parallel) }} className={inputClass(errMsg(IDS.parallel))} placeholder="1, 4, 8" />
          )}
        </Field>

        <Field id={IDS.number} name="number" labelKey="perf.number" error={errMsg(IDS.number)}>
          {(aria) => (
            <input {...aria} type="text" inputMode="numeric" value={number} onChange={(e) => { setNumber(e.target.value); clearErr(IDS.number) }} className={inputClass(errMsg(IDS.number))} placeholder="10, 100" />
          )}
        </Field>

        <Field id={IDS.rate} name="rate" labelKey="perf.rate" error={errMsg(IDS.rate)}>
          {(aria) => (
            <input {...aria} type="number" min={0} value={rate} onChange={(e) => { setRate(e.target.value); clearErr(IDS.rate) }} className={inputClass(errMsg(IDS.rate))} />
          )}
        </Field>

        <Field id={IDS.maxTokens} name="max_tokens" labelKey="perf.maxTokens" error={errMsg(IDS.maxTokens)}>
          {(aria) => (
            <input {...aria} type="number" min={1} value={maxTokens} onChange={(e) => { setMaxTokens(e.target.value); clearErr(IDS.maxTokens) }} className={inputClass(errMsg(IDS.maxTokens))} />
          )}
        </Field>

        <Field id={IDS.minTokens} name="min_tokens" labelKey="perf.minTokens" error={errMsg(IDS.minTokens)}>
          {(aria) => (
            <input {...aria} type="number" min={0} value={minTokens} onChange={(e) => { setMinTokens(e.target.value); clearErr(IDS.minTokens) }} className={inputClass(errMsg(IDS.minTokens))} />
          )}
        </Field>

        <Field id={IDS.dataset} name="dataset" labelKey="perf.dataset">
          {(aria) => (
            <input {...aria} type="text" value={dataset} onChange={(e) => setDataset(e.target.value)} className={FORM_INPUT_CLASS} placeholder="openqa" />
          )}
        </Field>

        <Field id={IDS.maxPromptLen} name="max_prompt_length" labelKey="perf.maxPromptLen" error={errMsg(IDS.maxPromptLen)}>
          {(aria) => (
            <input {...aria} type="number" min={0} value={maxPromptLen} onChange={(e) => { setMaxPromptLen(e.target.value); clearErr(IDS.maxPromptLen) }} className={inputClass(errMsg(IDS.maxPromptLen))} />
          )}
        </Field>

        <Field id={IDS.minPromptLen} name="min_prompt_length" labelKey="perf.minPromptLen" error={errMsg(IDS.minPromptLen)}>
          {(aria) => (
            <input {...aria} type="number" min={0} value={minPromptLen} onChange={(e) => { setMinPromptLen(e.target.value); clearErr(IDS.minPromptLen) }} className={inputClass(errMsg(IDS.minPromptLen))} />
          )}
        </Field>
      </div>

      <Button type="submit" variant="primary" disabled={disabled} className="btn-glow">
        {t('perf.startPerf')}
      </Button>
    </form>
  )
}
