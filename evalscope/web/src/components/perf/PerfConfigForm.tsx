import { useState, type SyntheticEvent } from 'react'
import { useLocale } from '@/contexts/LocaleContext'
import Button from '@/components/ui/Button'
import FormField from '@/components/ui/FormField'
import { FORM_INPUT_CLASS, inputClass } from '@/components/ui/formStyles'

interface Props {
  onSubmit: (config: Record<string, unknown>) => void
  disabled?: boolean
}

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

  const [errors, setErrors] = useState<Record<string, string>>({})

  const handleSubmit = (e: SyntheticEvent<HTMLFormElement>) => {
    e.preventDefault()
    const newErrors: Record<string, string> = {}
    if (!model.trim()) newErrors.model = 'Required'
    if (Object.keys(newErrors).length > 0) {
      setErrors(newErrors)
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
    <form onSubmit={handleSubmit} className="space-y-4">
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <FormField label={t('eval.modelName')} required error={errors.model}>
          <input
            value={model}
            onChange={(e) => { setModel(e.target.value); if (errors.model) setErrors((p) => ({ ...p, model: '' })) }}
            className={inputClass(errors.model)}
            placeholder="Qwen/Qwen2.5-0.5B-Instruct"
          />
        </FormField>

        <FormField label={t('perf.apiType')}>
          <select value={api} onChange={(e) => setApi(e.target.value)} className={FORM_INPUT_CLASS}>
            <option value="openai">OpenAI</option>
            <option value="dashscope">DashScope</option>
            <option value="local">Local</option>
          </select>
        </FormField>

        <FormField label={t('eval.apiUrl')}>
          <input value={url} onChange={(e) => setUrl(e.target.value)} className={FORM_INPUT_CLASS} placeholder="http://localhost:8000/v1" />
        </FormField>

        <FormField label={t('eval.apiKey')}>
          <input type="password" value={apiKey} onChange={(e) => setApiKey(e.target.value)} className={FORM_INPUT_CLASS} placeholder="sk-..." />
        </FormField>

        <FormField label={t('perf.parallel')}>
          <input value={parallel} onChange={(e) => setParallel(e.target.value)} className={FORM_INPUT_CLASS} placeholder="1, 4, 8" />
        </FormField>

        <FormField label={t('perf.number')}>
          <input value={number} onChange={(e) => setNumber(e.target.value)} className={FORM_INPUT_CLASS} placeholder="10, 100" />
        </FormField>

        <FormField label={t('perf.rate')}>
          <input type="number" value={rate} onChange={(e) => setRate(e.target.value)} className={FORM_INPUT_CLASS} />
        </FormField>

        <FormField label={t('perf.maxTokens')}>
          <input type="number" value={maxTokens} onChange={(e) => setMaxTokens(e.target.value)} className={FORM_INPUT_CLASS} />
        </FormField>

        <FormField label={t('perf.minTokens')}>
          <input type="number" value={minTokens} onChange={(e) => setMinTokens(e.target.value)} className={FORM_INPUT_CLASS} />
        </FormField>

        <FormField label={t('perf.dataset')}>
          <input value={dataset} onChange={(e) => setDataset(e.target.value)} className={FORM_INPUT_CLASS} placeholder="openqa" />
        </FormField>

        <FormField label={t('perf.maxPromptLen')}>
          <input type="number" value={maxPromptLen} onChange={(e) => setMaxPromptLen(e.target.value)} className={FORM_INPUT_CLASS} />
        </FormField>

        <FormField label={t('perf.minPromptLen')}>
          <input type="number" value={minPromptLen} onChange={(e) => setMinPromptLen(e.target.value)} className={FORM_INPUT_CLASS} />
        </FormField>
      </div>

      <Button type="submit" variant="primary" disabled={disabled} className="btn-glow">
        {t('perf.startPerf')}
      </Button>
    </form>
  )
}
