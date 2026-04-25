import { useState } from 'react'
import { useLocale } from '@/contexts/LocaleContext'

interface Props {
  onSubmit: (config: Record<string, unknown>) => void
  disabled?: boolean
}

export default function PerfConfigForm({ onSubmit, disabled }: Props) {
  const { t } = useLocale()
  const [model, setModel] = useState('')
  const [apiUrl, setApiUrl] = useState('')
  const [apiKey, setApiKey] = useState('')
  const [apiType, setApiType] = useState('openai')
  const [parallel, setParallel] = useState('1')
  const [number, setNumber] = useState('10')
  const [rate, setRate] = useState('')
  const [maxTokens, setMaxTokens] = useState('512')
  const [minTokens, setMinTokens] = useState('')
  const [dataset, setDataset] = useState('')
  const [maxPromptLen, setMaxPromptLen] = useState('')
  const [minPromptLen, setMinPromptLen] = useState('')

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    const config: Record<string, unknown> = {
      model,
      api_type: apiType,
      parallel: parallel.split(',').map((s) => Number(s.trim())).filter(Boolean),
      number: number.split(',').map((s) => Number(s.trim())).filter(Boolean),
    }
    if (apiUrl) config.api_url = apiUrl
    if (apiKey) config.api_key = apiKey
    if (rate) config.rate = Number(rate)
    if (maxTokens) config.max_tokens = Number(maxTokens)
    if (minTokens) config.min_tokens = Number(minTokens)
    if (dataset) config.dataset = dataset
    if (maxPromptLen) config.max_prompt_length = Number(maxPromptLen)
    if (minPromptLen) config.min_prompt_length = Number(minPromptLen)
    onSubmit(config)
  }

  const inputClass =
    'w-full px-2.5 py-1.5 text-sm rounded-md bg-[var(--color-bg-page)] border border-[var(--color-border)] focus:outline-none focus:border-[var(--color-primary)]'
  const labelClass = 'block text-xs font-medium text-[var(--color-ink-muted)] mb-1'

  return (
    <form onSubmit={handleSubmit} className="space-y-4">
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div>
          <label className={labelClass}>{t('eval.modelName')} *</label>
          <input value={model} onChange={(e) => setModel(e.target.value)} className={inputClass} required placeholder="Qwen/Qwen2.5-0.5B-Instruct" />
        </div>
        <div>
          <label className={labelClass}>{t('perf.apiType')}</label>
          <select value={apiType} onChange={(e) => setApiType(e.target.value)} className={inputClass}>
            <option value="openai">OpenAI</option>
            <option value="dashscope">DashScope</option>
            <option value="local">Local</option>
          </select>
        </div>
        <div>
          <label className={labelClass}>{t('eval.apiUrl')}</label>
          <input value={apiUrl} onChange={(e) => setApiUrl(e.target.value)} className={inputClass} placeholder="http://localhost:8000/v1" />
        </div>
        <div>
          <label className={labelClass}>{t('eval.apiKey')}</label>
          <input type="password" value={apiKey} onChange={(e) => setApiKey(e.target.value)} className={inputClass} placeholder="sk-..." />
        </div>
        <div>
          <label className={labelClass}>{t('perf.parallel')}</label>
          <input value={parallel} onChange={(e) => setParallel(e.target.value)} className={inputClass} placeholder="1, 4, 8" />
        </div>
        <div>
          <label className={labelClass}>{t('perf.number')}</label>
          <input value={number} onChange={(e) => setNumber(e.target.value)} className={inputClass} placeholder="10, 100" />
        </div>
        <div>
          <label className={labelClass}>{t('perf.rate')}</label>
          <input type="number" value={rate} onChange={(e) => setRate(e.target.value)} className={inputClass} />
        </div>
        <div>
          <label className={labelClass}>{t('perf.maxTokens')}</label>
          <input type="number" value={maxTokens} onChange={(e) => setMaxTokens(e.target.value)} className={inputClass} />
        </div>
        <div>
          <label className={labelClass}>{t('perf.minTokens')}</label>
          <input type="number" value={minTokens} onChange={(e) => setMinTokens(e.target.value)} className={inputClass} />
        </div>
        <div>
          <label className={labelClass}>{t('perf.dataset')}</label>
          <input value={dataset} onChange={(e) => setDataset(e.target.value)} className={inputClass} placeholder="openqa" />
        </div>
        <div>
          <label className={labelClass}>{t('perf.maxPromptLen')}</label>
          <input type="number" value={maxPromptLen} onChange={(e) => setMaxPromptLen(e.target.value)} className={inputClass} />
        </div>
        <div>
          <label className={labelClass}>{t('perf.minPromptLen')}</label>
          <input type="number" value={minPromptLen} onChange={(e) => setMinPromptLen(e.target.value)} className={inputClass} />
        </div>
      </div>

      <button
        type="submit"
        disabled={disabled || !model}
        className="px-4 py-2 text-sm font-medium rounded-md bg-[var(--color-primary)] text-white disabled:opacity-50 hover:opacity-90 transition-opacity"
      >
        {t('perf.startPerf')}
      </button>
    </form>
  )
}
