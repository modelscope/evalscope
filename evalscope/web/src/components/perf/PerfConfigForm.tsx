import { useState, type FormEvent } from 'react'
import { useLocale } from '@/contexts/LocaleContext'
import Button from '@/components/ui/Button'

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

  const [errors, setErrors] = useState<Record<string, string>>({})

  const handleSubmit = (e: FormEvent) => {
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

  const inputStyle =
    'w-full px-3 py-2 text-sm rounded-[var(--radius-sm)] bg-[var(--bg-deep)] border border-[var(--border)] text-[var(--text)] placeholder:text-[var(--text-dim)] focus:outline-none focus:border-[var(--accent)] focus:ring-1 focus:ring-[var(--accent-dim)] transition-all'
  const labelStyle = 'block text-xs font-medium text-[var(--text-muted)] mb-1'
  const errorInputStyle = 'border-[var(--danger)] focus:border-[var(--danger)] focus:ring-[var(--danger-bg)]'

  return (
    <form onSubmit={handleSubmit} className="space-y-4">
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* Model Name */}
        <div>
          <label className={labelStyle}>
            {t('eval.modelName')} <span className="text-[var(--danger)]">*</span>
          </label>
          <input
            value={model}
            onChange={(e) => { setModel(e.target.value); if (errors.model) setErrors((p) => ({ ...p, model: '' })) }}
            className={`${inputStyle} ${errors.model ? errorInputStyle : ''}`}
            placeholder="Qwen/Qwen2.5-0.5B-Instruct"
          />
          {errors.model && <p className="text-xs text-[var(--danger)] mt-1">{errors.model}</p>}
        </div>

        {/* API Type */}
        <div>
          <label className={labelStyle}>{t('perf.apiType')}</label>
          <select value={apiType} onChange={(e) => setApiType(e.target.value)} className={inputStyle}>
            <option value="openai">OpenAI</option>
            <option value="dashscope">DashScope</option>
            <option value="local">Local</option>
          </select>
        </div>

        {/* API URL */}
        <div>
          <label className={labelStyle}>{t('eval.apiUrl')}</label>
          <input value={apiUrl} onChange={(e) => setApiUrl(e.target.value)} className={inputStyle} placeholder="http://localhost:8000/v1" />
        </div>

        {/* API Key */}
        <div>
          <label className={labelStyle}>{t('eval.apiKey')}</label>
          <input type="password" value={apiKey} onChange={(e) => setApiKey(e.target.value)} className={inputStyle} placeholder="sk-..." />
        </div>

        {/* Parallel */}
        <div>
          <label className={labelStyle}>{t('perf.parallel')}</label>
          <input value={parallel} onChange={(e) => setParallel(e.target.value)} className={inputStyle} placeholder="1, 4, 8" />
        </div>

        {/* Number */}
        <div>
          <label className={labelStyle}>{t('perf.number')}</label>
          <input value={number} onChange={(e) => setNumber(e.target.value)} className={inputStyle} placeholder="10, 100" />
        </div>

        {/* Rate */}
        <div>
          <label className={labelStyle}>{t('perf.rate')}</label>
          <input type="number" value={rate} onChange={(e) => setRate(e.target.value)} className={inputStyle} />
        </div>

        {/* Max Tokens */}
        <div>
          <label className={labelStyle}>{t('perf.maxTokens')}</label>
          <input type="number" value={maxTokens} onChange={(e) => setMaxTokens(e.target.value)} className={inputStyle} />
        </div>

        {/* Min Tokens */}
        <div>
          <label className={labelStyle}>{t('perf.minTokens')}</label>
          <input type="number" value={minTokens} onChange={(e) => setMinTokens(e.target.value)} className={inputStyle} />
        </div>

        {/* Dataset */}
        <div>
          <label className={labelStyle}>{t('perf.dataset')}</label>
          <input value={dataset} onChange={(e) => setDataset(e.target.value)} className={inputStyle} placeholder="openqa" />
        </div>

        {/* Max Prompt Length */}
        <div>
          <label className={labelStyle}>{t('perf.maxPromptLen')}</label>
          <input type="number" value={maxPromptLen} onChange={(e) => setMaxPromptLen(e.target.value)} className={inputStyle} />
        </div>

        {/* Min Prompt Length */}
        <div>
          <label className={labelStyle}>{t('perf.minPromptLen')}</label>
          <input type="number" value={minPromptLen} onChange={(e) => setMinPromptLen(e.target.value)} className={inputStyle} />
        </div>
      </div>

      <Button type="submit" variant="primary" disabled={disabled} className="btn-glow">
        {t('perf.startPerf')}
      </Button>
    </form>
  )
}
