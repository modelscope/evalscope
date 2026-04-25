import { useState } from 'react'
import { useLocale } from '@/contexts/LocaleContext'
import { ChevronDown, ChevronUp } from 'lucide-react'

interface Props {
  onSubmit: (config: Record<string, unknown>) => void
  disabled?: boolean
}

export default function EvalConfigForm({ onSubmit, disabled }: Props) {
  const { t } = useLocale()
  const [model, setModel] = useState('')
  const [apiUrl, setApiUrl] = useState('')
  const [apiKey, setApiKey] = useState('')
  const [datasets, setDatasets] = useState('')
  const [limit, setLimit] = useState('5')
  const [batchSize, setBatchSize] = useState('16')
  const [showMore, setShowMore] = useState(false)
  const [repeats, setRepeats] = useState('1')
  const [timeout, setTimeout_] = useState('60')
  const [stream, setStream] = useState(false)
  const [temperature, setTemperature] = useState('')
  const [topP, setTopP] = useState('')
  const [maxTokens, setMaxTokens] = useState('')
  const [topK, setTopK] = useState('')
  const [datasetArgs, setDatasetArgs] = useState('')

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    const config: Record<string, unknown> = {
      model,
      datasets: datasets.split(',').map((s) => s.trim()).filter(Boolean),
      limit: limit ? Number(limit) : undefined,
      batch_size: batchSize ? Number(batchSize) : undefined,
    }
    if (apiUrl) config.api_url = apiUrl
    if (apiKey) config.api_key = apiKey
    if (repeats && Number(repeats) > 1) config.repeats = Number(repeats)
    if (timeout) config.timeout = Number(timeout)
    if (stream) config.stream = true
    if (temperature) config.temperature = Number(temperature)
    if (topP) config.top_p = Number(topP)
    if (maxTokens) config.max_tokens = Number(maxTokens)
    if (topK) config.top_k = Number(topK)
    if (datasetArgs) {
      try {
        config.dataset_args = JSON.parse(datasetArgs)
      } catch { /* ignore invalid JSON */ }
    }
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
          <label className={labelClass}>{t('eval.datasets')} *</label>
          <input value={datasets} onChange={(e) => setDatasets(e.target.value)} className={inputClass} required placeholder="gsm8k, arc" />
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
          <label className={labelClass}>{t('eval.limit')}</label>
          <input type="number" value={limit} onChange={(e) => setLimit(e.target.value)} className={inputClass} />
        </div>
        <div>
          <label className={labelClass}>{t('eval.batchSize')}</label>
          <input type="number" value={batchSize} onChange={(e) => setBatchSize(e.target.value)} className={inputClass} />
        </div>
      </div>

      {/* More params toggle */}
      <button
        type="button"
        onClick={() => setShowMore(!showMore)}
        className="flex items-center gap-1 text-xs text-[var(--color-primary)] hover:underline"
      >
        {t('eval.moreParams')}
        {showMore ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
      </button>

      {showMore && (
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 pt-2 border-t border-[var(--color-border)]">
          <div>
            <label className={labelClass}>{t('eval.repeats')}</label>
            <input type="number" value={repeats} onChange={(e) => setRepeats(e.target.value)} className={inputClass} />
          </div>
          <div>
            <label className={labelClass}>{t('eval.timeout')}</label>
            <input type="number" value={timeout} onChange={(e) => setTimeout_(e.target.value)} className={inputClass} />
          </div>
          <div className="flex items-end gap-2 pb-0.5">
            <label className="flex items-center gap-1.5 text-xs text-[var(--color-ink-muted)]">
              <input type="checkbox" checked={stream} onChange={(e) => setStream(e.target.checked)} className="accent-[var(--color-primary)]" />
              {t('eval.stream')}
            </label>
          </div>
          <div>
            <label className={labelClass}>{t('eval.temperature')}</label>
            <input type="number" step="0.1" value={temperature} onChange={(e) => setTemperature(e.target.value)} className={inputClass} />
          </div>
          <div>
            <label className={labelClass}>{t('eval.topP')}</label>
            <input type="number" step="0.1" value={topP} onChange={(e) => setTopP(e.target.value)} className={inputClass} />
          </div>
          <div>
            <label className={labelClass}>{t('eval.maxTokens')}</label>
            <input type="number" value={maxTokens} onChange={(e) => setMaxTokens(e.target.value)} className={inputClass} />
          </div>
          <div>
            <label className={labelClass}>{t('eval.topK')}</label>
            <input type="number" value={topK} onChange={(e) => setTopK(e.target.value)} className={inputClass} />
          </div>
          <div className="md:col-span-2">
            <label className={labelClass}>{t('eval.datasetArgs')}</label>
            <textarea
              value={datasetArgs}
              onChange={(e) => setDatasetArgs(e.target.value)}
              className={`${inputClass} h-20 resize-y font-mono`}
              placeholder='{"gsm8k": {"few_shot_num": 4}}'
            />
          </div>
        </div>
      )}

      <button
        type="submit"
        disabled={disabled || !model || !datasets}
        className="px-4 py-2 text-sm font-medium rounded-md bg-[var(--color-primary)] text-white disabled:opacity-50 hover:opacity-90 transition-opacity"
      >
        {t('eval.startEval')}
      </button>
    </form>
  )
}
