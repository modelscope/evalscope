import { useEffect, useRef, useState, type FormEvent } from 'react'
import { useLocale } from '@/contexts/LocaleContext'
import { listBenchmarks } from '@/api/eval'
import Button from '@/components/ui/Button'
import Card from '@/components/ui/Card'
import { ChevronDown, ChevronUp } from 'lucide-react'

interface Props {
  onSubmit: (config: Record<string, unknown>) => void
  disabled?: boolean
  initialDataset?: string
}

export default function EvalConfigForm({ onSubmit, disabled, initialDataset }: Props) {
  const { t } = useLocale()
  const [model, setModel] = useState('')
  const [apiUrl, setApiUrl] = useState('')
  const [apiKey, setApiKey] = useState('')
  const [datasets, setDatasets] = useState(initialDataset ?? '')
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

  // Validation
  const [errors, setErrors] = useState<Record<string, string>>({})

  // Dataset autocomplete
  const [benchmarkNames, setBenchmarkNames] = useState<string[]>([])
  const [showSuggestions, setShowSuggestions] = useState(false)
  const [filteredSuggestions, setFilteredSuggestions] = useState<string[]>([])
  const datasetInputRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (initialDataset) setDatasets(initialDataset)
  }, [initialDataset])

  useEffect(() => {
    listBenchmarks()
      .then((res) => {
        const names = [
          ...(res.text ?? []).map((b) => b.name),
          ...(res.multimodal ?? []).map((b) => b.name),
        ]
        setBenchmarkNames(names)
      })
      .catch(() => {})
  }, [])

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
    } else {
      setShowSuggestions(false)
    }
    if (errors.datasets) setErrors((prev) => ({ ...prev, datasets: '' }))
  }

  const selectSuggestion = (name: string) => {
    const parts = datasets.split(',').map((s) => s.trim())
    parts[parts.length - 1] = name
    setDatasets(parts.join(', '))
    setShowSuggestions(false)
  }

  const handleSubmit = (e: FormEvent) => {
    e.preventDefault()
    const newErrors: Record<string, string> = {}
    if (!model.trim()) newErrors.model = 'Required'
    if (!datasets.trim()) newErrors.datasets = 'Required'
    if (Object.keys(newErrors).length > 0) {
      setErrors(newErrors)
      return
    }
    setErrors({})

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

        {/* Datasets with autocomplete */}
        <div ref={datasetInputRef} className="relative">
          <label className={labelStyle}>
            {t('eval.datasets')} <span className="text-[var(--danger)]">*</span>
          </label>
          <input
            value={datasets}
            onChange={(e) => handleDatasetChange(e.target.value)}
            onFocus={() => { if (filteredSuggestions.length) setShowSuggestions(true) }}
            className={`${inputStyle} ${errors.datasets ? errorInputStyle : ''}`}
            placeholder="gsm8k, arc"
          />
          {errors.datasets && <p className="text-xs text-[var(--danger)] mt-1">{errors.datasets}</p>}
          {showSuggestions && (
            <div className="absolute z-50 left-0 right-0 mt-1 rounded-[var(--radius-sm)] border border-[var(--border-md)] bg-[var(--bg-card)] shadow-[var(--shadow)] overflow-hidden max-h-48 overflow-y-auto">
              {filteredSuggestions.map((name) => (
                <button
                  key={name}
                  type="button"
                  onClick={() => selectSuggestion(name)}
                  className="w-full text-left px-3 py-2 text-sm text-[var(--text)] hover:bg-[var(--bg-card2)] transition-colors cursor-pointer"
                >
                  {name}
                </button>
              ))}
            </div>
          )}
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

        {/* Limit */}
        <div>
          <label className={labelStyle}>{t('eval.limit')}</label>
          <input type="number" value={limit} onChange={(e) => setLimit(e.target.value)} className={inputStyle} />
        </div>

        {/* Batch Size */}
        <div>
          <label className={labelStyle}>{t('eval.batchSize')}</label>
          <input type="number" value={batchSize} onChange={(e) => setBatchSize(e.target.value)} className={inputStyle} />
        </div>
      </div>

      {/* More params toggle */}
      <button
        type="button"
        onClick={() => setShowMore(!showMore)}
        className="flex items-center gap-1 text-xs text-[var(--accent)] hover:underline cursor-pointer"
      >
        {t('eval.moreParams')}
        {showMore ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
      </button>

      {showMore && (
        <Card className="!p-0">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 p-4">
            <div>
              <label className={labelStyle}>{t('eval.repeats')}</label>
              <input type="number" value={repeats} onChange={(e) => setRepeats(e.target.value)} className={inputStyle} />
            </div>
            <div>
              <label className={labelStyle}>{t('eval.timeout')}</label>
              <input type="number" value={timeout} onChange={(e) => setTimeout_(e.target.value)} className={inputStyle} />
            </div>
            <div className="flex items-end gap-2 pb-0.5">
              <label className="flex items-center gap-1.5 text-xs text-[var(--text-muted)] cursor-pointer">
                <input type="checkbox" checked={stream} onChange={(e) => setStream(e.target.checked)} className="accent-[var(--accent)]" />
                {t('eval.stream')}
              </label>
            </div>
            <div>
              <label className={labelStyle}>{t('eval.temperature')}</label>
              <input type="number" step="0.1" value={temperature} onChange={(e) => setTemperature(e.target.value)} className={inputStyle} />
            </div>
            <div>
              <label className={labelStyle}>{t('eval.topP')}</label>
              <input type="number" step="0.1" value={topP} onChange={(e) => setTopP(e.target.value)} className={inputStyle} />
            </div>
            <div>
              <label className={labelStyle}>{t('eval.maxTokens')}</label>
              <input type="number" value={maxTokens} onChange={(e) => setMaxTokens(e.target.value)} className={inputStyle} />
            </div>
            <div>
              <label className={labelStyle}>{t('eval.topK')}</label>
              <input type="number" value={topK} onChange={(e) => setTopK(e.target.value)} className={inputStyle} />
            </div>
            <div className="md:col-span-2">
              <label className={labelStyle}>{t('eval.datasetArgs')}</label>
              <textarea
                value={datasetArgs}
                onChange={(e) => setDatasetArgs(e.target.value)}
                className={`${inputStyle} h-20 resize-y`}
                style={{ fontFamily: 'var(--font-mono)' }}
                placeholder='{"gsm8k": {"few_shot_num": 4}}'
              />
            </div>
          </div>
        </Card>
      )}

      <Button type="submit" variant="primary" disabled={disabled} className="btn-glow">
        {t('eval.startEval')}
      </Button>
    </form>
  )
}
