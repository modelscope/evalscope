import { useEffect, useRef, useState, type SyntheticEvent } from 'react'
import { useLocale } from '@/contexts/LocaleContext'
import { listBenchmarks } from '@/api/eval'
import Button from '@/components/ui/Button'
import Card from '@/components/ui/Card'
import FormField from '@/components/ui/FormField'
import { FORM_INPUT_CLASS, FORM_LABEL_CLASS, inputClass } from '@/components/ui/formStyles'
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
  const [evalBatchSize, setEvalBatchSize] = useState('16')
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

  const handleSubmit = (e: SyntheticEvent<HTMLFormElement>) => {
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
    if (datasetArgs) {
      try {
        config.dataset_args = JSON.parse(datasetArgs)
      } catch { /* ignore invalid JSON */ }
    }
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

        {/* Datasets with autocomplete */}
        <FormField label={t('eval.datasets')} required error={errors.datasets} className="relative">
          <div ref={datasetInputRef}>
            <input
              value={datasets}
              onChange={(e) => handleDatasetChange(e.target.value)}
              onFocus={() => { if (filteredSuggestions.length) setShowSuggestions(true) }}
              className={inputClass(errors.datasets)}
              placeholder="gsm8k, arc"
            />
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
        </FormField>

        <FormField label={t('eval.apiUrl')}>
          <input value={apiUrl} onChange={(e) => setApiUrl(e.target.value)} className={FORM_INPUT_CLASS} placeholder="http://localhost:8000/v1" />
        </FormField>

        <FormField label={t('eval.apiKey')}>
          <input type="password" value={apiKey} onChange={(e) => setApiKey(e.target.value)} className={FORM_INPUT_CLASS} placeholder="sk-..." />
        </FormField>

        <FormField label={t('eval.limit')}>
          <input type="number" value={limit} onChange={(e) => setLimit(e.target.value)} className={FORM_INPUT_CLASS} />
        </FormField>

        <FormField label={t('eval.batchSize')}>
          <input type="number" value={evalBatchSize} onChange={(e) => setEvalBatchSize(e.target.value)} className={FORM_INPUT_CLASS} />
        </FormField>
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
            <FormField label={t('eval.repeats')}>
              <input type="number" value={repeats} onChange={(e) => setRepeats(e.target.value)} className={FORM_INPUT_CLASS} />
            </FormField>
            <FormField label={t('eval.timeout')}>
              <input type="number" value={timeout} onChange={(e) => setTimeout_(e.target.value)} className={FORM_INPUT_CLASS} />
            </FormField>
            <div className="flex items-end gap-2 pb-0.5">
              <label className="flex items-center gap-1.5 text-xs text-[var(--text-muted)] cursor-pointer">
                <input type="checkbox" checked={stream} onChange={(e) => setStream(e.target.checked)} className="accent-[var(--accent)]" />
                {t('eval.stream')}
              </label>
            </div>
            <FormField label={t('eval.temperature')}>
              <input type="number" step="0.1" value={temperature} onChange={(e) => setTemperature(e.target.value)} className={FORM_INPUT_CLASS} />
            </FormField>
            <FormField label={t('eval.topP')}>
              <input type="number" step="0.1" value={topP} onChange={(e) => setTopP(e.target.value)} className={FORM_INPUT_CLASS} />
            </FormField>
            <FormField label={t('eval.maxTokens')}>
              <input type="number" value={maxTokens} onChange={(e) => setMaxTokens(e.target.value)} className={FORM_INPUT_CLASS} />
            </FormField>
            <FormField label={t('eval.topK')}>
              <input type="number" value={topK} onChange={(e) => setTopK(e.target.value)} className={FORM_INPUT_CLASS} />
            </FormField>
            <div className="md:col-span-2">
              <label className={FORM_LABEL_CLASS}>{t('eval.datasetArgs')}</label>
              <textarea
                value={datasetArgs}
                onChange={(e) => setDatasetArgs(e.target.value)}
                className={`${FORM_INPUT_CLASS} h-20 resize-y`}
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
