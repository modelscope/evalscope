import { useState, type FormEvent } from 'react'
import { runDiagnosis } from '@/api/reports'
import type { DiagnosisRunRequest } from '@/api/generated/contracts'
import type { DiagnosisResponse } from '@/api/types'
import { useLocale } from '@/contexts/LocaleContext'
import Button from '@/components/ui/Button'
import ErrorAlert from '@/components/ui/ErrorAlert'
import Field from '@/components/ui/Field'
import { FORM_INPUT_CLASS } from '@/components/ui/formStyles'

interface Props {
  rootPath: string
  reportName: string
  datasetName: string
  ready: boolean
  onFinished: (response: DiagnosisResponse) => void
}

export default function DiagnosisActions({ rootPath, reportName, datasetName, ready, onFinished }: Props) {
  const { t } = useLocale()
  const [limit, setLimit] = useState(50)
  const [running, setRunning] = useState(false)
  const [error, setError] = useState('')

  async function analyze(options: DiagnosisRunRequest) {
    setRunning(true)
    setError('')
    try {
      onFinished(await runDiagnosis(rootPath, reportName, datasetName, options))
    } catch (error) {
      setError(error instanceof Error ? error.message : t('diagnosis.runFailed'))
    } finally {
      setRunning(false)
    }
  }

  function submitModel(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    const form = new FormData(event.currentTarget)
    const field = (name: string) => String(form.get(name) ?? '').trim()
    const scope = field('scope')
    void analyze({
      level: 'diagnose',
      model_id: field('model'), api_url: field('url'), api_key: field('key') || null,
      scope: scope === 'unscored' || scope === 'risk' || scope === 'all' ? scope : 'failed',
      max_diagnosed_samples: Number(field('limit')),
      include_trace: form.has('trace'),
    })
  }

  return (
    <div className="flex flex-wrap items-start gap-3">
      <Button size="sm" variant={ready ? 'outline' : 'primary'} disabled={running} onClick={() => void analyze({ level: 'facts' })}>
        {t(ready ? 'diagnosis.updateFacts' : 'diagnosis.generateFacts')}
      </Button>
      {ready && <details className="pt-1 text-sm">
        <summary className="diagnosis-link w-fit cursor-pointer">{t('diagnosis.modelOptions')}</summary>
        <form onSubmit={submitModel} className="mt-2 max-w-3xl space-y-3 rounded-md border border-[var(--border)] p-3">
          <p className="text-xs text-[var(--text-muted)]">{t('diagnosis.modelInputNotice', { count: limit + 1 })}</p>
          <fieldset disabled={running} className="grid grid-cols-2 gap-3">
            <Field id="diagnosis-model" name="model" labelKey="diagnosis.analysisModel" required>
              {(aria) => <input {...aria} required className={FORM_INPUT_CLASS} />}
            </Field>
            <Field id="diagnosis-url" name="url" labelKey="diagnosis.modelEndpoint" required>
              {(aria) => <input {...aria} type="url" required className={FORM_INPUT_CLASS} />}
            </Field>
            <Field id="diagnosis-key" name="key" labelKey="diagnosis.modelKey" autoComplete="off">
              {(aria) => <input {...aria} type="password" className={FORM_INPUT_CLASS} />}
            </Field>
            <Field id="diagnosis-limit" name="limit" labelKey="diagnosis.modelLimit" required>
              {(aria) => <input {...aria} type="number" required min={1} max={50} defaultValue={50} onChange={event => setLimit(event.currentTarget.valueAsNumber || 0)} className={FORM_INPUT_CLASS} />}
            </Field>
            <Field id="diagnosis-scope" name="scope" labelKey="diagnosis.selectedScope">
              {(aria) => <select {...aria} className={FORM_INPUT_CLASS}>
                {(['failed', 'unscored', 'risk', 'all'] as const).map(scope => <option key={scope} value={scope}>{t(`diagnosis.reviewScope.${scope}`)}</option>)}
              </select>}
            </Field>
            <label className="flex items-center gap-2 self-end py-2 text-sm">
              <input type="checkbox" name="trace" defaultChecked />
              {t('diagnosis.includeTrace')}
            </label>
          </fieldset>
          <Button type="submit" size="sm" disabled={running}>{t('diagnosis.startModelAnalysis')}</Button>
        </form>
      </details>}
      {running && <p role="status" className="basis-full text-sm text-[var(--text-muted)]">{t('diagnosis.runningAnalysis')}</p>}
      {error && <div className="basis-full"><ErrorAlert>{error}</ErrorAlert></div>}
    </div>
  )
}
