import { useState } from 'react'
import { ChevronLeft, ChevronRight, ExternalLink } from 'lucide-react'
import { getPredictions, getSampleDiagnoses, type DiagnosisFilters } from '@/api/reports'
import type { SampleKey } from '@/api/types'
import type { ReportMetric } from '@/domain/report/primaryMetrics'
import { useAsyncResource } from '@/hooks/useAsyncResource'
import { useLocale } from '@/contexts/LocaleContext'
import ChatView from '@/components/chat/ChatView'
import ErrorAlert from '@/components/ui/ErrorAlert'
import Skeleton from '@/components/ui/Skeleton'

interface Props {
  rootPath: string
  reportName: string
  datasetName: string
  filters: DiagnosisFilters
  metrics?: ReportMetric[]
  focusStep?: number
  onSamples: (filters: DiagnosisFilters, sample?: SampleKey, step?: number) => void
}

/** Browse one issue's complete sample set, including identities from other subsets. */
export default function DiagnosisSamplePanel({
  rootPath,
  reportName,
  datasetName,
  filters,
  metrics,
  focusStep,
  onSamples,
}: Props) {
  const { t } = useLocale()
  const [position, setPosition] = useState(0)
  const page = Math.floor(position / 200) + 1
  const samples = useAsyncResource(
    async (signal) => ({
      page,
      response: await getSampleDiagnoses(rootPath, reportName, datasetName, filters, page, signal),
    }),
    [rootPath, reportName, datasetName, JSON.stringify(filters), page],
  )
  const data = samples.data?.page === page ? samples.data.response : undefined
  const sample = data?.status === 'ready' ? data.items[position % 200] : undefined
  const identity = sample ? `${sample.key.subset}:${sample.key.index}` : ''
  const prediction = useAsyncResource(
    async (signal) => ({
      identity,
      response: await getPredictions(rootPath, reportName, datasetName, sample!.key.subset, signal, sample!.key.index),
    }),
    [rootPath, reportName, datasetName, identity],
    { enabled: Boolean(sample) },
  )
  const row =
    prediction.data?.identity === identity
      ? prediction.data.response.predictions.find((row) => row.Index === String(sample?.key.index))
      : undefined
  return (
    <aside
      aria-label={t('diagnosis.samplePanel')}
      className="[--sample-reader-height:max(28rem,calc(100vh-330px))] min-w-0 overflow-hidden rounded-lg border border-[var(--border)] bg-[var(--bg-card)]"
    >
      <header className="flex items-center justify-between gap-3 border-b border-[var(--border)] px-4 py-3 text-xs">
        <span className="min-w-0 truncate text-sm">{sample ? `${sample.key.subset} · ${t('diagnosis.sampleLink', { index: sample.key.index })}` : '—'}</span>
        <div className="flex shrink-0 items-center gap-3">
          <button
            disabled={position === 0 || samples.loading}
            onClick={() => setPosition((p) => p - 1)}
            aria-label={t('prediction.previousSample')}
            className="disabled:opacity-40"
          >
            <ChevronLeft size={16} />
          </button>
          <span>{data ? `${data.total ? position + 1 : 0} / ${data.total}` : '—'}</span>
          <button
            disabled={!data || position + 1 >= data.total || samples.loading}
            onClick={() => setPosition((p) => p + 1)}
            aria-label={t('prediction.nextSample')}
            className="disabled:opacity-40"
          >
            <ChevronRight size={16} />
          </button>
          {sample && (
            <button
              onClick={() => onSamples(filters, sample.key, focusStep)}
              className="diagnosis-link ml-3 flex items-center gap-1"
            >
              <ExternalLink size={13} />
              {t('diagnosis.openSample')}
            </button>
          )}
        </div>
      </header>
      <div className="p-3">
        {samples.error || prediction.error ? (
          <ErrorAlert>{samples.error || prediction.error}</ErrorAlert>
        ) : samples.loading ||
          prediction.loading ||
          (sample && prediction.data?.identity !== identity && !prediction.error) ? (
          <Skeleton lines={6} />
        ) : data?.status !== 'ready' ? (
          <p className="p-3 text-sm">{t('diagnosis.stale')}</p>
        ) : !sample || !row ? (
          <p className="p-3 text-sm">{t('diagnosis.noSamples')}</p>
        ) : (
          <ChatView key={identity} prediction={row} diagnosis={sample} metrics={metrics} focusEvidence focusStep={focusStep} navigation="toolbar" />
        )}
      </div>
    </aside>
  )
}
