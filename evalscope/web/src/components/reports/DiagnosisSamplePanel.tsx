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
  position: number
  expectedSample?: SampleKey
  onPositionChange: (position: number) => void
  onSamples: (filters: DiagnosisFilters, sample?: SampleKey, step?: number, position?: number) => void
}

/** Browse one issue's complete sample set, including identities from other subsets. */
export default function DiagnosisSamplePanel({
  rootPath,
  reportName,
  datasetName,
  filters,
  metrics,
  focusStep,
  position,
  expectedSample,
  onPositionChange,
  onSamples,
}: Props) {
  const { t } = useLocale()
  const page = Math.floor(position / 200) + 1
  const query = JSON.stringify(filters)
  const samples = useAsyncResource(
    async (signal) => ({
      page, query,
      response: await getSampleDiagnoses(rootPath, reportName, datasetName, filters, page, signal),
    }),
    [rootPath, reportName, datasetName, query, page],
  )
  const data = samples.data?.page === page && samples.data.query === query ? samples.data.response : undefined
  const located = data?.status === 'ready' ? data.items[position % 200] : undefined
  const moved = data?.status === 'ready' && expectedSample &&
    (located?.key.subset !== expectedSample.subset || located?.key.index !== expectedSample.index)
  const sample = moved ? undefined : located
  const key = (moved ? expectedSample : sample?.key) ?? (data?.status === 'ready' && filters.subset_name && filters.index !== undefined
    ? { subset: filters.subset_name, index: filters.index } : undefined)
  const identity = key ? JSON.stringify(key) : ''
  const prediction = useAsyncResource(
    async (signal) => ({
      identity,
      response: await getPredictions(rootPath, reportName, datasetName, key!.subset, signal, key!.index),
    }),
    [rootPath, reportName, datasetName, identity],
    { enabled: Boolean(key) },
  )
  const row =
    prediction.data?.identity === identity
      ? prediction.data.response.predictions.find((row) => row.Index === String(key?.index))
      : undefined
  return (
    <aside
      aria-label={t('diagnosis.samplePanel')}
      className="[--sample-reader-height:max(28rem,calc(100vh-330px))] min-w-0 overflow-hidden rounded-lg border border-[var(--border)] bg-[var(--bg-card)]"
    >
      <header className="flex items-center justify-between gap-3 border-b border-[var(--border)] px-4 py-3 text-xs">
        <span className="min-w-0 truncate text-sm">{key ? `${key.subset} · ${t('diagnosis.sampleLink', { index: key.index })}` : '—'}</span>
        <div className="flex shrink-0 items-center gap-3">
          <button
            disabled={Boolean(moved) || position === 0 || samples.loading}
            onClick={() => onPositionChange(position - 1)}
            aria-label={t('prediction.previousSample')}
            className="disabled:opacity-40"
          >
            <ChevronLeft size={16} />
          </button>
          <span>{data ? `${moved ? 1 : key ? position + 1 : 0} / ${moved ? 1 : data.total || Number(Boolean(key))}` : '—'}</span>
          <button
            disabled={Boolean(moved) || !data || position + 1 >= data.total || samples.loading}
            onClick={() => onPositionChange(position + 1)}
            aria-label={t('prediction.nextSample')}
            className="disabled:opacity-40"
          >
            <ChevronRight size={16} />
          </button>
          {key && (
            <button
              onClick={() => onSamples(moved ? {} : filters, key, focusStep, !moved && filters.index === undefined ? position : undefined)}
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
          (key && prediction.data?.identity !== identity && !prediction.error) ? (
          <Skeleton lines={6} />
        ) : data?.status !== 'ready' ? (
          <p className="p-3 text-sm">{t('diagnosis.stale')}</p>
        ) : !key || !row ? (
          <p className="p-3 text-sm">{t('diagnosis.noSamples')}</p>
        ) : (
          <ChatView key={identity} prediction={row} diagnosis={sample} metrics={metrics} focusEvidence focusStep={focusStep} navigation="toolbar" />
        )}
      </div>
    </aside>
  )
}
