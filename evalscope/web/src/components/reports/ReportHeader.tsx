import { useNavigate } from 'react-router-dom'
import { ExternalLink, ArrowLeft } from 'lucide-react'
import { useLocale } from '@/contexts/LocaleContext'
import Button from '@/components/ui/Button'

interface Props {
  modelName: string
  datasetCount: number
  totalSamples: number
  htmlReportUrl: string
  reportsUrl: string
}

export default function ReportHeader({
  modelName,
  datasetCount,
  totalSamples,
  htmlReportUrl,
  reportsUrl,
}: Props) {
  const { t } = useLocale()
  const navigate = useNavigate()
  return (
    <div className="flex items-center justify-between gap-4 rounded-[var(--radius)] border border-[var(--border)] bg-[var(--bg-card)] px-5 py-3">
      <div className="flex min-w-0 flex-1 flex-wrap items-baseline gap-x-4 gap-y-1">
        <h1 className="break-words text-xl font-bold">{modelName}</h1>
        <p className="text-sm text-[var(--text-muted)]">
          {t('reportDetail.datasetCount', { count: datasetCount })} ·{' '}
          {t('diagnosis.sampleCount', { count: totalSamples })}
        </p>
      </div>
      <div className="flex flex-wrap items-center gap-2">
        <Button
          variant="outline"
          size="sm"
          onClick={() => window.open(`/viewer?url=${encodeURIComponent(htmlReportUrl)}`, '_blank')}
        >
          <ExternalLink size={14} />
          {t('reportDetail.viewHtml')}
        </Button>
        <Button variant="ghost" size="sm" onClick={() => navigate(reportsUrl)}>
          <ArrowLeft size={14} />
          {t('reportDetail.backToReports')}
        </Button>
      </div>
    </div>
  )
}
