import { useNavigate } from 'react-router-dom'
import { ExternalLink, ArrowLeft } from 'lucide-react'
import { useLocale } from '@/contexts/LocaleContext'
import Badge from '@/components/ui/Badge'
import Button from '@/components/ui/Button'
import { formatScore } from '@/utils/formatUtils'
import { scoreColor } from '@/components/ui/Table'

interface Props {
  modelName: string
  datasetName: string
  score: number
  totalSamples: number
  htmlReportUrl: string
}

export default function ReportHeader({
  modelName,
  datasetName,
  score,
  totalSamples,
  htmlReportUrl,
}: Props) {
  const { t } = useLocale()
  const navigate = useNavigate()

  const normalizedScore = score > 1 ? score / 100 : score
  const variant = normalizedScore >= 0.7 ? 'success' : normalizedScore >= 0.4 ? 'warning' : 'danger'

  return (
    <div
      className="rounded-[var(--radius)] border border-[var(--border)] bg-[var(--bg-card)] p-5"
      style={{ boxShadow: 'var(--shadow-sm)' }}
    >
      <div className="flex flex-wrap items-start justify-between gap-4">
        {/* Left side - info */}
        <div className="flex flex-col gap-2">
          <div className="flex items-center gap-3 flex-wrap">
            <h1 className="text-xl font-bold text-[var(--text)]">{modelName}</h1>
            <span className="text-[var(--text-dim)]">×</span>
            <span className="text-lg text-[var(--text-muted)]">{datasetName}</span>
          </div>
          <div className="flex items-center gap-3">
            <Badge variant={variant} className="font-mono text-sm px-3 py-1">
              <span style={{ color: scoreColor(normalizedScore) }}>{formatScore(score)}</span>
            </Badge>
            <span className="text-sm text-[var(--text-muted)]">
              {totalSamples.toLocaleString()} {t('single.samples')}
            </span>
          </div>
        </div>

        {/* Right side - actions */}
        <div className="flex items-center gap-2">
          <Button
            variant="outline"
            size="sm"
            onClick={() => window.open(`/viewer?url=${encodeURIComponent(htmlReportUrl)}`, '_blank')}
          >
            <ExternalLink size={14} />
            {t('reportDetail.viewHtml')}
          </Button>
          <Button
            variant="ghost"
            size="sm"
            onClick={() => navigate(-1)}
          >
            <ArrowLeft size={14} />
            {t('reportDetail.backToReports')}
          </Button>
        </div>
      </div>
    </div>
  )
}
