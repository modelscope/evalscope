import type { EvalInvokeResponse } from '@/api/types'
import LogViewer from '@/components/common/LogViewer'
import Badge from '@/components/ui/Badge'
import Button from '@/components/ui/Button'
import { useLocale } from '@/contexts/LocaleContext'
import { ExternalLink, CheckCircle2, XCircle, Loader2 } from 'lucide-react'

interface Props {
  running: boolean
  progress: number
  logText: string
  result: EvalInvokeResponse | null
  reportUrl: string | null
  readyLabel: string
}

export default function TaskMonitor({ running, progress, logText, result, reportUrl, readyLabel }: Props) {
  const { t } = useLocale()

  return (
    <div className="space-y-4">
      {/* Status */}
      <div className="flex items-center gap-2 text-sm">
        {running && (
          <>
            <Loader2 size={16} className="animate-spin text-[var(--accent)]" />
            <Badge variant="warning">Running{progress > 0 ? ` ${Math.round(progress)}%` : '...'}</Badge>
          </>
        )}
        {!running && result?.status === 'error' && (
          <>
            <XCircle size={16} className="text-[#ef4444]" />
            <Badge variant="danger">{result.error}</Badge>
          </>
        )}
        {!running && result && result.status !== 'error' && (
          <>
            <CheckCircle2 size={16} className="text-[var(--green)]" />
            <Badge variant="success">Completed</Badge>
          </>
        )}
        {!running && !result && (
          <span className="text-[var(--text-muted)]">{readyLabel}</span>
        )}
      </div>

      {/* Progress bar */}
      {(running || (result && result.status !== 'error')) && (
        <div className="h-1.5 rounded-full overflow-hidden" style={{ background: 'var(--bg-deep)' }}>
          <div
            className="h-full rounded-full transition-all duration-500"
            style={{ width: `${Math.min(progress, 100)}%`, background: 'var(--accent)' }}
          />
        </div>
      )}

      {/* Log */}
      {logText && <LogViewer content={logText} />}

      {/* Report link */}
      {!running && result && result.status !== 'error' && reportUrl && (
        <Button
          variant="primary"
          size="sm"
          onClick={() => window.open(reportUrl, '_blank')}
          className="btn-glow"
        >
          <ExternalLink size={14} />
          {t('common.openNewTab')}
        </Button>
      )}
    </div>
  )
}
