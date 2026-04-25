import type { EvalInvokeResponse } from '@/api/types'
import LogViewer from '@/components/common/LogViewer'
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
    <div className="space-y-3">
      {/* Status */}
      <div className="flex items-center gap-2 text-sm">
        {running && (
          <>
            <Loader2 size={16} className="animate-spin text-[var(--color-primary)]" />
            <span>Running... {progress > 0 ? `${Math.round(progress)}%` : ''}</span>
          </>
        )}
        {!running && result?.status === 'error' && (
          <>
            <XCircle size={16} className="text-red-500" />
            <span className="text-red-400">{result.error}</span>
          </>
        )}
        {!running && result && result.status !== 'error' && (
          <>
            <CheckCircle2 size={16} className="text-green-500" />
            <span className="text-green-400">Completed</span>
          </>
        )}
        {!running && !result && (
          <span className="text-[var(--color-ink-muted)]">{readyLabel}</span>
        )}
      </div>

      {/* Progress bar */}
      {(running || (result && result.status !== 'error')) && (
        <div className="h-1.5 rounded-full bg-[var(--color-surface-hover)] overflow-hidden">
          <div
            className="h-full rounded-full bg-[var(--color-primary)] transition-all duration-500"
            style={{ width: `${Math.min(progress, 100)}%` }}
          />
        </div>
      )}

      {/* Log */}
      {logText && <LogViewer content={logText} />}

      {/* Report link */}
      {!running && result && result.status !== 'error' && reportUrl && (
        <a
          href={reportUrl}
          target="_blank"
          rel="noreferrer"
          className="inline-flex items-center gap-1.5 text-sm text-[var(--color-primary)] hover:underline"
        >
          <ExternalLink size={14} />
          {t('common.openNewTab')}
        </a>
      )}
    </div>
  )
}
