import type { EvalInvokeResponse } from '@/api/types'
import LogViewer from '@/components/ui/LogViewer'
import Badge from '@/components/ui/Badge'
import Button from '@/components/ui/Button'
import { useLocale } from '@/contexts/LocaleContext'
import { ExternalLink, CheckCircle2, XCircle, Loader2, Square, OctagonX } from 'lucide-react'

interface Props {
  running: boolean
  progress: number | null
  logText: string
  result: EvalInvokeResponse | null
  reportUrl: string | null
  readyLabel: string
  onStop?: () => void
  warning?: string | null
  stopping?: boolean
  statusLabel?: string
}

export default function TaskMonitor({
  running, progress, logText, result, reportUrl, readyLabel, onStop, warning, stopping, statusLabel,
}: Props) {
  const { t } = useLocale()

  return (
    <div className="space-y-4">
      {/* Status */}
      <div className="flex items-center gap-2 text-sm">
        {running && (
          <>
            <Loader2 size={16} className="animate-spin text-[var(--accent)]" />
            <Badge variant="warning">
              {statusLabel ?? t(progress === null ? 'tasks.state.preparing' : 'tasks.state.running')}
              {progress !== null && ` ${Math.round(progress)}%`}
            </Badge>
          </>
        )}
        {!running && result?.status === 'stopped' && (
          <>
            <OctagonX size={16} className="text-[var(--yellow)]" />
            <Badge variant="warning">{t('tasks.state.stopped')}</Badge>
          </>
        )}
        {!running && result?.status === 'error' && (
          <>
            <XCircle size={16} className="text-[var(--danger)]" />
            <Badge variant="danger">{result.error}</Badge>
          </>
        )}
        {!running && result && result.status !== 'error' && result.status !== 'stopped' && (
          <>
            <CheckCircle2 size={16} className="text-[var(--green)]" />
            <Badge variant="success">{t('tasks.state.completed')}</Badge>
          </>
        )}
        {!running && !result && (
          <span className="text-[var(--text-muted)]">{readyLabel}</span>
        )}
      </div>

      {/* Progress bar */}
      {(running || (result && result.status !== 'error' && result.status !== 'stopped')) && (
        <div className="h-1.5 rounded-full overflow-hidden" style={{ background: 'var(--bg-deep)' }}>
          <div
            role="progressbar"
            aria-label={statusLabel ?? t('tasks.state.running')}
            aria-valuemin={0}
            aria-valuemax={100}
            aria-valuenow={progress === null ? undefined : Math.max(0, Math.min(progress, 100))}
            style={{
              width: progress === null ? '30%' : `${Math.max(0, Math.min(progress, 100))}%`,
              background: 'var(--accent)',
            }}
            className={progress === null
              ? 'h-full animate-pulse rounded-full'
              : 'h-full rounded-full transition-all duration-500'}
          />
        </div>
      )}

      {/* Warning */}
      {warning && <p className="text-sm text-[var(--yellow)]">{warning}</p>}

      {/* Stop button */}
      {running && onStop && (
        <Button
          variant="outline"
          size="sm"
          onClick={onStop}
          disabled={stopping}
          className="text-[var(--danger)] border-[var(--danger)] hover:bg-[var(--danger-bg)]"
        >
          <Square size={14} />
          {t(stopping ? 'tasks.stopping' : 'common.stop')}
        </Button>
      )}

      {/* Log */}
      {logText && <LogViewer content={logText} />}

      {/* Report link */}
      {!running && result && result.status !== 'error' && result.status !== 'stopped' && reportUrl && (
        <Button
          variant="primary"
          size="sm"
          onClick={() => window.open(reportUrl, '_blank')}
          className="btn-glow"
        >
          <ExternalLink size={14} />
          {t('tasks.viewReport')}
        </Button>
      )}
    </div>
  )
}
