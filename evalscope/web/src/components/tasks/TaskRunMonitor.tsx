import { useEffect, useRef, useState } from 'react'
import { Link } from 'react-router-dom'
import type { ActiveTaskInfo, EvalInvokeResponse, LogResponse } from '@/api/types'
import { getTask } from '@/api/tasks'
import { isDomainError } from '@/api/errors'
import { usePolling } from '@/hooks/usePolling'
import { useCopy } from '@/hooks/useCopy'
import { useLocale } from '@/contexts/LocaleContext'
import TaskMonitor from '@/components/eval/TaskMonitor'

interface Props {
  taskId: string
  active?: boolean
  result: EvalInvokeResponse | null
  stopTask: (taskId: string) => Promise<unknown>
  getLog: (taskId: string, tailLine: number) => Promise<LogResponse>
}

type StopRequestState = 'idle' | 'pending' | 'confirmed' | 'failed'

export default function TaskRunMonitor({ taskId, result, stopTask, getLog, active = true }: Props) {
  const { t } = useLocale()
  const { copy, copied } = useCopy()
  const title = useRef<HTMLHeadingElement>(null)
  const [logText, setLogText] = useState('')
  const [logLine, setLogLine] = useState(0)
  const [stopRequest, setStopRequest] = useState<StopRequestState>('idle')
  const [snapshot, setSnapshot] = useState<ActiveTaskInfo | null>(null)
  const live = useRef(true)
  useEffect(() => {
    live.current = true
    return () => {
      live.current = false
    }
  }, [])
  useEffect(() => {
    if (active) title.current?.focus()
  }, [active])

  const status = stopRequest === 'confirmed' ? 'stopped' : snapshot?.status ?? 'preparing'
  const terminal = ['completed', 'stopped', 'error'].includes(status)
  const task = usePolling({
    fn: () => getTask(taskId),
    enabled: !terminal || (status === 'completed' && !snapshot?.report_url && !snapshot?.html_report_url),
    interval: 5000,
    onData: setSnapshot,
  })
  usePolling({
    fn: () => getLog(taskId, logLine),
    enabled: !terminal,
    onData: (data) => {
      if (!data.text) return
      setLogText((previous) => previous + data.text)
      setLogLine(data.tail_line)
    },
  })
  const handleStop = async () => {
    setStopRequest('pending')
    try {
      await stopTask(taskId)
      if (live.current) setStopRequest('confirmed')
    } catch {
      if (live.current) setStopRequest('failed')
    }
  }

  let verdict: EvalInvokeResponse | null = null
  if (status === 'completed' || status === 'stopped' || status === 'error') {
    verdict = { status, task_id: taskId }
    if (status === 'error') verdict.error = t('tasks.failedWithLogHint')
  }

  let progress: number | null = null
  if (status === 'completed') {
    progress = 100
  } else if (typeof snapshot?.percent === 'number' && Number.isFinite(snapshot.percent)) {
    progress = snapshot.percent
  }

  const displayName = snapshot?.alias || snapshot?.model || taskId
  const endpoint = snapshot?.endpoint
  const copyLabel = t(copied ? 'tasks.copied' : 'tasks.copy')
  const configEntries = Object.entries(snapshot?.config ?? {})
    .filter(([, value]) => value !== null && value !== undefined)
  const taskNotFound = isDomainError(task.failure) && task.failure.status === 404
  return (
    <div className="min-w-0 space-y-4">
      <h2 ref={title} tabIndex={-1} className="break-words font-semibold outline-none" aria-live="polite">
        {t('tasks.viewing')}: {displayName}
      </h2>
      <div className="space-y-1 text-sm text-[var(--text-muted)]">
        {snapshot?.model && (
          <p className="break-all">{t('eval.modelName')}: {snapshot.model}</p>
        )}
        <p className="flex flex-wrap items-center gap-2 break-all">
          {t('perf.archive.endpoint')}: {endpoint || '—'}
          {endpoint && (
            <button type="button" onClick={() => copy(endpoint)}>{copyLabel}</button>
          )}
        </p>
        <p className="flex flex-wrap items-center gap-2 break-all">
          ID: {taskId}
          <button type="button" onClick={() => copy(taskId)}>{copyLabel}</button>
        </p>
        {snapshot?.started_at && (
          <p>{t('tasks.startedAt')}: {snapshot.started_at}</p>
        )}
      </div>
      {['unknown', 'finalizing'].includes(status) && (
        <p role="status">{t(`tasks.state.${status}`)}</p>
      )}
      {result?.status === 'error' && !terminal && (
        <p role="alert">{result.error}</p>
      )}
      {task.error && (
        <div role="alert" className="text-[var(--danger)]">
          {t(taskNotFound ? 'tasks.notFound' : 'tasks.connectionError')}
          <button type="button" onClick={() => task.refresh()}>{t('common.retry')}</button>
        </div>
      )}
      {stopRequest === 'failed' && !terminal && (
        <p role="alert" className="text-[var(--danger)]">{t('tasks.stopUnconfirmed')}</p>
      )}
      <details className="rounded border border-[var(--border)] p-3">
        <summary className="cursor-pointer">{t('tasks.actualConfig')}</summary>
        {snapshot?.config ? (
          <dl className="mt-2 space-y-2 text-sm">
            {configEntries.map(([key, value]) => (
              <div key={key} className="flex flex-wrap justify-between gap-2">
                <dt>{key}</dt>
                <dd className="break-all">{Array.isArray(value) ? value.join(', ') : String(value)}</dd>
              </div>
            ))}
          </dl>
        ) : (
          <p className="text-sm text-[var(--text-muted)]">{t('tasks.configUnavailable')}</p>
        )}
      </details>
      {status === 'completed' && !snapshot?.report_url && !snapshot?.html_report_url && (
        <p>{t('tasks.reportUnavailable')}</p>
      )}
      {snapshot?.report_url && (
        <Link
          className="inline-block rounded bg-[var(--accent)] px-4 py-2 text-[var(--text-on-filled)]"
          to={snapshot.report_url}
        >
          {t('tasks.viewReport')}
        </Link>
      )}
      {snapshot?.html_report_url && (
        <a className="ml-3 text-sm text-[var(--accent)]" href={snapshot.html_report_url} target="_blank" rel="noreferrer">
          {t('tasks.htmlReport')}
        </a>
      )}
      <TaskMonitor
        running={!terminal && status !== 'unknown'}
        progress={progress}
        logText={logText}
        result={verdict}
        reportUrl={null}
        readyLabel=""
        statusLabel={t(`tasks.state.${status}`)}
        onStop={handleStop}
        stopping={stopRequest === 'pending'}
      />
    </div>
  )
}
