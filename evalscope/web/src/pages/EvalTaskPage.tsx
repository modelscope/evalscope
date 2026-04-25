import { useCallback, useMemo, useState } from 'react'
import { useLocale } from '@/contexts/LocaleContext'
import EvalConfigForm from '@/components/eval/EvalConfigForm'
import TaskMonitor from '@/components/eval/TaskMonitor'
import { submitEvalTask, getEvalProgress, getEvalLog, getEvalReportUrl } from '@/api/eval'
import type { EvalInvokeResponse, LogResponse, ProgressResponse } from '@/api/types'
import { usePolling } from '@/hooks/usePolling'

export default function EvalTaskPage() {
  const { t } = useLocale()
  const [taskId, setTaskId] = useState<string | null>(null)
  const [running, setRunning] = useState(false)
  const [result, setResult] = useState<EvalInvokeResponse | null>(null)
  const [logText, setLogText] = useState('')
  const [logLine, setLogLine] = useState(0)
  const [progress, setProgress] = useState(0)

  const handleSubmit = async (config: Record<string, unknown>) => {
    const id = `eval_${Date.now()}`
    setTaskId(id)
    setRunning(true)
    setLogText('')
    setLogLine(0)
    setProgress(0)
    setResult(null)
    try {
      const res = await submitEvalTask(config, id)
      setResult(res)
      if (res.status === 'error') setRunning(false)
    } catch (e) {
      setResult({ status: 'error', task_id: id, error: String(e) })
      setRunning(false)
    }
  }

  const progressFn = useCallback(async () => {
    if (!taskId) throw new Error('no task')
    return getEvalProgress(taskId)
  }, [taskId])

  const logFn = useCallback(async () => {
    if (!taskId) throw new Error('no task')
    return getEvalLog(taskId, logLine)
  }, [taskId, logLine])

  usePolling<ProgressResponse>({
    fn: progressFn,
    enabled: running && !!taskId,
    interval: 2000,
    onData: (d) => {
      setProgress(d.percent ?? 0)
      if (d.percent >= 100) setRunning(false)
    },
  })

  usePolling<LogResponse>({
    fn: logFn,
    enabled: running && !!taskId,
    interval: 2000,
    onData: (d) => {
      if (d.text) {
        setLogText((prev) => prev + d.text)
        setLogLine(d.tail_line)
      }
    },
  })

  const reportUrl = useMemo(() => (taskId ? getEvalReportUrl(taskId) : null), [taskId])

  return (
    <div className="space-y-6 max-w-4xl">
      <h1 className="text-xl font-semibold">{t('eval.title')}</h1>

      <section className="rounded-lg border border-[var(--color-border)] bg-[var(--color-surface)] p-4">
        <h2 className="text-base font-medium mb-4">{t('eval.config')}</h2>
        <EvalConfigForm onSubmit={handleSubmit} disabled={running} />
      </section>

      <section className="rounded-lg border border-[var(--color-border)] bg-[var(--color-surface)] p-4">
        <h2 className="text-base font-medium mb-4">{t('eval.status')}</h2>
        <TaskMonitor
          running={running}
          progress={progress}
          logText={logText}
          result={result}
          reportUrl={reportUrl}
          readyLabel={t('eval.ready')}
        />
      </section>
    </div>
  )
}
