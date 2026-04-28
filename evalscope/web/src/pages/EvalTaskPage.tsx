import { useCallback, useMemo, useState } from 'react'
import { useLocale } from '@/contexts/LocaleContext'
import { useQueryParams } from '@/hooks/useQueryParams'
import EvalConfigForm from '@/components/eval/EvalConfigForm'
import TaskMonitor from '@/components/eval/TaskMonitor'
import Card from '@/components/ui/Card'
import { submitEvalTask, getEvalProgress, getEvalLog, getEvalReportUrl } from '@/api/eval'
import type { EvalInvokeResponse, LogResponse, ProgressResponse } from '@/api/types'
import { usePolling } from '@/hooks/usePolling'

export default function EvalTaskPage() {
  const { t } = useLocale()
  const queryParams = useQueryParams()
  const initialDataset = queryParams.get('dataset')

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
    <div className="page-enter">
      <h1 className="text-xl font-semibold mb-6">{t('eval.title')}</h1>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Left: Config Form */}
        <Card title={t('eval.config')}>
          <EvalConfigForm
            onSubmit={handleSubmit}
            disabled={running}
            initialDataset={initialDataset}
          />
        </Card>

        {/* Right: Task Monitor */}
        <Card title={t('eval.status')}>
          <TaskMonitor
            running={running}
            progress={progress}
            logText={logText}
            result={result}
            reportUrl={reportUrl}
            readyLabel={t('eval.ready')}
          />
        </Card>
      </div>
    </div>
  )
}
