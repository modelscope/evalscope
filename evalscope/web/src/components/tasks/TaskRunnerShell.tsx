import { useCallback, useEffect, useMemo, useRef, useState, type ReactNode } from 'react'
import type { EvalInvokeResponse, LogResponse, ProgressResponse } from '@/api/types'
import { usePolling } from '@/hooks/usePolling'
import { useLocale } from '@/contexts/LocaleContext'
import Card from '@/components/ui/Card'
import TaskMonitor from '@/components/eval/TaskMonitor'

interface FormRenderProps {
  onSubmit: (config: Record<string, unknown>) => Promise<void>
  disabled: boolean
}

interface TaskRunnerShellProps {
  idPrefix: string
  title: string
  configTitle: string
  statusTitle: string
  readyLabel: string
  renderForm: (props: FormRenderProps) => ReactNode
  submitTask: (config: Record<string, unknown>, taskId: string) => Promise<EvalInvokeResponse>
  stopTask: (taskId: string) => Promise<unknown>
  getProgress: (taskId: string) => Promise<ProgressResponse>
  getLog: (taskId: string, tailLine: number) => Promise<LogResponse>
  getReportUrl: (taskId: string) => string
}

function createTaskId(prefix: string): string {
  return `${prefix}_${Date.now()}`
}

// Consecutive progress poll failures before the panel warns that the task may be lost.
const PROGRESS_FAILURE_HINT_THRESHOLD = 3

function TaskForm({ renderForm, ...props }: FormRenderProps & Pick<TaskRunnerShellProps, 'renderForm'>) {
  return renderForm(props)
}

export default function TaskRunnerShell({
  idPrefix,
  title,
  configTitle,
  statusTitle,
  readyLabel,
  renderForm,
  submitTask,
  stopTask,
  getProgress,
  getLog,
  getReportUrl,
}: TaskRunnerShellProps) {
  const { t } = useLocale()
  const storageKey = `evalscope-active-task-${idPrefix}`
  const [taskId, setTaskId] = useState<string | null>(() => {
    try {
      return sessionStorage.getItem(storageKey)
    } catch {
      return null
    }
  })
  const [running, setRunning] = useState(!!taskId)
  const [result, setResult] = useState<EvalInvokeResponse | null>(null)
  const [logText, setLogText] = useState('')
  const [logLine, setLogLine] = useState(0)
  const [progress, setProgress] = useState(0)
  const [progressFailures, setProgressFailures] = useState(0)
  const activeTaskId = useRef(taskId)

  useEffect(() => {
    try {
      // Keep only the task identity, never the submitted config or its secrets.
      // Session storage survives reloads for this tab.
      if (running && taskId) {
        sessionStorage.setItem(storageKey, taskId)
      } else if (sessionStorage.getItem(storageKey) === taskId) {
        sessionStorage.removeItem(storageKey)
      }
    } catch {
      // Task execution still works when browser storage is unavailable.
    }
  }, [running, storageKey, taskId])

  const handleSubmit = async (config: Record<string, unknown>) => {
    const id = createTaskId(idPrefix)
    activeTaskId.current = id
    setTaskId(id)
    setRunning(true)
    setLogText('')
    setLogLine(0)
    setProgress(0)
    setProgressFailures(0)
    setResult(null)
    try {
      const response = await submitTask(config, id)
      if (activeTaskId.current === id) setResult(response)
    } catch (error) {
      if (activeTaskId.current === id) setResult({ status: 'error', task_id: id, error: String(error) })
    } finally {
      if (activeTaskId.current === id) setRunning(false)
    }
  }

  const handleStop = async () => {
    if (!taskId) return
    try {
      await stopTask(taskId)
    } catch {
      // The local task state still needs to stop when the backend is unavailable.
    }
    if (activeTaskId.current !== taskId) return
    activeTaskId.current = null
    setRunning(false)
    setProgressFailures(0)
    setResult({ status: 'stopped', task_id: taskId })
  }

  const progressFn = useCallback(async () => {
    if (!taskId) throw new Error('no task')
    return { taskId, data: await getProgress(taskId) }
  }, [getProgress, taskId])

  const logFn = useCallback(async () => {
    if (!taskId) throw new Error('no task')
    return { taskId, data: await getLog(taskId, logLine) }
  }, [getLog, logLine, taskId])

  usePolling<{ taskId: string; data: ProgressResponse }>({
    fn: progressFn,
    enabled: running && !!taskId,
    interval: 5000,
    onData: ({ taskId: polledTaskId, data }) => {
      if (!taskId || activeTaskId.current !== polledTaskId) return
      setProgressFailures(0)
      setProgress(data.percent ?? 0)
      // Stop polling when benchmark finishes (via percent) or when the
      // backend progress tracker reports error / completed (status key).
      if (data.percent >= 100 || data.status === 'error' || data.status === 'completed') {
        setRunning(false)
        // After a reload the original invoke response is gone. The progress
        // endpoint must also restore the terminal state and report link, but
        // must not overwrite a more specific result from the invoke response.
        setResult((previous) => previous ?? (data.status === 'error'
          ? { status: 'error', task_id: taskId, error: t('tasks.failedWithLogHint') }
          : { status: 'completed', task_id: taskId }))
      }
    },
    onError: () => setProgressFailures((count) => count + 1),
  })

  usePolling<{ taskId: string; data: LogResponse }>({
    fn: logFn,
    enabled: running && !!taskId,
    interval: 5000,
    onData: ({ taskId: polledTaskId, data }) => {
      if (activeTaskId.current !== polledTaskId || !data.text) return
      setLogText((previous) => previous + data.text)
      setLogLine(data.tail_line)
    },
  })

  const reportUrl = useMemo(() => (taskId ? getReportUrl(taskId) : null), [getReportUrl, taskId])

  return (
    <div className="page-enter">
      <h1 className="text-xl font-semibold mb-6">{title}</h1>
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Card title={configTitle}>
          <TaskForm renderForm={renderForm} onSubmit={handleSubmit} disabled={running} />
        </Card>
        <Card title={statusTitle}>
          <TaskMonitor
            running={running}
            progress={progress}
            logText={logText}
            result={result}
            reportUrl={reportUrl}
            readyLabel={readyLabel}
            onStop={handleStop}
            warning={running && progressFailures >= PROGRESS_FAILURE_HINT_THRESHOLD
              ? t('tasks.progressUnreachable')
              : null}
          />
        </Card>
      </div>
    </div>
  )
}
