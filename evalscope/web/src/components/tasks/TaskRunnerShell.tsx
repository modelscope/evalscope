import { useCallback, useMemo, useState, type ReactNode } from 'react'
import type { EvalInvokeResponse, LogResponse, ProgressResponse } from '@/api/types'
import { usePolling } from '@/hooks/usePolling'
import Card from '@/components/ui/Card'
import TaskMonitor from '@/components/eval/TaskMonitor'

interface FormRenderProps {
  onSubmit: (config: Record<string, unknown>, outputDirName?: string) => Promise<void>
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

/**
 * Turn a user-supplied output directory name into a task id. Filesystem- and
 * header-safe: only word characters and dashes survive, so the result can
 * never traverse out of the task's storage root or break the
 * `EvalScope-Task-Id` request header.
 */
function taskIdFromName(prefix: string, name: string): string {
  const cleaned = name.trim().replace(/[^A-Za-z0-9_-]+/g, '-').replace(/^-+|-+$/g, '').slice(0, 100)
  return cleaned ? `${prefix}_${cleaned}` : createTaskId(prefix)
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
  const [taskId, setTaskId] = useState<string | null>(null)
  const [running, setRunning] = useState(false)
  const [result, setResult] = useState<EvalInvokeResponse | null>(null)
  const [logText, setLogText] = useState('')
  const [logLine, setLogLine] = useState(0)
  const [progress, setProgress] = useState(0)

  const handleSubmit = async (config: Record<string, unknown>, outputDirName?: string) => {
    const id = outputDirName ? taskIdFromName(idPrefix, outputDirName) : createTaskId(idPrefix)
    setTaskId(id)
    setRunning(true)
    setLogText('')
    setLogLine(0)
    setProgress(0)
    setResult(null)
    try {
      setResult(await submitTask(config, id))
    } catch (error) {
      setResult({ status: 'error', task_id: id, error: String(error) })
    } finally {
      setRunning(false)
    }
  }

  const handleStop = async () => {
    if (!taskId) return
    try {
      await stopTask(taskId)
    } catch {
      // The local task state still needs to stop when the backend is unavailable.
    }
    setRunning(false)
    setResult({ status: 'stopped', task_id: taskId })
  }

  const progressFn = useCallback(async () => {
    if (!taskId) throw new Error('no task')
    return getProgress(taskId)
  }, [getProgress, taskId])

  const logFn = useCallback(async () => {
    if (!taskId) throw new Error('no task')
    return getLog(taskId, logLine)
  }, [getLog, logLine, taskId])

  usePolling<ProgressResponse>({
    fn: progressFn,
    enabled: running && !!taskId,
    interval: 5000,
    onData: (data) => {
      setProgress(data.percent ?? 0)
      // Stop polling when benchmark finishes (via percent) or when the
      // backend progress tracker reports error / completed (status key).
      if (data.percent >= 100 || data.status === 'error' || data.status === 'completed') {
        setRunning(false)
      }
    },
  })

  usePolling<LogResponse>({
    fn: logFn,
    enabled: running && !!taskId,
    interval: 5000,
    onData: (data) => {
      if (!data.text) return
      setLogText((previous) => previous + data.text)
      setLogLine(data.tail_line)
    },
  })

  const reportUrl = useMemo(() => (taskId ? getReportUrl(taskId) : null), [getReportUrl, taskId])

  return (
    <div className="page-enter">
      <h1 className="text-xl font-semibold mb-6">{title}</h1>
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Card title={configTitle}>{renderForm({ onSubmit: handleSubmit, disabled: running })}</Card>
        <Card title={statusTitle}>
          <TaskMonitor
            running={running}
            progress={progress}
            logText={logText}
            result={result}
            reportUrl={reportUrl}
            readyLabel={readyLabel}
            onStop={handleStop}
          />
        </Card>
      </div>
    </div>
  )
}
