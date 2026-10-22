import { useEffect, useState, type ReactNode } from 'react'
import type { EvalInvokeResponse, LogResponse } from '@/api/types'
import { isDomainError } from '@/api/errors'
import { useLocale } from '@/contexts/LocaleContext'
import Card from '@/components/ui/Card'
import TaskRunMonitor from '@/components/tasks/TaskRunMonitor'

interface FormRenderProps {
  onSubmit: (config: Record<string, unknown>) => Promise<void>
  disabled: boolean
}

export interface TaskSelectionProps {
  selectedTaskId: string | null
  onSelect: (taskId: string) => void
  active?: boolean
}

interface TaskRunnerShellProps extends TaskSelectionProps {
  idPrefix: string
  title: string
  configTitle: string
  statusTitle: string
  readyLabel: string
  renderForm: (props: FormRenderProps) => ReactNode
  submitTask: (config: Record<string, unknown>, taskId: string) => Promise<EvalInvokeResponse>
  stopTask: (taskId: string) => Promise<unknown>
  getLog: (taskId: string, tailLine: number) => Promise<LogResponse>
}

function createTaskId(prefix: string): string {
  return `${prefix}_${Date.now()}`
}

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
  getLog,
  selectedTaskId,
  onSelect,
  active = true,
}: TaskRunnerShellProps) {
  const { t } = useLocale()
  const [submitting, setSubmitting] = useState(false)
  const [result, setResult] = useState<EvalInvokeResponse | null>(null)
  const [rejectedTaskId, setRejectedTaskId] = useState<string | null>(null)
  useEffect(() => {
    try {
      if (selectedTaskId && selectedTaskId === rejectedTaskId) {
        sessionStorage.removeItem(`evalscope-active-task-${idPrefix}`)
      } else if (selectedTaskId) {
        sessionStorage.setItem(`evalscope-active-task-${idPrefix}`, selectedTaskId)
      }
    } catch {
      return
    }
  }, [idPrefix, selectedTaskId, rejectedTaskId])
  const handleSubmit = async (config: Record<string, unknown>): Promise<void> => {
    const id = createTaskId(idPrefix)
    setSubmitting(true)
    onSelect(id)
    try {
      setResult(await submitTask(config, id))
    } catch (error) {
      const rejected = isDomainError(error) && (error.status === 400 || error.status === 422)
      setRejectedTaskId(rejected ? id : null)
      setResult({ status: 'error', task_id: id, error: rejected ? error.message : t('tasks.submitUnconfirmed') })
    } finally {
      setSubmitting(false)
    }
  }
  return (
    <div className="page-enter min-w-0">
      <h2 className="mb-4 text-base font-semibold">{title}</h2>
      <div className="grid min-w-0 grid-cols-1 items-start gap-4 lg:grid-cols-[minmax(0,3fr)_minmax(0,2fr)]">
        <Card title={`${t('tasks.newTask')} · ${configTitle}`}>
          <TaskForm renderForm={renderForm} onSubmit={handleSubmit} disabled={submitting} />
        </Card>
        <Card title={statusTitle}>
          {selectedTaskId ? (
            <TaskRunMonitor
              key={selectedTaskId}
              taskId={selectedTaskId}
              active={active}
              result={result?.task_id === selectedTaskId ? result : null}
              submissionRejected={rejectedTaskId === selectedTaskId}
              stopTask={stopTask}
              getLog={getLog}
            />
          ) : (
            <div className="space-y-2 py-3">
              <p className="text-sm text-[var(--text)]">{readyLabel}</p>
              <p className="text-xs text-[var(--text-muted)]">{t('tasks.library.emptyMonitor')}</p>
            </div>
          )}
        </Card>
      </div>
    </div>
  )
}
