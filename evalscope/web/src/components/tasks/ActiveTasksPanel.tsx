import { Activity } from 'lucide-react'
import { listActiveTasks } from '@/api/tasks'
import type { ActiveTaskInfo, ListActiveTasksResponse } from '@/api/types'
import Card from '@/components/ui/Card'
import { useLocale } from '@/contexts/LocaleContext'
import { usePolling } from '@/hooks/usePolling'
import { formatTimestamp } from '@/utils/formatUtils'

interface ActiveTasksPanelProps {
  /** Attach a running task to its runner panel for live monitoring. */
  onAdopt: (kind: 'eval' | 'perf', taskId: string) => void
}

function ActiveTaskRow({ task, onAdopt }: { task: ActiveTaskInfo; onAdopt: ActiveTasksPanelProps['onAdopt'] }) {
  const { t } = useLocale()
  const kindLabel = task.kind === 'perf' ? t('tasks.activeKindPerf') : t('tasks.activeKindEval')
  const name = task.alias || task.model || task.task_id
  const percent = Math.max(0, Math.min(100, task.percent ?? 0))

  return (
    <div className="flex items-center gap-3 px-5 py-3">
      <span className="inline-flex shrink-0 items-center rounded-full bg-[var(--accent-dim)] px-2 py-0.5 text-xs font-medium text-[var(--accent)]">
        {kindLabel}
      </span>
      <div className="flex min-w-0 flex-1 flex-col gap-0.5">
        <span className="type-body-sm truncate font-semibold text-[var(--text)]" title={name}>
          {name}
        </span>
        <span className="type-caption-mono truncate text-[var(--text-muted)]">
          {[task.endpoint, task.task_id].filter(Boolean).join(' · ')}
        </span>
      </div>
      <div className="hidden w-32 shrink-0 items-center gap-2 sm:flex">
        <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-[var(--bg-card2)]">
          <div className="h-full rounded-full bg-[var(--accent)] transition-all" style={{ width: `${percent}%` }} />
        </div>
        <span className="type-caption-mono w-10 text-right text-[var(--text-muted)]">{Math.round(percent)}%</span>
      </div>
      <span className="type-caption-mono hidden shrink-0 text-[var(--text-muted)] md:block">
        {formatTimestamp(task.started_at, 'seconds')}
      </span>
      <button
        type="button"
        onClick={() => onAdopt(task.kind === 'perf' ? 'perf' : 'eval', task.task_id)}
        className="shrink-0 cursor-pointer rounded-[var(--radius-sm)] border border-[var(--border-md)] px-3 py-1.5 text-sm text-[var(--text-muted)] transition-colors hover:bg-[var(--bg-card2)] hover:text-[var(--text)]"
      >
        {t('tasks.activeAdopt')}
      </button>
    </div>
  )
}

/**
 * Live list of the service's running tasks. Answers "is it still running?"
 * after a reload, from another tab, or from another machine: any registered
 * subprocess appears here and can be attached to its runner panel.
 *
 * Renders nothing when no task is active so the launch forms stay focused.
 */
export default function ActiveTasksPanel({ onAdopt }: ActiveTasksPanelProps) {
  const { t } = useLocale()
  const { data } = usePolling<ListActiveTasksResponse>({ fn: () => listActiveTasks(), interval: 5000, enabled: true })
  const tasks = data?.tasks ?? []
  if (tasks.length === 0) return null

  return (
    <Card
      title={t('tasks.activeTitle')}
      badge={(
        <span className="inline-flex items-center gap-1 rounded-full bg-[var(--success-bg)] px-2 py-0.5 text-xs font-medium text-[var(--success)]">
          <Activity size={12} />
          {tasks.length}
        </span>
      )}
    >
      <div className="divide-y divide-[var(--border)]">
        {tasks.map((task) => (
          <ActiveTaskRow key={task.task_id} task={task} onAdopt={onAdopt} />
        ))}
      </div>
    </Card>
  )
}
