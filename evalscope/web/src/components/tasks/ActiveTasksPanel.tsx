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
  selectedTaskId?: string | null
}

interface ActiveTaskRowProps {
  task: ActiveTaskInfo
  onAdopt: ActiveTasksPanelProps['onAdopt']
  selected: boolean
}

function ActiveTaskRow({ task, onAdopt, selected }: ActiveTaskRowProps) {
  const { t } = useLocale()
  const kindLabel = task.kind === 'perf' ? t('tasks.activeKindPerf') : t('tasks.activeKindEval')
  const name = task.alias || task.model || task.task_id
  let percent: number | null = null
  if (typeof task.percent === 'number' && Number.isFinite(task.percent)) {
    percent = Math.max(0, Math.min(100, task.percent))
  }
  const progressLabel = percent === null ? t(`tasks.state.${task.status}`) : `${Math.round(percent)}%`

  return (
    <div className={`flex flex-wrap items-center gap-3 px-5 py-3 ${selected ? 'bg-[var(--accent-dim)]' : ''}`}>
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
      <div className="flex w-32 shrink-0 items-center gap-2">
        <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-[var(--bg-card2)]">
          <div
            role="progressbar"
            aria-label={name}
            aria-valuenow={percent ?? undefined}
            aria-valuemin={0}
            aria-valuemax={100}
            className={`h-full rounded-full bg-[var(--accent)] transition-all ${percent === null ? 'animate-pulse' : ''}`}
            style={{ width: percent === null ? '30%' : `${percent}%` }}
          />
        </div>
        <span className="type-caption-mono w-10 text-right text-[var(--text-muted)]">{progressLabel}</span>
      </div>
      <span className="type-caption-mono hidden shrink-0 text-[var(--text-muted)] md:block">
        {formatTimestamp(task.started_at, 'seconds')}
      </span>
      <button
        type="button"
        aria-pressed={selected}
        aria-label={`${t('tasks.activeAdopt')}: ${name}`}
        onClick={() => onAdopt(task.kind === 'perf' ? 'perf' : 'eval', task.task_id)}
        className="shrink-0 cursor-pointer rounded-[var(--radius-sm)] border border-[var(--border-md)] px-3 py-1.5 text-sm text-[var(--text-muted)] transition-colors hover:bg-[var(--bg-card2)] hover:text-[var(--text)]"
      >
        {t(selected ? 'tasks.selected' : 'tasks.activeAdopt')}
      </button>
    </div>
  )
}

/**
 * Live list of the service's running tasks. Answers "is it still running?"
 * after a reload, from another tab, or from another machine: any registered
 * subprocess appears here and can be attached to its runner panel.
 */
export default function ActiveTasksPanel({ onAdopt, selectedTaskId }: ActiveTasksPanelProps) {
  const { t } = useLocale()
  const { data, error, refresh } = usePolling<ListActiveTasksResponse>({
    fn: () => listActiveTasks(),
    interval: 5000,
    enabled: true,
  })
  const tasks = data?.tasks ?? []

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
      {error && (
        <div role="alert" className="p-4 text-sm text-[var(--danger)]">
          {t(data ? 'tasks.staleData' : 'tasks.connectionError')}
          <button type="button" className="ml-2 underline" onClick={() => refresh()}>
            {t('common.retry')}
          </button>
        </div>
      )}
      {!data && !error && (
        <p role="status" className="p-4 text-sm">{t('common.loading')}</p>
      )}
      {data && tasks.length === 0 && (
        <p className="p-4 text-sm text-[var(--text-muted)]">{t('tasks.empty')}</p>
      )}
      <div className="divide-y divide-[var(--border)]">
        {tasks.map((task) => (
          <ActiveTaskRow
            key={task.task_id}
            task={task}
            onAdopt={onAdopt}
            selected={task.task_id === selectedTaskId}
          />
        ))}
      </div>
    </Card>
  )
}
