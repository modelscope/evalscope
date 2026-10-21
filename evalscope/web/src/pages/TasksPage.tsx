import { useCallback, useState } from 'react'
import { useLocale } from '@/contexts/LocaleContext'
import { useQueryParams } from '@/hooks/useQueryParams'
import Tabs from '@/components/ui/Tabs'
import ActiveTasksPanel from '@/components/tasks/ActiveTasksPanel'
import EvalTaskPanel from '@/components/tasks/EvalTaskPanel'
import PerfTaskPanel from '@/components/tasks/PerfTaskPanel'

/**
 * Unified "run a task" page — hosts the Evaluation and Performance task
 * runners as sub-tabs. The active sub-tab is reflected in the `?tab=` query
 * param so it survives refresh / sharing.
 */
export default function TasksPage() {
  const { t } = useLocale()
  const { get, set } = useQueryParams()
  const tab = get('tab') === 'perf' ? 'perf' : 'eval'

  // Adopting a task from the active list seeds the panel's restore slot and
  // remounts it: the runner then picks the task up through exactly the same
  // sessionStorage restore path used after a page reload.
  const [adopted, setAdopted] = useState<{ kind: 'eval' | 'perf'; taskId: string } | null>(null)
  const handleAdopt = useCallback(
    (kind: 'eval' | 'perf', taskId: string) => {
      try {
        sessionStorage.setItem(`evalscope-active-task-${kind}`, taskId)
      } catch {
        // Monitoring still works in-session when browser storage is unavailable.
      }
      set('tab', kind)
      setAdopted({ kind, taskId })
    },
    [set],
  )

  const tabs = [
    { key: 'eval', label: t('tasks.evalTab'), panelId: 'tasks-eval-panel' },
    { key: 'perf', label: t('tasks.perfTab'), panelId: 'tasks-perf-panel' },
  ]

  return (
    <div className="page-enter flex flex-col gap-4">
      <ActiveTasksPanel onAdopt={handleAdopt} />
      <Tabs
        tabs={tabs}
        activeKey={tab}
        onChange={(k) => set('tab', k)}
        panels={{
          'tasks-eval-panel': <EvalTaskPanel key={adopted?.kind === 'eval' ? `adopt-${adopted.taskId}` : 'eval'} />,
          'tasks-perf-panel': <PerfTaskPanel key={adopted?.kind === 'perf' ? `adopt-${adopted.taskId}` : 'perf'} />,
        }}
      />
    </div>
  )
}
