import { useLocale } from '@/contexts/LocaleContext'
import { useQueryParams } from '@/hooks/useQueryParams'
import Tabs from '@/components/ui/Tabs'
import EvalTaskPage from './EvalTaskPage'
import PerfTaskPage from './PerfTaskPage'

/**
 * Unified "run a task" page — hosts the Evaluation and Performance task
 * runners as sub-tabs. The active sub-tab is reflected in the `?tab=` query
 * param so it survives refresh / sharing.
 */
export default function TasksPage() {
  const { t } = useLocale()
  const { get, set } = useQueryParams()
  const tab = get('tab') === 'perf' ? 'perf' : 'eval'

  const tabs = [
    { key: 'eval', label: t('tasks.evalTab') },
    { key: 'perf', label: t('tasks.perfTab') },
  ]

  return (
    <div className="page-enter flex flex-col gap-4">
      <Tabs tabs={tabs} activeKey={tab} onChange={(k) => set('tab', k)} />
      {tab === 'perf' ? <PerfTaskPage /> : <EvalTaskPage />}
    </div>
  )
}
