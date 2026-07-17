import { useLocale } from '@/contexts/LocaleContext'
import PerfConfigForm from '@/components/perf/PerfConfigForm'
import TaskRunnerPage from '@/components/tasks/TaskRunnerPage'
import { submitPerfTask, stopPerfTask, getPerfProgress, getPerfLog, getPerfReportUrl } from '@/api/perf'

export default function PerfTaskPage() {
  const { t } = useLocale()
  return (
    <TaskRunnerPage
      idPrefix="perf"
      title={t('perf.title')}
      configTitle={t('perf.config')}
      statusTitle={t('perf.status')}
      readyLabel={t('perf.ready')}
      submitTask={submitPerfTask}
      stopTask={stopPerfTask}
      getProgress={getPerfProgress}
      getLog={getPerfLog}
      getReportUrl={getPerfReportUrl}
      renderForm={({ onSubmit, disabled }) => <PerfConfigForm onSubmit={onSubmit} disabled={disabled} />}
    />
  )
}
