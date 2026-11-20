import type { ReactNode } from 'react'
import { useLocale } from '@/contexts/LocaleContext'
import { connectionPayload, type ModelConnection, type PerfDraft, type FieldErrors } from './taskConfig'
import PerfConfigForm from '@/components/perf/PerfConfigForm'
import TaskRunnerShell, { type TaskSelectionProps } from '@/components/tasks/TaskRunnerShell'
import { submitPerfTask, stopPerfTask, getPerfLog } from '@/api/perf'

interface Props extends TaskSelectionProps {
  connection: ModelConnection
  draft: PerfDraft
  onChange: (draft: PerfDraft) => void
  toolbar: ReactNode
  onValidateConnection: () => FieldErrors
}

export default function PerfTaskPanel({
  connection,
  draft,
  onChange,
  toolbar,
  onValidateConnection,
  ...selection
}: Props) {
  const { t } = useLocale()
  return (
    <TaskRunnerShell
      {...selection}
      idPrefix="perf"
      title={t('perf.task.title')}
      configTitle={t('perf.task.config')}
      statusTitle={t('perf.task.status')}
      readyLabel={t('perf.task.ready')}
      submitTask={(payload, id) => submitPerfTask({ ...payload, ...connectionPayload(connection, 'perf') }, id)}
      stopTask={stopPerfTask}
      getLog={getPerfLog}
      renderForm={({ onSubmit, disabled }) => (
        <PerfConfigForm
          onSubmit={onSubmit}
          disabled={disabled}
          draft={draft}
          onChange={onChange}
          toolbar={toolbar}
          onValidateConnection={onValidateConnection}
        />
      )}
    />
  )
}
