import { useLocale } from '@/contexts/LocaleContext'
import type { ReactNode } from 'react'
import { connectionPayload, type ModelConnection, type EvalDraft, type FieldErrors } from './taskConfig'
import EvalConfigForm from '@/components/eval/EvalConfigForm'
import TaskRunnerShell, { type TaskSelectionProps } from '@/components/tasks/TaskRunnerShell'
import { submitEvalTask, stopEvalTask, getEvalLog } from '@/api/eval'

interface Props extends TaskSelectionProps {
  connection: ModelConnection
  draft: EvalDraft
  onChange: (draft: EvalDraft) => void
  toolbar: ReactNode
  onValidateConnection: () => FieldErrors
}

export default function EvalTaskPanel({
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
      idPrefix="eval"
      title={t('eval.title')}
      configTitle={t('eval.config')}
      statusTitle={t('eval.status')}
      readyLabel={t('eval.ready')}
      submitTask={(payload, id) => submitEvalTask({ ...payload, ...connectionPayload(connection, 'eval') }, id)}
      stopTask={stopEvalTask}
      getLog={getEvalLog}
      renderForm={({ onSubmit, disabled }) => (
        <EvalConfigForm
          onSubmit={onSubmit}
          disabled={disabled || connection.api === 'local'}
          draft={draft}
          onChange={onChange}
          toolbar={toolbar}
          onValidateConnection={onValidateConnection}
        />
      )}
    />
  )
}
