import { useState } from 'react'
import EvalConfigForm from '@/components/eval/EvalConfigForm'
import PerfConfigForm from '@/components/perf/PerfConfigForm'
import ModelConnectionEditor from '@/components/tasks/ModelConnectionEditor'
import {
  connectionPayload,
  DEFAULT_CONNECTION,
  DEFAULT_EVAL,
  DEFAULT_PERF,
  editConnection,
  validateConnection,
  type FieldErrors,
} from '@/components/tasks/taskConfig'

interface Props {
  onSubmit: (payload: Record<string, unknown>) => void
}

/** Exercises real connection and form components without invoking a task service. */
function TaskFormHarness({ kind, onSubmit }: Props & { kind: 'eval' | 'perf' }) {
  const [connection, setConnection] = useState({ ...DEFAULT_CONNECTION })
  const [evalDraft, setEvalDraft] = useState({ ...DEFAULT_EVAL })
  const [perfDraft, setPerfDraft] = useState({ ...DEFAULT_PERF })
  const [errors, setErrors] = useState<FieldErrors>({})
  function checkConnection() {
    const next = validateConnection(connection)
    setErrors(next)
    return Object.fromEntries(Object.entries(next).map(([key, value]) => [`connection-${key}`, value]))
  }
  function submit(payload: Record<string, unknown>) {
    onSubmit({ ...payload, ...connectionPayload(connection, kind) })
  }
  return (
    <>
      <ModelConnectionEditor
        connection={connection}
        entries={[]}
        selectedId=""
        errors={errors}
        rememberKey={false}
        onRememberKeyChange={() => {}}
        onChange={(patch) => {
          setConnection(editConnection(connection, patch))
          setErrors({})
        }}
        onLoad={() => {}}
        onSave={() => false}
        onReset={() => {}}
        onDelete={() => {}}
        onForgetKey={() => {}}
      />
      {kind === 'eval' ? (
        <EvalConfigForm
          onSubmit={submit}
          draft={evalDraft}
          onChange={setEvalDraft}
          onValidateConnection={checkConnection}
          disabled={connection.api === 'local'}
        />
      ) : (
        <PerfConfigForm
          onSubmit={submit}
          draft={perfDraft}
          onChange={setPerfDraft}
          onValidateConnection={checkConnection}
        />
      )}
    </>
  )
}

export function EvalFormHarness(props: Props) {
  return <TaskFormHarness {...props} kind="eval" />
}
export function PerfFormHarness(props: Props) {
  return <TaskFormHarness {...props} kind="perf" />
}
