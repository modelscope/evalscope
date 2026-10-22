import Field from '@/components/ui/Field'
import { FORM_INPUT_CLASS, inputClass } from '@/components/ui/formStyles'

interface TaskTextFieldProps {
  id: string
  value: string
  onChange: (value: string) => void
}

interface ModelFieldProps extends TaskTextFieldProps {
  error?: string
}

export function ModelField({ id, value, error, onChange }: ModelFieldProps) {
  return (
    <Field id={id} name="model" labelKey="eval.modelName" required error={error} autoComplete="off">
      {(aria) => (
        <input
          {...aria}
          type="text"
          value={value}
          onChange={(event) => onChange(event.target.value)}
          className={inputClass(error)}
          placeholder="Qwen/Qwen2.5-0.5B-Instruct"
        />
      )}
    </Field>
  )
}

interface ApiUrlFieldProps extends TaskTextFieldProps {
  name: 'api_url' | 'url'
  error?: string
  required?: boolean
}

export function ApiUrlField({ id, name, value, onChange, error, required }: ApiUrlFieldProps) {
  return (
    <Field id={id} name={name} labelKey="eval.apiUrl" autoComplete="url" required={required} error={error} hintKey="tasks.apiUrlHint">
      {(aria) => (
        <input
          {...aria}
          type="text"
          value={value}
          onChange={(event) => onChange(event.target.value)}
          className={FORM_INPUT_CLASS}
          placeholder="http://localhost:8000/v1"
        />
      )}
    </Field>
  )
}

export function AliasField({ id, value, onChange }: TaskTextFieldProps) {
  return (
    <Field id={id} name="alias" labelKey="tasks.alias" autoComplete="off" hintKey="tasks.aliasHint">
      {(aria) => (
        <input
          {...aria}
          type="text"
          value={value}
          onChange={(event) => onChange(event.target.value)}
          className={FORM_INPUT_CLASS}
          placeholder="server-A-8xGPU"
        />
      )}
    </Field>
  )
}

export function ApiKeyField({ id, value, onChange }: TaskTextFieldProps) {
  return (
    <Field id={id} name="api_key" labelKey="eval.apiKey" autoComplete="off">
      {(aria) => (
        <input
          {...aria}
          type="password"
          value={value}
          onChange={(event) => onChange(event.target.value)}
          className={FORM_INPUT_CLASS}
          placeholder="sk-..."
        />
      )}
    </Field>
  )
}
