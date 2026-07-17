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
}

export function ApiUrlField({ id, name, value, onChange }: ApiUrlFieldProps) {
  return (
    <Field id={id} name={name} labelKey="eval.apiUrl" autoComplete="url">
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
