import { useMemo } from 'react'
import { useLocale } from '@/contexts/LocaleContext'
import { useAsyncResource } from '@/hooks/useAsyncResource'
import { listPlugins } from '@/api/plugins'
import Field from '@/components/ui/Field'
import { inputClass } from '@/components/ui/formStyles'
import { pluginApiOptions, type PluginApiKind } from './taskConfig'

interface Props {
  id: string
  name: string
  kind: PluginApiKind
  labelKey: string
  hintKey: string
  value: string
  onChange: (value: string) => void
}

/**
 * Optional selector for a model API or perf protocol registered by a plugin the service
 * enabled. It renders nothing when the service has no such plugin, so the form is
 * unchanged for services started without `--plugin`.
 */
export default function PluginApiField({ id, name, kind, labelKey, hintKey, value, onChange }: Props) {
  const { t } = useLocale()
  // A failed read only hides the field: built-in protocols stay usable.
  const plugins = useAsyncResource(async (signal) => (await listPlugins(signal)).plugins, [])
  const options = useMemo(() => pluginApiOptions(plugins.data ?? [], kind), [plugins.data, kind])
  const stale = value !== '' && !options.some((option) => option.value === value)

  if (options.length === 0 && !value) return null
  return (
    <Field id={id} name={name} labelKey={labelKey} hintKey={hintKey}>
      {(aria) => (
        <select {...aria} className={inputClass()} value={value} onChange={(event) => onChange(event.target.value)}>
          <option value="">{t('tasks.plugin.none')}</option>
          {options.map((option) => (
            <option key={`${option.plugin}:${option.value}`} value={option.value}>
              {option.value} ({option.plugin})
            </option>
          ))}
          {stale && (
            <option value={value}>
              {value} ({t('tasks.plugin.unavailable')})
            </option>
          )}
        </select>
      )}
    </Field>
  )
}
