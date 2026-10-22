import { useState } from 'react'
import { Eye, EyeOff, PlugZap } from 'lucide-react'
import { useLocale } from '@/contexts/LocaleContext'
import Card from '@/components/ui/Card'
import Field from '@/components/ui/Field'
import Switch from '@/components/ui/Switch'
import Button from '@/components/ui/Button'
import ConfirmDialog from '@/components/ui/ConfirmDialog'
import { inputClass } from '@/components/ui/formStyles'
import { AliasField, ApiUrlField, ModelField } from './TaskFormFields'
import PresetToolbar from './PresetToolbar'
import { DEFAULT_CONNECTION, type ModelConnection, type FieldErrors } from './taskConfig'
import type { ConnectionPreset } from './configStorage'

interface Props {
  connection: ModelConnection
  entries: ConnectionPreset[]
  selectedId: string
  errors: FieldErrors
  rememberKey: boolean
  onRememberKeyChange: (value: boolean) => void
  onChange: (patch: Partial<ModelConnection>) => void
  onLoad: (id: string) => void
  onSave: (name: string, asNew: boolean, rememberKey: boolean) => boolean
  onDelete: () => void
  onReset: () => void
  onForgetKey: () => void
}

export default function ModelConnectionEditor({
  connection,
  entries,
  selectedId,
  errors,
  rememberKey,
  onRememberKeyChange,
  onChange,
  onLoad,
  onSave,
  onDelete,
  onReset,
  onForgetKey,
}: Props) {
  const { t } = useLocale()
  const selected = entries.find((entry) => entry.id === selectedId)
  const [showKey, setShowKey] = useState(false)
  const [pendingSave, setPendingSave] = useState<{ name: string; asNew: boolean } | null>(null)
  const [forgetOpen, setForgetOpen] = useState(false)
  const dirty =
    JSON.stringify({ ...connection, apiKey: rememberKey ? connection.apiKey : '' }) !==
      JSON.stringify(selected?.value ?? DEFAULT_CONNECTION) || rememberKey !== (selected?.rememberKey ?? false)

  function save(name: string, asNew: boolean): boolean {
    if (rememberKey && connection.apiKey && !selected?.rememberKey) {
      setPendingSave({ name, asNew })
      return false
    }
    return onSave(name, asNew, rememberKey)
  }
  function load(id: string) {
    onLoad(id)
    setShowKey(false)
  }
  function reset() {
    onReset()
    onRememberKeyChange(false)
    setShowKey(false)
  }

  return (
    <Card>
      <div className="mb-4 flex items-start gap-3">
        <PlugZap size={20} className="mt-0.5 shrink-0 text-[var(--accent)]" aria-hidden="true" />
        <div>
          <h2 className="text-base font-semibold">{t('tasks.library.connection')}</h2>
          <p className="mt-1 text-xs text-[var(--text-muted)]">{t('tasks.library.sharedHint')}</p>
        </div>
      </div>
      <PresetToolbar
        label={t('tasks.library.connection')}
        entries={entries}
        selectedId={selectedId}
        dirty={dirty}
        onLoad={load}
        onSave={save}
        onDelete={onDelete}
        onReset={reset}
      />
      <div className="mt-4 grid grid-cols-1 gap-4 md:grid-cols-2">
        <Field id="connection-api" name="api" labelKey="perf.task.apiType" hintKey="perf.task.apiHint">
          {(aria) => (
            <select
              {...aria}
              className={inputClass()}
              value={connection.api}
              onChange={(event) => {
                const api = event.target.value
                if (api === 'openai' || api === 'local') onChange({ api })
              }}
            >
              <option value="openai">OpenAI-compatible</option>
              <option value="local">Local Transformers</option>
            </select>
          )}
        </Field>
        <ModelField
          id="connection-model"
          value={connection.model}
          error={errors.model ? t(errors.model) : undefined}
          onChange={(model) => onChange({ model })}
        />
        {connection.api === 'openai' && (
          <>
            <ApiUrlField
              id="connection-url"
              name="api_url"
              value={connection.url}
              required
              error={errors.url ? t(errors.url) : undefined}
              onChange={(url) => onChange({ url })}
            />
            <Field id="connection-apiKey" name="api_key" labelKey="eval.apiKey" autoComplete="off">
              {(aria) => (
                <div className="flex items-center gap-2">
                  <input
                    {...aria}
                    type={showKey ? 'text' : 'password'}
                    className={inputClass()}
                    value={connection.apiKey}
                    onChange={(event) => onChange({ apiKey: event.target.value })}
                    placeholder="sk-..."
                  />
                  <Button
                    type="button"
                    size="sm"
                    variant="ghost"
                    aria-label={t(showKey ? 'tasks.library.hideKey' : 'tasks.library.showKey')}
                    aria-pressed={showKey}
                    onClick={() => setShowKey(!showKey)}
                  >
                    {showKey ? <EyeOff size={16} /> : <Eye size={16} />}
                  </Button>
                </div>
              )}
            </Field>
          </>
        )}
        <AliasField id="connection-alias" value={connection.alias} onChange={(alias) => onChange({ alias })} />
      </div>
      {connection.api === 'local' ? (
        <p className="mt-3 text-sm text-[var(--text-muted)]">{t('tasks.library.localOnly')}</p>
      ) : (
        <div className="mt-4 border-t border-[var(--border)] pt-3">
          <div className="flex flex-wrap items-center gap-2">
            <Switch
              id="connection-rememberKey"
              aria-labelledby="connection-rememberKey-label"
              checked={rememberKey}
              onCheckedChange={onRememberKeyChange}
            />
            <label
              id="connection-rememberKey-label"
              htmlFor="connection-rememberKey"
              className="text-xs text-[var(--text)]"
            >
              {t('tasks.library.rememberKey')}
            </label>
            <Button
              type="button"
              variant="ghost"
              size="sm"
              disabled={!connection.apiKey && !selected?.value.apiKey}
              onClick={() => setForgetOpen(true)}
            >
              {t('tasks.library.forgetKey')}
            </Button>
          </div>
          {rememberKey && (
            <p className="mt-1 text-xs text-[var(--text-muted)]">{t('tasks.library.plaintextWarning')}</p>
          )}
        </div>
      )}
      <ConfirmDialog
        open={Boolean(pendingSave)}
        title={t('tasks.library.rememberKey')}
        message={t('tasks.library.plaintextWarning')}
        confirmLabel={t('tasks.library.confirm')}
        cancelLabel={t('tasks.library.cancel')}
        onCancel={() => setPendingSave(null)}
        onConfirm={() => {
          if (pendingSave) onSave(pendingSave.name, pendingSave.asNew, true)
          setPendingSave(null)
        }}
      />
      <ConfirmDialog
        open={forgetOpen}
        title={t('tasks.library.forgetKey')}
        message={t('tasks.library.forgetHint')}
        danger
        confirmLabel={t('tasks.library.confirm')}
        cancelLabel={t('tasks.library.cancel')}
        onCancel={() => setForgetOpen(false)}
        onConfirm={() => {
          onForgetKey()
          onRememberKeyChange(false)
          setShowKey(false)
          setForgetOpen(false)
        }}
      />
    </Card>
  )
}
