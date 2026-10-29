import { useRef, useState } from 'react'
import { Download, Upload } from 'lucide-react'
import { useLocale } from '@/contexts/LocaleContext'
import Button from '@/components/ui/Button'
import ConfirmDialog from '@/components/ui/ConfirmDialog'
import { exportLibrary, MAX_IMPORT_BYTES, parseImport, type ConfigLibrary } from './configStorage'
import Switch from '@/components/ui/Switch'

interface Props {
  library: ConfigLibrary
  onImport: (library: ConfigLibrary) => boolean
  onClear: () => void
}

export default function ConfigLibraryTools({ library, onImport, onClear }: Props) {
  const { t } = useLocale()
  const fileRef = useRef<HTMLInputElement>(null)
  const [incoming, setIncoming] = useState<ConfigLibrary | null>(null)
  const [error, setError] = useState('')
  const [includeKeys, setIncludeKeys] = useState(false)
  const [exportOpen, setExportOpen] = useState(false)
  const [clearOpen, setClearOpen] = useState(false)

  async function importFile(file?: File) {
    if (!file) return
    setError('')
    try {
      if (file.size > MAX_IMPORT_BYTES) throw new Error('File too large')
      setIncoming(parseImport(await file.text()))
    } catch {
      setError(t('tasks.library.importError'))
    }
    if (fileRef.current) fileRef.current.value = ''
  }

  function download() {
    const url = URL.createObjectURL(new Blob([exportLibrary(library, includeKeys)], { type: 'application/json' }))
    const anchor = document.createElement('a')
    anchor.href = url
    anchor.download = 'evalscope-config-library.json'
    anchor.click()
    URL.revokeObjectURL(url)
    setExportOpen(false)
  }

  const importItems = incoming
    ? [
        t('tasks.library.importConnections', { count: incoming.connections.length }),
        t('tasks.library.importEval', { count: incoming.evalTemplates.length }),
        t('tasks.library.importPerf', { count: incoming.perfTemplates.length }),
      ]
    : []
  const hasKeys = incoming?.connections.some((entry) => Boolean(entry.value.apiKey))

  return (
    <div className="min-w-0 space-y-2">
      <div className="flex flex-wrap items-center gap-2">
        <input
          ref={fileRef}
          type="file"
          accept="application/json,.json"
          className="hidden"
          aria-label={t('tasks.library.import')}
          onChange={(event) => {
            void importFile(event.target.files?.[0])
          }}
        />
        <Button type="button" variant="outline" size="sm" onClick={() => fileRef.current?.click()}>
          <Upload size={14} />
          {t('tasks.library.import')}
        </Button>
        <Button
          type="button"
          variant="outline"
          size="sm"
          onClick={() => {
            setIncludeKeys(false)
            setExportOpen(true)
          }}
        >
          <Download size={14} />
          {t('tasks.library.export')}
        </Button>
        <details
          className="relative"
          onKeyDown={(event) => {
            if (event.key === 'Escape') event.currentTarget.open = false
          }}
        >
          <summary className="coarse-target cursor-pointer rounded-[var(--radius-sm)] px-2 py-2 text-xs">
            {t('tasks.library.actions')}
          </summary>
          <div className="absolute right-0 z-30 w-48 rounded-[var(--radius-sm)] border border-[var(--border)] bg-[var(--bg-card)] p-2 shadow-[var(--shadow)]">
            <Button type="button" variant="ghost" size="sm" onClick={() => setClearOpen(true)}>
              {t('tasks.library.clear')}
            </Button>
          </div>
        </details>
      </div>
      {error && (
        <p role="alert" className="text-xs text-[var(--danger)]">
          {error}
        </p>
      )}
      <ConfirmDialog
        open={Boolean(incoming)}
        title={t('tasks.library.import')}
        items={importItems}
        message={t(hasKeys ? 'tasks.library.plaintextWarning' : 'tasks.library.importHint')}
        confirmLabel={t('tasks.library.confirm')}
        cancelLabel={t('tasks.library.cancel')}
        onCancel={() => setIncoming(null)}
        onConfirm={() => {
          if (incoming && onImport(incoming)) setIncoming(null)
        }}
      />
      <ConfirmDialog
        open={exportOpen}
        title={t('tasks.library.export')}
        message={t(includeKeys ? 'tasks.library.exportKeysWarning' : 'tasks.library.exportHint')}
        confirmLabel={t('tasks.library.export')}
        cancelLabel={t('tasks.library.cancel')}
        onCancel={() => setExportOpen(false)}
        onConfirm={download}
      >
        <div className="flex items-center gap-2 text-xs">
          <Switch
            id="library-includeKeys"
            aria-labelledby="library-includeKeys-label"
            checked={includeKeys}
            onCheckedChange={setIncludeKeys}
          />
          <label id="library-includeKeys-label" htmlFor="library-includeKeys">
            {t('tasks.library.includeKeys')}
          </label>
        </div>
      </ConfirmDialog>
      <ConfirmDialog
        open={clearOpen}
        title={t('tasks.library.clear')}
        message={t('tasks.library.clearHint')}
        danger
        confirmLabel={t('tasks.library.confirm')}
        cancelLabel={t('tasks.library.cancel')}
        onCancel={() => setClearOpen(false)}
        onConfirm={() => {
          onClear()
          setClearOpen(false)
        }}
      />
    </div>
  )
}
