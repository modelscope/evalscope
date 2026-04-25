import { useLocale } from '@/contexts/LocaleContext'
import { useQueryParams } from '@/hooks/useQueryParams'

export default function ReportViewerPage() {
  const { t } = useLocale()
  const { get } = useQueryParams()
  const url = get('url')

  if (!url) {
    return (
      <div className="flex items-center justify-center h-[60vh] text-[var(--color-ink-muted)]">
        <p>No report URL specified. Add <code>?url=...</code> to the URL.</p>
      </div>
    )
  }

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <h1 className="text-lg font-semibold">Report Viewer</h1>
        <a
          href={url}
          target="_blank"
          rel="noreferrer"
          className="text-sm text-[var(--color-primary)] hover:underline"
        >
          {t('common.openNewTab')}
        </a>
      </div>
      <iframe
        src={url}
        title="Report"
        className="w-full border border-[var(--color-border)] rounded-lg bg-white"
        style={{ height: 'calc(100vh - 140px)' }}
        sandbox="allow-scripts allow-same-origin"
      />
    </div>
  )
}
