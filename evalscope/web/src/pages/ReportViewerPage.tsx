import { useLocale } from '@/contexts/LocaleContext'
import { useQueryParams } from '@/hooks/useQueryParams'

/**
 * Only allow same-origin relative URLs starting with /api/v1/reports/
 * to prevent open-redirect and javascript:/data: injection.
 */
function isSafeReportUrl(url: string): boolean {
  // Must be a relative path starting with the reports API prefix
  return /^\/api\/v1\/reports\//.test(url)
}

export default function ReportViewerPage() {
  const { t } = useLocale()
  const { get } = useQueryParams()
  const rawUrl = get('url')
  const url = rawUrl && isSafeReportUrl(rawUrl) ? rawUrl : null

  if (!rawUrl) {
    return (
      <div className="flex items-center justify-center h-[60vh] text-[var(--color-ink-muted)]">
        <p>No report URL specified. Add <code>?url=...</code> to the URL.</p>
      </div>
    )
  }

  if (!url) {
    return (
      <div className="flex items-center justify-center h-[60vh] text-[var(--color-ink-muted)]">
        <p>Invalid report URL. Only same-origin report paths are allowed.</p>
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
        className="w-full border border-[var(--color-border)] rounded-lg bg-[var(--bg-card)]"
        style={{ height: 'calc(100vh - 140px)' }}
        sandbox="allow-scripts allow-same-origin"
      />
    </div>
  )
}
