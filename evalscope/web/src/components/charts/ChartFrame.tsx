import { useCallback, useEffect, useReducer, type ReactElement } from 'react'
import { AlertTriangle, RotateCcw } from 'lucide-react'
import { cn } from '@/lib/utils'
import Skeleton from '@/components/ui/Skeleton'
import { useLocale } from '@/contexts/LocaleContext'
import DataTableFallback, { type DataTableModel } from '@/components/common/DataTableFallback'
import { withTheme, type ChartTheme } from '@/domain/chart/chartTheme'
import { classifyChartResponse, type ChartFailureKind } from '@/domain/chart/chartPreflight'

/** Public phase of the chart lifecycle exposed to consumers (Req 2.6, 2.7). */
export type ChartPhase =
  | { status: 'loading' }
  | { status: 'ready' }
  | { status: 'error'; kind: ChartFailureKind }

export interface ChartFrameProps {
  /**
   * Base chart URL. The active `theme` is injected by this component via
   * {@link withTheme}; callers must not pre-append a theme parameter (Req 2.1).
   */
  baseSrc: string
  /** Active theme carried on every chart request path (Req 2.1, 2.2). */
  theme: ChartTheme
  height?: number
  className?: string
  title?: string
  /** Authoritative data-table fallback presenting the same data (Req 2.8). */
  fallbackTable: DataTableModel
  /** Preflight timeout in milliseconds; defaults to 10000 (Req 2.5). */
  preflightTimeoutMs?: number
  /** Optional callback invoked when the user triggers a retry. */
  onRetry?: () => void
}

/** Default preflight timeout: 10 seconds (Req 2.5). */
const DEFAULT_PREFLIGHT_TIMEOUT_MS = 10000

/**
 * Internal state machine. `loading` optionally renders the iframe: `iframe` is
 * `false` while the preflight is in flight and `true` once preflight succeeds
 * and we are waiting for the iframe's own `load` event. The error state never
 * renders an iframe so a blank iframe is never shown (Req 2.6).
 */
type InternalPhase =
  | { status: 'loading'; iframe: boolean }
  | { status: 'ready' }
  | { status: 'error'; kind: ChartFailureKind }

type PhaseAction =
  | { type: 'reset' }
  | { type: 'preflight-ok' }
  | { type: 'iframe-loaded' }
  | { type: 'fail'; kind: ChartFailureKind }

function phaseReducer(_state: InternalPhase, action: PhaseAction): InternalPhase {
  switch (action.type) {
    case 'reset':
      return { status: 'loading', iframe: false }
    case 'preflight-ok':
      return { status: 'loading', iframe: true }
    case 'iframe-loaded':
      return { status: 'ready' }
    case 'fail':
      return { status: 'error', kind: action.kind }
  }
}

/** Map a failure kind to its localized message key. */
function errorMessageKey(kind: ChartFailureKind): string {
  switch (kind) {
    case '4xx':
      return 'charts.error4xx'
    case '5xx':
      return 'charts.error5xx'
    case 'timeout':
      return 'charts.errorTimeout'
    case 'network':
      return 'charts.errorNetwork'
  }
}

/**
 * Theme-aware chart frame with preflight, visible loading/error states and an
 * authoritative data-table fallback.
 *
 * Behavior:
 * - Injects the active `theme` into the request URL, so a light theme never
 *   renders a dark chart (Req 2.1, 2.2).
 * - Re-runs the preflight and reloads the iframe whenever the theme changes, so
 *   an already-rendered chart re-renders to match the new theme (Req 2.3). The
 *   themed URL is a `useEffect` dependency, so the browser reloads the iframe
 *   immediately (well within 1s).
 * - Preflights the chart request with `fetch` + `AbortSignal`; a request that
 *   does not respond within `preflightTimeoutMs` is aborted and classified as a
 *   `timeout` failure (Req 2.4, 2.5).
 * - Shows a visible loading state while the preflight (and subsequent iframe
 *   load) is in flight, and only decides on an error state after loading
 *   completes (Req 2.7).
 * - On failure renders visible error text plus a retry control and never shows
 *   a blank iframe (Req 2.6).
 * - Always offers the authoritative data-table fallback when the chart cannot
 *   render (Req 2.8).
 * - All loading and error copy is localized (Req 2.9).
 */
export default function ChartFrame({
  baseSrc,
  theme,
  height = 400,
  className,
  title,
  fallbackTable,
  preflightTimeoutMs = DEFAULT_PREFLIGHT_TIMEOUT_MS,
  onRetry,
}: ChartFrameProps): ReactElement {
  const { t } = useLocale()

  // The themed URL is the single dependency that drives the preflight. It
  // changes when either `baseSrc` or `theme` changes (Req 2.1, 2.3).
  const themedSrc = withTheme(baseSrc, theme)

  const [phase, dispatch] = useReducer(phaseReducer, { status: 'loading', iframe: false })

  // `retryNonce` forces the preflight effect to re-run on an explicit retry
  // even when the themed URL is unchanged (Req 2.6).
  const [retryNonce, bumpRetry] = useReducer((n: number) => n + 1, 0)

  useEffect(() => {
    const controller = new AbortController()
    let cancelled = false
    let timedOut = false

    dispatch({ type: 'reset' })

    // Abort the request if it does not respond within the timeout budget. The
    // abort is distinguished from a cleanup abort via the `timedOut` flag so it
    // is classified as a `timeout` failure (Req 2.5).
    const timeoutId = window.setTimeout(() => {
      timedOut = true
      controller.abort()
    }, preflightTimeoutMs)

    const runPreflight = async (): Promise<void> => {
      try {
        const response = await fetch(themedSrc, { method: 'GET', signal: controller.signal })
        window.clearTimeout(timeoutId)
        if (cancelled) return
        const responseClass = classifyChartResponse(response.status)
        if (responseClass === 'success') {
          // Preflight passed; render the iframe and wait for its load event
          // before declaring success (Req 2.7).
          dispatch({ type: 'preflight-ok' })
        } else {
          dispatch({ type: 'fail', kind: responseClass })
        }
      } catch {
        window.clearTimeout(timeoutId)
        if (timedOut) {
          // A timeout is a user-visible failure even though the abort was ours.
          if (!cancelled) dispatch({ type: 'fail', kind: 'timeout' })
          return
        }
        // A cleanup-driven abort (dependency change/unmount) is not a failure.
        if (cancelled) return
        dispatch({ type: 'fail', kind: 'network' })
      }
    }

    void runPreflight()

    return () => {
      cancelled = true
      window.clearTimeout(timeoutId)
      controller.abort()
    }
  }, [themedSrc, preflightTimeoutMs, retryNonce])

  const handleIframeLoad = useCallback(() => {
    dispatch({ type: 'iframe-loaded' })
  }, [])

  const handleIframeError = useCallback(() => {
    dispatch({ type: 'fail', kind: 'network' })
  }, [])

  const handleRetry = useCallback(() => {
    onRetry?.()
    bumpRetry()
  }, [onRetry])

  const isLoading = phase.status === 'loading'
  const isError = phase.status === 'error'
  // Render the iframe once the preflight passes and keep it mounted while
  // ready. It is never mounted in the error state (Req 2.6).
  const showIframe = (phase.status === 'loading' && phase.iframe) || phase.status === 'ready'

  return (
    <div
      className={cn(
        'rounded-[var(--radius)] border border-[var(--border)] bg-[var(--bg-card)] overflow-hidden',
        className,
      )}
    >
      {title && (
        <div className="px-4 py-2.5 border-b border-[var(--border)]">
          <h4 className="type-label-xs">{title}</h4>
        </div>
      )}
      {isError ? (
        // Error state: visible error text + retry entry point + authoritative
        // data-table fallback. No iframe is rendered (Req 2.6, 2.8, 2.9).
        <div className="p-4">
          <div
            role="alert"
            className="flex flex-col items-center justify-center gap-3 py-8 text-center text-[var(--text-muted)]"
          >
            <AlertTriangle size={24} aria-hidden="true" />
            <div className="flex flex-col gap-1">
              <span className="text-sm font-medium text-[var(--text)]">{t('charts.loadError')}</span>
              <span className="text-sm">{t(errorMessageKey(phase.kind))}</span>
            </div>
            <button
              type="button"
              onClick={handleRetry}
              className="inline-flex min-h-[44px] items-center gap-1.5 rounded-[var(--radius)] border border-[var(--border)] px-4 py-2 text-sm font-medium text-[var(--text)] transition-colors hover:bg-[var(--bg-card2)]"
            >
              <RotateCcw size={14} aria-hidden="true" />
              {t('charts.retry')}
            </button>
          </div>
          <DataTableFallback model={fallbackTable} className="border-t border-[var(--border)] pt-4" />
        </div>
      ) : (
        <div className="relative" style={{ height }}>
          {isLoading && (
            <div
              className="absolute inset-0 flex flex-col items-center justify-center gap-3 p-6"
              role="status"
              aria-live="polite"
            >
              <Skeleton width="100%" height="100%" />
              <span className="absolute text-sm text-[var(--text-muted)]">{t('charts.loading')}</span>
            </div>
          )}
          {showIframe && (
            <iframe
              src={themedSrc}
              className={cn('w-full h-full border-0 transition-opacity duration-200', isLoading && 'opacity-0')}
              sandbox="allow-scripts allow-same-origin"
              onLoad={handleIframeLoad}
              onError={handleIframeError}
              title={title ?? 'Chart'}
            />
          )}
        </div>
      )}
    </div>
  )
}
