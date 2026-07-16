import { useEffect, useMemo, useState, type ReactNode } from 'react'
import { useNavigate } from 'react-router-dom'
import { AlertCircle, Inbox, SearchX } from 'lucide-react'
import { useLocale } from '@/contexts/LocaleContext'
import { localeDictionaries, type Locale } from '@/i18n/translations'
import {
  buildEmptyStateSpec,
  resolveEmptyState,
  type EmptyLocaleMaps,
  type EmptyReason,
  type EmptyStateContext,
  type ResolvedEmptyStateAction,
} from '@/domain/empty/emptyState'
import EmptyState from '@/components/common/EmptyState'
import Button from '@/components/ui/Button'
import { cn } from '@/lib/utils'

/** Default locale used when the active locale lacks a string (Req 6.5). */
const FALLBACK_LOCALE: Locale = 'en'

/**
 * Upper bound on how long the empty state may take to appear after a load
 * completes (Req 6.1). The reveal delay is clamped to this budget so the empty
 * state is always shown within 300ms.
 */
export const MAX_REVEAL_DELAY_MS = 300

type NestedDict = { [key: string]: string | NestedDict }

/** Flatten a nested translation dictionary into dot-separated keys. */
function flattenDict(dict: NestedDict, prefix: string, out: Record<string, string>): Record<string, string> {
  for (const [key, value] of Object.entries(dict)) {
    const path = prefix ? `${prefix}.${key}` : key
    if (typeof value === 'string') {
      out[path] = value
    } else {
      flattenDict(value, path, out)
    }
  }
  return out
}

/**
 * Flattened locale maps derived once from the static translation dictionaries.
 * Keys that are absent from a locale are simply missing from its map, which is
 * exactly what {@link resolveEmptyState} needs to trigger fallback (Req 6.5).
 */
const LOCALE_MAPS: EmptyLocaleMaps = Object.fromEntries(
  Object.entries(localeDictionaries).map(([locale, dict]) => [
    locale,
    flattenDict(dict as NestedDict, '', {}),
  ]),
)

/** Default reason-specific icon (28px Lucide, matching `EmptyState`). */
const REASON_ICON: Record<EmptyReason, ReactNode> = {
  'no-data': <Inbox size={28} strokeWidth={1.5} />,
  'load-error': <AlertCircle size={28} strokeWidth={1.5} />,
  'no-match': <SearchX size={28} strokeWidth={1.5} />,
}

export interface EmptyStateSystemProps {
  /** The reason the view is empty; selects the message + default actions (Req 6.1). */
  reason: EmptyReason
  /**
   * When true the empty state is suppressed. Callers gate this on their loading
   * flag so the empty state never flashes mid-load and only appears once a load
   * has completed (Req 6.1).
   */
  loading?: boolean
  /** View context selecting default recovery targets and route overrides (Req 6.2, 6.3). */
  context?: EmptyStateContext
  /** Override the default reason icon. */
  icon?: ReactNode
  /** Optional already-localized secondary hint rendered under the message. */
  hint?: string
  className?: string
  /**
   * Per-action handler invoked before navigation. Return `true` to indicate the
   * action was handled in-view (e.g. retry or clear-filters) and suppress the
   * default route navigation. Any other return value falls through to navigation.
   */
  onAction?: (action: ResolvedEmptyStateAction) => boolean | void
  /**
   * Delay before the empty state is revealed once loading completes. Clamped to
   * `[0, MAX_REVEAL_DELAY_MS]`; defaults to 0 (revealed on the next render).
   */
  revealDelayMs?: number
}

function RevealAfterDelay({ delay, children }: { delay: number; children: ReactNode }) {
  const [visible, setVisible] = useState(delay === 0)

  useEffect(() => {
    if (delay === 0) return
    const timer = window.setTimeout(() => setVisible(true), delay)
    return () => window.clearTimeout(timer)
  }, [delay])

  return visible ? children : null
}

/**
 * Empty_State_System — actionable empty state (Req 6.1–6.5).
 *
 * Consumes the pure {@link buildEmptyStateSpec}/{@link resolveEmptyState} logic to
 * render a reason-specific message plus 1–3 in-product recovery actions. Each
 * action navigates to its corresponding in-product flow via the router (Req 6.2);
 * callers may intercept individual actions through {@link EmptyStateSystemProps.onAction}
 * to handle in-view recovery (retry, clear filters) without leaving the page.
 */
export default function EmptyStateSystem({
  reason,
  loading = false,
  context,
  icon,
  hint,
  className,
  onAction,
  revealDelayMs = 0,
}: EmptyStateSystemProps) {
  const { locale } = useLocale()
  const navigate = useNavigate()
  const delay = Math.min(Math.max(revealDelayMs, 0), MAX_REVEAL_DELAY_MS)

  const resolved = useMemo(() => {
    const spec = buildEmptyStateSpec(reason, context ?? {})
    return resolveEmptyState(spec, locale, FALLBACK_LOCALE, LOCALE_MAPS)
  }, [reason, context, locale])

  if (loading) return null

  return (
    <RevealAfterDelay key={`${reason}:${delay}`} delay={delay}>
      <div className={cn('flex flex-col items-center gap-4', className)}>
        <EmptyState icon={icon ?? REASON_ICON[reason]} title={resolved.message} hint={hint} className="py-8" />
        {resolved.actions.length > 0 && (
          <div className="flex flex-wrap items-center justify-center gap-2 -mt-2 pb-8">
            {resolved.actions.map((action, index) => (
              <Button
                key={action.navigateTo}
                variant={index === 0 ? 'primary' : 'outline'}
                size="sm"
                onClick={() => {
                  if (onAction?.(action) === true) return
                  navigate(action.navigateTo)
                }}
              >
                {action.label}
              </Button>
            ))}
          </div>
        )}
      </div>
    </RevealAfterDelay>
  )
}
