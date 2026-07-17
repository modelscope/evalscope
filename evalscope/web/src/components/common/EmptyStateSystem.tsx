import { useEffect, useMemo, useState, type ReactNode } from 'react'
import { useNavigate } from 'react-router-dom'
import { AlertCircle, Inbox, SearchX } from 'lucide-react'
import { useLocale } from '@/contexts/LocaleContext'
import EmptyState from '@/components/common/EmptyState'
import Button from '@/components/ui/Button'
import { cn } from '@/lib/utils'

export const MAX_REVEAL_DELAY_MS = 300

export type EmptyReason = 'no-data' | 'load-error' | 'no-match'
export type EmptyStateView =
  | 'dashboard'
  | 'reports'
  | 'evaluations'
  | 'compare'
  | 'performance'
  | 'perf-compare'
  | 'benchmarks'

interface EmptyStateAction {
  labelKey: string
  navigateTo: string
}

export interface ResolvedEmptyStateAction {
  label: string
  navigateTo: string
}

export interface EmptyStateContext {
  view?: EmptyStateView
  retryTo?: string
  clearFiltersTo?: string
  createTaskTo?: string
  extraActions?: EmptyStateAction[]
}

const VIEW_ROUTES: Record<EmptyStateView, string> = {
  dashboard: '/dashboard',
  reports: '/reports',
  evaluations: '/reports',
  compare: '/compare',
  performance: '/performance',
  'perf-compare': '/perf-compare',
  benchmarks: '/benchmarks',
}

function viewRoute(context: EmptyStateContext): string {
  return context.view ? VIEW_ROUTES[context.view] : '/dashboard'
}

function createTaskRoute(context: EmptyStateContext): string {
  if (context.createTaskTo?.trim()) return context.createTaskTo.trim()
  if (context.view === 'performance' || context.view === 'perf-compare') return '/tasks?tab=perf'
  if (context.view === 'reports' || context.view === 'evaluations' || context.view === 'compare') {
    return '/tasks?tab=eval'
  }
  return '/tasks'
}

function actionsFor(reason: EmptyReason, context: EmptyStateContext): EmptyStateAction[] {
  const actions: EmptyStateAction[] = reason === 'no-data'
    ? [
        { labelKey: 'empty.action.createTask', navigateTo: createTaskRoute(context) },
        { labelKey: 'empty.action.browseBenchmarks', navigateTo: '/benchmarks' },
      ]
    : reason === 'load-error'
      ? [
          { labelKey: 'empty.action.retry', navigateTo: context.retryTo?.trim() || viewRoute(context) },
          { labelKey: 'empty.action.backToDashboard', navigateTo: '/dashboard' },
        ]
      : [
          { labelKey: 'empty.action.clearFilters', navigateTo: context.clearFiltersTo?.trim() || viewRoute(context) },
          { labelKey: 'empty.action.createTask', navigateTo: createTaskRoute(context) },
        ]

  const seen = new Set<string>()
  return [...actions, ...(context.extraActions ?? [])]
    .map((action) => ({ ...action, navigateTo: action.navigateTo.trim() }))
    .filter((action) => action.navigateTo && !seen.has(action.navigateTo) && seen.add(action.navigateTo))
    .slice(0, 3)
}

/** Default reason-specific icon (28px Lucide, matching `EmptyState`). */
const REASON_ICON: Record<EmptyReason, ReactNode> = {
  'no-data': <Inbox size={28} strokeWidth={1.5} />,
  'load-error': <AlertCircle size={28} strokeWidth={1.5} />,
  'no-match': <SearchX size={28} strokeWidth={1.5} />,
}

export interface EmptyStateSystemProps {
  reason: EmptyReason
  loading?: boolean
  context?: EmptyStateContext
  icon?: ReactNode
  hint?: string
  className?: string
  onAction?: (action: ResolvedEmptyStateAction) => boolean | void
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
  const { t } = useLocale()
  const navigate = useNavigate()
  const delay = Math.min(Math.max(revealDelayMs, 0), MAX_REVEAL_DELAY_MS)

  const resolved = useMemo(() => {
    const actions = actionsFor(reason, context ?? {}).map((action) => ({
      label: t(action.labelKey),
      navigateTo: action.navigateTo,
    }))
    return { message: t(`empty.${reason}.message`), actions }
  }, [reason, context, t])

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
