/**
 * Empty State System — pure logic (Req 6.1, 6.2, 6.4, 6.5).
 *
 * This module builds framework-agnostic empty-state specifications and resolves
 * their localized text. It has no dependency on React, the DOM, the network, the
 * system clock, or randomness, which makes it the target of property tests
 * (Property 11: action count & navigability; Property 12: locale fallback).
 *
 * Rendering (300ms display timing, live regions, navigation wiring) lives in the
 * component layer (task 10.4); this module only produces the data contract.
 */

/** The three distinguishable reasons a view can be empty (Req 6.1). */
export type EmptyReason = 'no-data' | 'load-error' | 'no-match'

/**
 * A single in-product recovery action (Req 6.2).
 * `navigateTo` is an in-product target route; it is always non-empty for actions
 * produced by {@link buildEmptyStateSpec}.
 */
export interface EmptyStateAction {
  /** Localized label key, resolved via the locale system (Req 6.4). */
  labelKey: string
  /** In-product destination route triggered by the action (Req 6.2). */
  navigateTo: string
}

/** The empty-state contract consumed by the rendering layer. */
export interface EmptyStateSpec {
  reason: EmptyReason
  /** Localized message key describing the reason (Req 6.1, 6.4). */
  messageKey: string
  /** Between {@link MIN_EMPTY_ACTIONS} and {@link MAX_EMPTY_ACTIONS} actions (Req 6.2). */
  actions: EmptyStateAction[]
}

/** Minimum number of recovery actions in a spec (Req 6.2). */
export const MIN_EMPTY_ACTIONS = 1
/** Maximum number of recovery actions in a spec (Req 6.2). */
export const MAX_EMPTY_ACTIONS = 3

/** Logical views that can surface an empty state; drives recovery targets. */
export type EmptyStateView =
  | 'dashboard'
  | 'reports'
  | 'evaluations'
  | 'compare'
  | 'performance'
  | 'perf-compare'
  | 'benchmarks'

/** In-product routes (kept in sync with the router in `App.tsx`). */
const VIEW_ROUTES: Record<EmptyStateView, string> = {
  dashboard: '/dashboard',
  reports: '/reports',
  evaluations: '/reports',
  compare: '/compare',
  performance: '/performance',
  'perf-compare': '/perf-compare',
  benchmarks: '/benchmarks',
}

/** Unified task-creation flow route. */
const TASKS_ROUTE = '/tasks'
/** Fallback route guaranteeing at least one navigable action exists. */
const FALLBACK_ROUTE = '/dashboard'

/**
 * Optional context that tunes the recovery actions for a specific view.
 * Any missing field falls back to sensible in-product defaults so that the
 * invariants of {@link buildEmptyStateSpec} always hold.
 */
export interface EmptyStateContext {
  /** The view requesting the empty state; selects default recovery targets. */
  view?: EmptyStateView
  /** Route used to retry a failed load; defaults to the current view route. */
  retryTo?: string
  /** Route used to clear filters on `no-match`; defaults to the current view route. */
  clearFiltersTo?: string
  /** Route used to start task creation; defaults to a tab-scoped `/tasks` route. */
  createTaskTo?: string
  /** Additional caller-supplied actions, sanitized and capped like the rest. */
  extraActions?: EmptyStateAction[]
}

/** Resolve the task-creation route, tab-scoped when the view is known. */
function resolveCreateTaskRoute(context: EmptyStateContext): string {
  if (context.createTaskTo && context.createTaskTo.trim().length > 0) {
    return context.createTaskTo
  }
  switch (context.view) {
    case 'performance':
    case 'perf-compare':
      return `${TASKS_ROUTE}?tab=perf`
    case 'reports':
    case 'evaluations':
    case 'compare':
      return `${TASKS_ROUTE}?tab=eval`
    default:
      return TASKS_ROUTE
  }
}

/** Resolve the route for the current view, falling back to the dashboard. */
function resolveViewRoute(context: EmptyStateContext): string {
  return context.view ? VIEW_ROUTES[context.view] : FALLBACK_ROUTE
}

/**
 * Produce the default (unsanitized) recovery actions for a reason.
 * The output is always sanitized and bounded by {@link buildEmptyStateSpec}.
 */
function defaultActions(reason: EmptyReason, context: EmptyStateContext): EmptyStateAction[] {
  const createTaskRoute = resolveCreateTaskRoute(context)
  const viewRoute = resolveViewRoute(context)

  switch (reason) {
    case 'no-data':
      // No records yet — steer the user toward creating work (Req 6.2, 6.3).
      return [
        { labelKey: 'empty.action.createTask', navigateTo: createTaskRoute },
        { labelKey: 'empty.action.browseBenchmarks', navigateTo: VIEW_ROUTES.benchmarks },
      ]
    case 'load-error':
      // Loading failed — offer retry plus a safe navigation fallback.
      return [
        {
          labelKey: 'empty.action.retry',
          navigateTo: context.retryTo && context.retryTo.trim().length > 0 ? context.retryTo : viewRoute,
        },
        { labelKey: 'empty.action.backToDashboard', navigateTo: FALLBACK_ROUTE },
      ]
    case 'no-match':
      // Filters excluded everything — offer to clear them or start fresh work.
      return [
        {
          labelKey: 'empty.action.clearFilters',
          navigateTo:
            context.clearFiltersTo && context.clearFiltersTo.trim().length > 0
              ? context.clearFiltersTo
              : viewRoute,
        },
        { labelKey: 'empty.action.createTask', navigateTo: createTaskRoute },
      ]
    default:
      return []
  }
}

/**
 * Sanitize actions: drop entries with empty/blank `navigateTo`, de-duplicate by
 * target, and cap at {@link MAX_EMPTY_ACTIONS}.
 */
function sanitizeActions(actions: EmptyStateAction[]): EmptyStateAction[] {
  const seen = new Set<string>()
  const result: EmptyStateAction[] = []
  for (const action of actions) {
    if (!action || typeof action.navigateTo !== 'string') continue
    const navigateTo = action.navigateTo.trim()
    if (navigateTo.length === 0) continue
    if (seen.has(navigateTo)) continue
    seen.add(navigateTo)
    result.push({ labelKey: action.labelKey, navigateTo })
    if (result.length >= MAX_EMPTY_ACTIONS) break
  }
  return result
}

/**
 * Build an {@link EmptyStateSpec} for a reason and optional view context.
 *
 * Invariants (Property 11): the returned `actions` always has between
 * {@link MIN_EMPTY_ACTIONS} and {@link MAX_EMPTY_ACTIONS} entries, and every
 * action has a non-empty in-product `navigateTo`.
 */
export function buildEmptyStateSpec(reason: EmptyReason, context: EmptyStateContext = {}): EmptyStateSpec {
  const candidateActions = [...defaultActions(reason, context), ...(context.extraActions ?? [])]
  const actions = sanitizeActions(candidateActions)

  // Guarantee at least one navigable action even if every candidate was invalid.
  if (actions.length === 0) {
    actions.push({ labelKey: 'empty.action.backToDashboard', navigateTo: FALLBACK_ROUTE })
  }

  return {
    reason,
    messageKey: `empty.${reason}.message`,
    actions,
  }
}

/** A flat map of locale key -> localized string for a single locale. */
export type LocaleTextMap = Record<string, string>

/** Locale code -> its text map. */
export type EmptyLocaleMaps = Record<string, LocaleTextMap>

/** Return true when a resolved value is a usable, non-empty string. */
function isNonEmpty(value: string | undefined): value is string {
  return typeof value === 'string' && value.trim().length > 0
}

/**
 * Resolve empty-state text for `key`, falling back to the default locale when the
 * current locale is missing the string (Req 6.5, Property 12).
 *
 * Resolution order: current locale -> fallback locale -> the key itself. The key
 * is returned as a last resort so callers never receive an empty string (recovery
 * actions therefore remain usable because their `navigateTo` is independent of
 * text resolution).
 */
export function resolveEmptyText(
  key: string,
  locale: string,
  fallbackLocale: string,
  localeMaps: EmptyLocaleMaps,
): string {
  const current = localeMaps[locale]?.[key]
  if (isNonEmpty(current)) return current

  const fallback = localeMaps[fallbackLocale]?.[key]
  if (isNonEmpty(fallback)) return fallback

  // Last resort: the key is non-empty by contract, keeping the UI non-blank.
  return key
}

/** An action whose label has been resolved to display text. */
export interface ResolvedEmptyStateAction {
  label: string
  /** Preserved verbatim from the spec so the action stays navigable (Req 6.5). */
  navigateTo: string
}

/** A fully resolved empty state ready for rendering. */
export interface ResolvedEmptyState {
  reason: EmptyReason
  message: string
  actions: ResolvedEmptyStateAction[]
}

/**
 * Resolve every localized key in a spec while preserving each action's
 * `navigateTo`. Missing current-locale strings fall back to the default locale
 * (Req 6.5); actions are never dropped, so recovery stays available.
 */
export function resolveEmptyState(
  spec: EmptyStateSpec,
  locale: string,
  fallbackLocale: string,
  localeMaps: EmptyLocaleMaps,
): ResolvedEmptyState {
  return {
    reason: spec.reason,
    message: resolveEmptyText(spec.messageKey, locale, fallbackLocale, localeMaps),
    actions: spec.actions.map((action) => ({
      label: resolveEmptyText(action.labelKey, locale, fallbackLocale, localeMaps),
      navigateTo: action.navigateTo,
    })),
  }
}
