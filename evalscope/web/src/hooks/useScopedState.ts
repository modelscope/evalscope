import { useCallback, useState } from 'react'

/**
 * State that belongs to a scope, and is abandoned when the scope changes.
 *
 * Views in this app hold choices that only mean something inside the data they
 * were made in: a selected subset, a page number, a ticked set of runs. When the
 * scope moves — a different report, a rescan, a new root — the old choice must
 * not carry over.
 *
 * The scope is compared at read time rather than cleared by an effect. That
 * matters for correctness, not just tidiness: an effect runs *after* the render,
 * so one frame would paint with the previous scope's value, and a write from an
 * async task that started under the old scope would land after the reset and
 * stick. Here a value written under a scope simply stops being readable once the
 * scope differs, so neither case can happen.
 *
 * Remounting via `key` would also reset the value, but it discards *all* state in
 * the subtree and re-runs its reads. This hook exists for the cases where only
 * part of a view's state follows the scope.
 *
 * Pass `null` as the fallback when the view needs to distinguish "the user has
 * not chosen in this scope" from a real value, and resolve the default itself:
 *
 * ```ts
 * const [picked, setTab] = useScopedState<TabKey | null>(scope, null)
 * const activeTab = picked ?? (singleRun ? 'runs' : 'overview')
 * ```
 *
 * Exactly one value is held, keyed by the scope it was written in. It is not
 * cleared on the way out, so returning to a scope reveals the choice made there
 * again — which is what lets a user leave a report and come back to the dataset
 * they were reading. A write under a second scope overwrites the first, so at
 * most one scope is ever remembered.
 *
 * @param scope Identity of the data the value belongs to. Build it from every
 *   input that changes the meaning of the value.
 * @param fallback Returned whenever the held value belongs to another scope.
 *   Must be a stable reference — a literal `[]` or `{}` written inline would give
 *   the setter a new identity on every render. Use a module-level constant.
 * @returns The in-scope value and a setter, which also accepts an updater.
 */
export function useScopedState<T>(
  scope: string,
  fallback: T,
): [T, (next: T | ((current: T) => T)) => void] {
  const [held, setHeld] = useState<{ scope: string; value: T }>(() => ({ scope, value: fallback }))

  const value = held.scope === scope ? held.value : fallback

  const setValue = useCallback((next: T | ((current: T) => T)) => {
    setHeld((prev) => {
      // An updater must resolve against the in-scope value, never the previous
      // scope's, or a stale choice would be merged into the new scope.
      const base = prev.scope === scope ? prev.value : fallback
      return {
        scope,
        value: typeof next === 'function' ? (next as (current: T) => T)(base) : next,
      }
    })
  }, [scope, fallback])

  return [value, setValue]
}
