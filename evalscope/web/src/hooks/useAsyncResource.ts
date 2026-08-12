import { useCallback, useEffect, useRef, useState } from 'react'
import { isDomainError } from '@/api/errors'

/** Outcome of a single async read, plus the control to run it again. */
export interface AsyncResource<T> {
  /** Latest successful value, or `undefined` before the first one arrives. */
  data: T | undefined
  /** True while a read is in flight. */
  loading: boolean
  /** Human-readable failure message; empty string when there is none. */
  error: string
  /** Re-run the read with the current inputs. */
  reload: () => void
  /** Replace the value locally, for views that mutate what they just loaded. */
  setData: (next: T) => void
}

interface AsyncResourceOptions {
  /**
   * Skip the read while false — for inputs that are not ready yet (no root path,
   * no selected subset). A disabled resource is never loading and never errors.
   */
  enabled?: boolean
  /** Message used when the rejection carries none. */
  fallbackMessage?: string
}

/**
 * Read a value that lives behind an async call, keyed on its inputs.
 *
 * Every read gets an `AbortSignal` and the previous read is aborted whenever the
 * inputs change, so only the newest one can settle the state. An aborted read is
 * not a failure: its outcome is dropped without touching `error` or `loading`,
 * which is what keeps a superseded request from flashing an error on a view that
 * has already moved on.
 *
 * The last successful value is kept across a reload so a refresh does not blank
 * the view it is refreshing.
 *
 * @param fetcher Reads the value; receives the signal for the current attempt.
 *   Re-created on every render by design — it is held in a ref rather than
 *   joining the dependency list, so `deps` alone decides when a read happens.
 * @param deps Inputs the read is keyed on, in the `useEffect` dependency sense.
 * @param options Gating and the fallback failure message.
 */
export function useAsyncResource<T>(
  fetcher: (signal: AbortSignal) => Promise<T>,
  deps: readonly unknown[],
  options: AsyncResourceOptions = {},
): AsyncResource<T> {
  const { enabled = true, fallbackMessage = 'Failed to load' } = options

  const [data, setData] = useState<T | undefined>(undefined)
  // Seeded from `enabled` so the very first render already reports the read that
  // the mount effect is about to start; otherwise the view paints its empty
  // state for one frame before the load begins.
  const [loading, setLoading] = useState(enabled)
  const [error, setError] = useState('')
  const [reloadToken, setReloadToken] = useState(0)

  // The fetcher closes over fresh props on every render; reading it through a
  // ref keeps its identity out of the effect's dependency list.
  const fetcherRef = useRef(fetcher)
  useEffect(() => { fetcherRef.current = fetcher }, [fetcher])

  const messageRef = useRef(fallbackMessage)
  useEffect(() => { messageRef.current = fallbackMessage }, [fallbackMessage])

  useEffect(() => {
    if (!enabled) return
    const controller = new AbortController()

    const read = async () => {
      setLoading(true)
      setError('')
      try {
        const value = await fetcherRef.current(controller.signal)
        if (controller.signal.aborted) return
        setData(value)
      } catch (err) {
        // A superseded read aborts; drop its outcome rather than surfacing it.
        if (controller.signal.aborted || (isDomainError(err) && err.kind === 'aborted')) return
        setError(err instanceof Error ? err.message : messageRef.current)
      } finally {
        if (!controller.signal.aborted) setLoading(false)
      }
    }
    read()

    return () => controller.abort()
    // `deps` is the caller's dependency list, spread here on purpose: this hook
    // exists to key a read on inputs it cannot name.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [enabled, reloadToken, ...deps])

  const reload = useCallback(() => setReloadToken((token) => token + 1), [])
  const replace = useCallback((next: T) => setData(next), [])

  return {
    data,
    loading: enabled ? loading : false,
    error: enabled ? error : '',
    reload,
    setData: replace,
  }
}
