/**
 * `useLatestRequest` — consumer-side request race handling (Req 13.5–13.7).
 *
 * Wraps the pure {@link staleDropReducer} with an `AbortController` and the
 * `ReportsContext` ref-mirror pattern so detail / search / list views can:
 *
 *  - cancel the in-flight request when inputs/dependencies change (Req 13.5),
 *  - update the UI only from the most recent request (Req 13.6),
 *  - drop late responses from superseded/aborted requests (Req 13.7).
 *
 * Validation failures propagate as {@link DomainError} and are surfaced via the
 * `error` state while the previously displayed `data` is retained, so the view
 * keeps its existing state and shows an error state without throwing (Req 13.4).
 * Aborted requests are swallowed (they are an expected consequence of a newer
 * request superseding an older one), not surfaced as user-visible errors.
 */
import { useCallback, useEffect, useReducer, useRef } from 'react'

import { DomainError, isDomainError } from '@/api/errors'
import {
  createInitialState,
  staleDropReducer,
  type StaleDropStatus,
} from '@/domain/api/staleDrop'

/** A request factory that performs the fetch under the supplied abort signal. */
export type LatestRequestFn<T> = (signal: AbortSignal) => Promise<T>

/** Public shape returned by {@link useLatestRequest}. */
export interface UseLatestRequestResult<T, E = DomainError> {
  /** Payload of the most recent successful request, or `null`. */
  data: T | null
  /** Error from the most recent failed (non-aborted) request, or `null`. */
  error: E | null
  /** Lifecycle status of the tracked request cycle. */
  status: StaleDropStatus
  /** Convenience flag: `true` while a request is in flight. */
  isLoading: boolean
  /**
   * Start a new request, cancelling any in-flight one first. Only the newest
   * request's outcome is applied to state; stale/aborted responses are dropped.
   */
  run: (fn: LatestRequestFn<T>) => Promise<void>
  /** Cancel the in-flight request without starting a new one. */
  cancel: () => void
  /** Reset back to the initial (idle) state and cancel any in-flight request. */
  reset: () => void
}

/** Narrow an unknown thrown value to a request cancellation. */
function isAbort(err: unknown): boolean {
  if (isDomainError(err)) return err.kind === 'aborted'
  if (typeof DOMException !== 'undefined' && err instanceof DOMException) return err.name === 'AbortError'
  return err instanceof Error && err.name === 'AbortError'
}

/**
 * Track the latest request and expose its result, discarding stale outcomes.
 *
 * @typeParam T Success payload type.
 * @typeParam E Error type surfaced on failure (defaults to {@link DomainError}).
 */
export function useLatestRequest<T, E = DomainError>(): UseLatestRequestResult<T, E> {
  const [state, dispatch] = useReducer(staleDropReducer<T, E>, undefined, createInitialState<T, E>)

  // Monotonically-increasing sequence for tagging requests, mirrored in a ref so
  // async continuations read a stable, up-to-date value without re-renders.
  const seqRef = useRef(0)
  // Controller for the in-flight request so a new/cancelled request can abort it.
  const abortRef = useRef<AbortController | null>(null)
  // Guard against dispatching after unmount.
  const mountedRef = useRef(true)

  useEffect(() => {
    mountedRef.current = true
    return () => {
      mountedRef.current = false
      abortRef.current?.abort()
    }
  }, [])

  const cancel = useCallback(() => {
    abortRef.current?.abort()
    abortRef.current = null
  }, [])

  const reset = useCallback(() => {
    cancel()
    dispatch({ type: 'reset' })
  }, [cancel])

  const run = useCallback(async (fn: LatestRequestFn<T>) => {
    // Cancel the previous in-flight request (Req 13.5) and open a fresh scope.
    abortRef.current?.abort()
    const controller = new AbortController()
    abortRef.current = controller

    const seq = ++seqRef.current
    dispatch({ type: 'request', seq })

    try {
      const data = await fn(controller.signal)
      // Drop the result if this request was aborted after resolving, or if the
      // component unmounted. The reducer additionally drops stale sequences.
      if (controller.signal.aborted || !mountedRef.current) return
      dispatch({ type: 'resolve', seq, data })
    } catch (err) {
      // Aborted requests are an expected outcome of being superseded; swallow
      // them rather than surfacing a user-visible error (Req 13.7).
      if (isAbort(err) || controller.signal.aborted || !mountedRef.current) return
      dispatch({ type: 'reject', seq, error: err as E })
    } finally {
      if (abortRef.current === controller) abortRef.current = null
    }
  }, [])

  return {
    data: state.data,
    error: state.error,
    status: state.status,
    isLoading: state.status === 'loading',
    run,
    cancel,
    reset,
  }
}
