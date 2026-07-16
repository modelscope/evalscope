/**
 * Stale-drop reducer for consumer-side request race handling (Req 13.5–13.7).
 *
 * When a view issues successive requests (e.g. a detail load whose inputs change,
 * or a debounced search), responses can arrive out of order: a slower earlier
 * request may resolve *after* a faster later one. Rendering whichever response
 * happens to arrive last would show data that no longer matches the current
 * inputs. This reducer makes the displayed state depend solely on the *latest*
 * request, dropping any late/stale response from a superseded request.
 *
 * The reducer is intentionally pure (no DOM/network/clock/AbortController): the
 * consuming hook owns request initiation and cancellation and simply dispatches
 * numbered actions here. That keeps the ordering logic directly testable
 * (property test 15.6 / Property 27).
 *
 * Sequencing model:
 *  - Every request is tagged with a monotonically-increasing sequence number.
 *  - `latestSeq` tracks the newest request the view has initiated (the frontier).
 *  - A `resolve`/`reject` is applied only when its `seq` is *not older* than
 *    `latestSeq`; a response whose `seq < latestSeq` belongs to a superseded
 *    request and is dropped (Req 13.6, 13.7).
 *
 * The generic error type `E` defaults to `unknown` so this module stays
 * decoupled from the API/error layer; consumers typically specialise it to
 * their domain error type.
 */

/** Lifecycle status of the tracked request/response cycle. */
export type StaleDropStatus = 'idle' | 'loading' | 'success' | 'error'

/**
 * Reducer state.
 *
 * `data`/`error` are retained across a new `request` so the UI keeps its existing
 * state while a fresh load is in flight and while surfacing an error state
 * (Req 13.4). `displayedSeq` records which request's response is currently
 * reflected (`-1` when nothing has been displayed yet).
 */
export interface StaleDropState<T, E = unknown> {
  /** Sequence number of the most recently initiated request (the frontier). */
  latestSeq: number
  /** Sequence number whose response is currently reflected; `-1` if none. */
  displayedSeq: number
  status: StaleDropStatus
  /** Latest applied success payload, or `null` when never resolved / reset. */
  data: T | null
  /** Latest applied failure, or `null` when not in an error state. */
  error: E | null
}

/**
 * Actions dispatched by the consuming hook.
 *
 * - `request`: a new request was initiated with sequence `seq`.
 * - `resolve`: request `seq` completed successfully with `data`.
 * - `reject`: request `seq` failed with `error`.
 * - `reset`: clear back to the initial state.
 */
export type StaleDropAction<T, E = unknown> =
  | { type: 'request'; seq: number }
  | { type: 'resolve'; seq: number; data: T }
  | { type: 'reject'; seq: number; error: E }
  | { type: 'reset' }

/** Build a fresh initial state. */
export function createInitialState<T, E = unknown>(): StaleDropState<T, E> {
  return {
    latestSeq: -1,
    displayedSeq: -1,
    status: 'idle',
    data: null,
    error: null,
  }
}

/**
 * Whether a response tagged with `seq` is stale relative to `state`.
 *
 * A response is stale when a newer request has since been initiated
 * (`seq < latestSeq`); such responses must be dropped (Req 13.7).
 */
export function isStaleResponse<T, E>(state: StaleDropState<T, E>, seq: number): boolean {
  return seq < state.latestSeq
}

/**
 * Pure stale-drop reducer.
 *
 * Semantics:
 *  - `request` advances the frontier to `seq` (ignoring out-of-order/duplicate
 *    requests whose `seq <= latestSeq`) and enters `loading`, preserving any
 *    previously displayed `data`/`error` so the UI does not flicker to empty.
 *  - `resolve`/`reject` are applied only when the response is not stale
 *    (`seq >= latestSeq`); a stale response returns the state unchanged so a
 *    superseded request never updates the UI (Req 13.6, 13.7).
 *  - On a fresh `resolve` the frontier is pinned to `seq` so any later-arriving
 *    lower-sequence response is still recognised as stale even if its own
 *    `request` action was never observed.
 *  - `reject` retains the previous `data` so the view keeps its existing state
 *    while showing the error (Req 13.4).
 */
export function staleDropReducer<T, E = unknown>(
  state: StaleDropState<T, E>,
  action: StaleDropAction<T, E>,
): StaleDropState<T, E> {
  switch (action.type) {
    case 'request': {
      // Ignore stale/duplicate request initiations so the frontier only moves
      // forward.
      if (action.seq <= state.latestSeq) return state
      return { ...state, latestSeq: action.seq, status: 'loading' }
    }
    case 'resolve': {
      if (isStaleResponse(state, action.seq)) return state
      return {
        ...state,
        latestSeq: action.seq,
        displayedSeq: action.seq,
        status: 'success',
        data: action.data,
        error: null,
      }
    }
    case 'reject': {
      if (isStaleResponse(state, action.seq)) return state
      return {
        ...state,
        latestSeq: action.seq,
        displayedSeq: action.seq,
        status: 'error',
        error: action.error,
        // Keep `data` so the UI retains its existing state alongside the error.
      }
    }
    case 'reset':
      return createInitialState<T, E>()
    default:
      return state
  }
}
