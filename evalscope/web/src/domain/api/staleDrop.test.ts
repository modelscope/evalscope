// Feature: frontend-refactor-2026-07, Property 27: 仅最新请求更新界面
//
// For any sequence of numbered requests whose responses arrive out of order
// (including late responses from superseded requests), the stale-drop reducer's
// final displayed state must be determined *solely* by the latest request (the
// maximum sequence number). Late/stale responses (seq < latestSeq) must be
// dropped and must never update the displayed state.
//
// These properties model the consumer-side race scenario described in
// Requirements 13.6 (only the latest request's result updates the UI) and 13.7
// (a late/superseded response is discarded). The reducer is pure, so we can
// exercise arbitrary arrival orders directly and cross-check the outcome against
// an independent expected model derived only from the latest request's response.
//
// Validates: Requirements 13.6, 13.7

import { describe, expect, it } from 'vitest'
import fc from 'fast-check'

import { createInitialState, staleDropReducer, type StaleDropAction } from './staleDrop'

/** A per-request response outcome: either a success payload or a failure. */
type Outcome = { kind: 'resolve'; data: number } | { kind: 'reject'; error: string }

const outcomeArb: fc.Arbitrary<Outcome> = fc.oneof(
  fc.record({ kind: fc.constant('resolve' as const), data: fc.integer() }),
  fc.record({ kind: fc.constant('reject' as const), error: fc.string() }),
)

/**
 * A race scenario: `n` requests tagged with the strictly-increasing sequences
 * `0..n-1`, each request paired with an `Outcome`, plus an arbitrary arrival
 * order (`arrival`) modelling responses racing back out of order.
 */
interface Scenario {
  n: number
  outcomes: Outcome[]
  arrival: number[]
}

const scenarioArb: fc.Arbitrary<Scenario> = fc
  .integer({ min: 1, max: 8 })
  .chain((n) => {
    const seqs = Array.from({ length: n }, (_, i) => i)
    return fc
      .tuple(...seqs.map(() => outcomeArb))
      .chain((outcomes) =>
        // A full-length shuffled subarray of every seq is a random permutation,
        // i.e. an arbitrary arrival order for the n responses.
        fc
          .shuffledSubarray(seqs, { minLength: n, maxLength: n })
          .map((arrival) => ({ n, outcomes: outcomes as Outcome[], arrival })),
      )
  })

/** Convert an outcome for a given seq into the corresponding response action. */
function responseAction(seq: number, outcome: Outcome): StaleDropAction<number, string> {
  return outcome.kind === 'resolve'
    ? { type: 'resolve', seq, data: outcome.data }
    : { type: 'reject', seq, error: outcome.error }
}

/**
 * Reduce a race scenario: dispatch every request in increasing seq order, then
 * deliver all responses in the given arrival order. This mirrors n in-flight
 * requests whose responses race back after the frontier has advanced to the
 * latest request.
 */
function runScenario(n: number, outcomes: Outcome[], arrival: number[]) {
  let state = createInitialState<number, string>()
  for (let seq = 0; seq < n; seq += 1) {
    state = staleDropReducer(state, { type: 'request', seq })
  }
  for (const seq of arrival) {
    state = staleDropReducer(state, responseAction(seq, outcomes[seq]))
  }
  return state
}

describe('staleDropReducer (Property 27: 仅最新请求更新界面)', () => {
  it('final displayed state is determined solely by the latest request (Req 13.6, 13.7)', () => {
    fc.assert(
      fc.property(scenarioArb, ({ n, outcomes, arrival }) => {
        const state = runScenario(n, outcomes, arrival)

        // Independent expected model: only the maximum sequence (the latest
        // request) may determine the displayed state; every lower-seq response
        // is stale once all requests are dispatched, so it is dropped.
        const maxSeq = n - 1
        const latest = outcomes[maxSeq]

        expect(state.latestSeq).toBe(maxSeq)
        expect(state.displayedSeq).toBe(maxSeq)

        if (latest.kind === 'resolve') {
          expect(state.status).toBe('success')
          expect(state.data).toBe(latest.data)
          expect(state.error).toBeNull()
        } else {
          expect(state.status).toBe('error')
          expect(state.error).toBe(latest.error)
          // No non-stale success was ever applied, so data stays at its initial
          // null (a reject preserves the previously displayed data).
          expect(state.data).toBeNull()
        }
      }),
    )
  })

  it('arrival order of racing responses never changes the displayed state (Req 13.7)', () => {
    const twoArrivalsArb = fc.integer({ min: 1, max: 8 }).chain((n) => {
      const seqs = Array.from({ length: n }, (_, i) => i)
      return fc
        .tuple(...seqs.map(() => outcomeArb))
        .chain((outcomes) =>
          fc
            .tuple(
              fc.shuffledSubarray(seqs, { minLength: n, maxLength: n }),
              fc.shuffledSubarray(seqs, { minLength: n, maxLength: n }),
            )
            .map(([a, b]) => ({ n, outcomes: outcomes as Outcome[], a, b })),
        )
    })

    fc.assert(
      fc.property(twoArrivalsArb, ({ n, outcomes, a, b }) => {
        const first = runScenario(n, outcomes, a)
        const second = runScenario(n, outcomes, b)

        // The two runs share the same requests and responses but differ only in
        // arrival order; the displayed outcome must be identical.
        expect(second.displayedSeq).toBe(first.displayedSeq)
        expect(second.latestSeq).toBe(first.latestSeq)
        expect(second.status).toBe(first.status)
        expect(second.data).toBe(first.data)
        expect(second.error).toBe(first.error)
      }),
    )
  })

  it('a late response from a superseded request is dropped without touching state (Req 13.7)', () => {
    const lateArb = fc
      .integer({ min: 1, max: 8 })
      .chain((latestSeq) =>
        fc.record({
          latestSeq: fc.constant(latestSeq),
          latestOutcome: outcomeArb,
          staleSeq: fc.integer({ min: 0, max: latestSeq - 1 }),
          staleOutcome: outcomeArb,
        }),
      )

    fc.assert(
      fc.property(lateArb, ({ latestSeq, latestOutcome, staleSeq, staleOutcome }) => {
        let state = createInitialState<number, string>()
        state = staleDropReducer(state, { type: 'request', seq: latestSeq })
        state = staleDropReducer(state, responseAction(latestSeq, latestOutcome))

        const before = state
        const after = staleDropReducer(state, responseAction(staleSeq, staleOutcome))

        // The stale response predates the frontier, so the reducer returns the
        // exact same state reference — the UI is untouched.
        expect(after).toBe(before)
      }),
    )
  })

  it('drops every earlier response even when the latest resolve arrives first (concrete example)', () => {
    let state = createInitialState<number, string>()
    state = staleDropReducer(state, { type: 'request', seq: 0 })
    state = staleDropReducer(state, { type: 'request', seq: 1 })
    state = staleDropReducer(state, { type: 'request', seq: 2 })

    // Latest (seq 2) resolves first, then the slower earlier responses arrive.
    state = staleDropReducer(state, { type: 'resolve', seq: 2, data: 202 })
    state = staleDropReducer(state, { type: 'resolve', seq: 0, data: 200 })
    state = staleDropReducer(state, { type: 'reject', seq: 1, error: 'boom' })

    expect(state.displayedSeq).toBe(2)
    expect(state.status).toBe('success')
    expect(state.data).toBe(202)
    expect(state.error).toBeNull()
  })
})
