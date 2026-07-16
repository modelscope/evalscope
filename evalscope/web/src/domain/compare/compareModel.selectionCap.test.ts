import { describe, expect, it } from 'vitest'
import fc from 'fast-check'

import { addToSelection, MAX_COMPARE_SELECTION } from './compareModel'

// Feature: frontend-refactor-2026-07, Property 10: 对比选择上限
//
// For any current selection of run ids, `addToSelection` must enforce the
// selection cap and de-duplicate:
//   - When the selection is already at or above MAX_COMPARE_SELECTION (5), the
//     addition is rejected (`rejected === true`) and the selection is returned
//     unchanged.
//   - When below the cap and the run is already selected, the run is de-duped:
//     the selection is unchanged and the addition is not rejected.
//   - When below the cap and the run is new, it is appended: the size grows by
//     one, never exceeds the cap, and the result stays de-duplicated.
//
// Validates: Requirements 5.9
describe('addToSelection (Property 10: compare selection cap)', () => {
  // A set of unique run ids of size 0..8, so we exercise below-cap, at-cap and
  // above-cap states around MAX_COMPARE_SELECTION (5).
  const uniqueSelection = fc.uniqueArray(fc.string(), { minLength: 0, maxLength: 8 })
  // A run id to add; drawn from the same string space so it may or may not
  // already be present in the selection.
  const runId = fc.string()

  it('MAX_COMPARE_SELECTION is 5', () => {
    expect(MAX_COMPARE_SELECTION).toBe(5)
  })

  it('rejects and leaves the selection unchanged when at or above the cap', () => {
    fc.assert(
      fc.property(uniqueSelection, runId, (state, id) => {
        fc.pre(state.length >= MAX_COMPARE_SELECTION)
        const { next, rejected } = addToSelection(state, id)
        expect(rejected).toBe(true)
        // Selection is returned unchanged (same reference and contents).
        expect(next).toBe(state)
        expect(next).toEqual(state)
      }),
    )
  })

  it('de-duplicates without rejecting when below the cap and the run already exists', () => {
    fc.assert(
      fc.property(uniqueSelection, (state) => {
        fc.pre(state.length < MAX_COMPARE_SELECTION && state.length > 0)
        // Pick an id that is already present in the selection.
        const existing = state[state.length - 1]
        const { next, rejected } = addToSelection(state, existing)
        expect(rejected).toBe(false)
        // Already selected: selection is unchanged.
        expect(next).toEqual(state)
        expect(next.length).toBe(state.length)
      }),
    )
  })

  it('appends and grows by one (deduped, <= cap) when below the cap and the run is new', () => {
    fc.assert(
      fc.property(uniqueSelection, runId, (state, id) => {
        fc.pre(state.length < MAX_COMPARE_SELECTION && !state.includes(id))
        const { next, rejected } = addToSelection(state, id)
        expect(rejected).toBe(false)
        expect(next.length).toBe(state.length + 1)
        expect(next.length).toBeLessThanOrEqual(MAX_COMPARE_SELECTION)
        expect(next).toContain(id)
        // Result stays de-duplicated.
        expect(new Set(next).size).toBe(next.length)
      }),
    )
  })

  it('keeps a within-cap selection within the cap and de-duplicated', () => {
    fc.assert(
      fc.property(uniqueSelection, runId, (state, id) => {
        // Starting from any within-cap selection, adding a run never exceeds the
        // cap and always yields a de-duplicated result.
        fc.pre(state.length <= MAX_COMPARE_SELECTION)
        const { next } = addToSelection(state, id)
        expect(next.length).toBeLessThanOrEqual(MAX_COMPARE_SELECTION)
        expect(new Set(next).size).toBe(next.length)
      }),
    )
  })
})
