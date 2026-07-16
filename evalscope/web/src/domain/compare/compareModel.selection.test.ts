import { describe, expect, it } from 'vitest'
import fc from 'fast-check'

import { preserveSelectionAcrossReorder } from './compareModel'

// Feature: frontend-refactor-2026-07, Property 9: 排序/筛选保留选中集合
//
// For any current selection of run ids and any list reorder (sort/filter), the
// preserved selection must be identical to the pre-operation selection *as a
// set* (set identity). Reordering or filtering the list only changes how runs
// are arranged / which are visible; it must never add or drop a selected run
// (Req 5.8). This means:
//   - No selected run is lost, even when the reordered/filtered list omits it.
//   - No run is introduced that was not already selected.
//   - The result is de-duplicated (the selection is a set).
//   - Selected runs present in the reordered list follow that list's order.
//
// Validates: Requirements 5.8
describe('preserveSelectionAcrossReorder (Property 9: retain selection across reorder)', () => {
  // A selection of run ids of size 0..8. Uses a small id alphabet so that
  // reorderings and filters frequently overlap the selection.
  const runId = fc.string({ minLength: 1, maxLength: 4 })
  const selection = fc.uniqueArray(runId, { minLength: 0, maxLength: 8 })

  // Build a reordered/filtered list from the selection: shuffle it, drop some
  // entries (filter), and mix in some ids that are not selected (other runs in
  // the list). This exercises sort (reorder), filter (omission) and unrelated
  // rows all at once.
  const scenario = selection.chain((selected) => {
    const others = fc.uniqueArray(runId, { minLength: 0, maxLength: 6 })
    return fc.record({
      selected: fc.constant(selected),
      // A subset of the selection, reordered (models sort + filter).
      keptSubset: fc.subarray(selected).chain((sub) => fc.shuffledSubarray(sub, { minLength: 0, maxLength: sub.length })),
      others,
    })
  })

  const asSet = (xs: string[]): Set<string> => new Set(xs)

  it('preserves the selection set unchanged for any reorder/filter', () => {
    fc.assert(
      fc.property(scenario, ({ selected, keptSubset, others }) => {
        // The reordered list contains a filtered/sorted subset of the selection
        // interleaved with unrelated runs.
        const reorderedList = fc.sample(fc.shuffledSubarray([...keptSubset, ...others]), 1)[0]
        const result = preserveSelectionAcrossReorder(selected, reorderedList)

        // Set identity: the preserved selection equals the original selection.
        expect(asSet(result)).toEqual(asSet(selected))
        // Result is de-duplicated (it is a set).
        expect(new Set(result).size).toBe(result.length)
        // No run outside the original selection is introduced.
        for (const id of result) {
          expect(selected).toContain(id)
        }
      }),
    )
  })

  it('is idempotent under repeated application', () => {
    fc.assert(
      fc.property(scenario, ({ selected, keptSubset, others }) => {
        const reorderedList = fc.sample(fc.shuffledSubarray([...keptSubset, ...others]), 1)[0]
        const once = preserveSelectionAcrossReorder(selected, reorderedList)
        const twice = preserveSelectionAcrossReorder(once, reorderedList)
        expect(twice).toEqual(once)
      }),
    )
  })

  it('collapses duplicates in the incoming selection to a set', () => {
    fc.assert(
      fc.property(selection, fc.array(runId, { maxLength: 12 }), (base, dupSource) => {
        // Construct a selection that may contain duplicates.
        const withDuplicates = [...base, ...dupSource.filter((id) => base.includes(id))]
        const result = preserveSelectionAcrossReorder(withDuplicates, base)
        expect(new Set(result).size).toBe(result.length)
        expect(asSet(result)).toEqual(asSet(withDuplicates))
      }),
    )
  })

  it('orders selected runs present in the list to follow the list order', () => {
    fc.assert(
      fc.property(selection, (selected) => {
        fc.pre(selected.length >= 2)
        // Reordered list is the full selection reversed: every selected run is
        // present, so the result must exactly match the reversed order.
        const reversed = [...selected].reverse()
        const result = preserveSelectionAcrossReorder(selected, reversed)
        expect(result).toEqual(reversed)
      }),
    )
  })

  it('retains selected runs that are filtered out of the list', () => {
    fc.assert(
      fc.property(selection, (selected) => {
        // Empty list models "everything filtered out": selection is still kept.
        const result = preserveSelectionAcrossReorder(selected, [])
        expect(asSet(result)).toEqual(asSet(selected))
      }),
    )
  })
})
