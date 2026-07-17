import { describe, expect, it } from 'vitest'
import fc from 'fast-check'

import { computeFirstInvalid } from './validation'

// Feature: frontend-refactor-2026-07, Property 22: First-invalid focus order
//
// For any field DOM order and any subset of invalid field ids,
// computeFirstInvalid must return the id of the earliest invalid field in DOM
// order, and null when the invalid subset is empty. Invalid ids that are not
// part of the DOM order are ignored, so the result always reflects DOM
// ordering regardless of how the invalid set was built.
//
// Validates: Requirements 10.4
describe('computeFirstInvalid (Property 22: first-invalid focus order)', () => {
  // A DOM order of unique field ids. `fc.uniqueArray` guarantees no duplicates
  // so the "first in DOM order" notion is unambiguous.
  const uniqueOrder = fc.uniqueArray(fc.string({ minLength: 1, maxLength: 8 }), {
    minLength: 1,
    maxLength: 20,
  })

  // Ids that are guaranteed NOT to appear in `order`, used to prove that
  // out-of-order invalid ids are ignored. The `#` prefix cannot collide with
  // generated order ids (which are plain strings without that marker below).
  const foreignId = fc.string({ maxLength: 8 }).map((s) => `#foreign:${s}`)

  it('returns the earliest invalid field in DOM order', () => {
    fc.assert(
      fc.property(
        uniqueOrder.chain((order) =>
          // Pick an arbitrary subset of `order` as the invalid ids.
          fc.subarray(order).map((invalidSubset) => ({ order, invalidSubset })),
        ),
        ({ order, invalidSubset }) => {
          const result = computeFirstInvalid(order, invalidSubset)
          const invalid = new Set(invalidSubset)
          const expected = order.find((id) => invalid.has(id)) ?? null
          expect(result).toBe(expected)
        },
      ),
    )
  })

  it('returns null when the invalid subset is empty', () => {
    fc.assert(
      fc.property(uniqueOrder, (order) => {
        expect(computeFirstInvalid(order, [])).toBeNull()
        expect(computeFirstInvalid(order, new Set<string>())).toBeNull()
      }),
    )
  })

  it('ignores invalid ids that are not part of the DOM order', () => {
    fc.assert(
      fc.property(
        uniqueOrder.chain((order) =>
          fc
            .subarray(order)
            .chain((invalidSubset) =>
              // Add some foreign ids that never appear in `order`.
              fc
                .array(foreignId, { maxLength: 5 })
                .map((foreign) => ({ order, invalidSubset, foreign })),
            ),
        ),
        ({ order, invalidSubset, foreign }) => {
          // The foreign ids must be ignored, so adding them to the invalid set
          // does not change the result relative to the in-order subset alone.
          const withForeign = computeFirstInvalid(order, [...foreign, ...invalidSubset])
          const withoutForeign = computeFirstInvalid(order, invalidSubset)
          expect(withForeign).toBe(withoutForeign)
        },
      ),
    )
  })

  it('returns null when every invalid id is foreign to the DOM order', () => {
    fc.assert(
      fc.property(
        uniqueOrder,
        fc.array(foreignId, { minLength: 1, maxLength: 5 }),
        (order, foreign) => {
          expect(computeFirstInvalid(order, foreign)).toBeNull()
        },
      ),
    )
  })

  it('accepts a Set and an array equivalently', () => {
    fc.assert(
      fc.property(
        uniqueOrder.chain((order) =>
          fc.subarray(order).map((invalidSubset) => ({ order, invalidSubset })),
        ),
        ({ order, invalidSubset }) => {
          const fromArray = computeFirstInvalid(order, invalidSubset)
          const fromSet = computeFirstInvalid(order, new Set(invalidSubset))
          expect(fromSet).toBe(fromArray)
        },
      ),
    )
  })
})
