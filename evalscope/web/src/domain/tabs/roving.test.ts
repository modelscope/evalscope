import { describe, expect, it } from 'vitest'
import fc from 'fast-check'

import { moveRovingIndex, nextRovingIndex, type RovingDirection } from './roving'

// Feature: frontend-refactor-2026-07, Property 25: Roving tabindex 键盘回绕
//
// For any current index, tab count and direction, the roving navigation helpers
// must return a valid index within [0, count) and wrap around at the ends:
// moving 'next' past the last tab returns the first, and moving 'prev' before
// the first tab returns the last.
//
// Validates: Requirements 11.5
describe('nextRovingIndex (Property 25: roving tabindex wrap-around)', () => {
  const direction = fc.constantFrom<RovingDirection>('next', 'prev')
  // `current` intentionally includes out-of-range and negative values.
  const current = fc.integer({ min: -1000, max: 1000 })
  // A tablist has at least one tab.
  const count = fc.integer({ min: 1, max: 500 })

  it('always returns a valid index within [0, count)', () => {
    fc.assert(
      fc.property(current, count, direction, (cur, n, dir) => {
        const result = nextRovingIndex(cur, n, dir)
        expect(result).toBeGreaterThanOrEqual(0)
        expect(result).toBeLessThan(n)
        expect(Number.isInteger(result)).toBe(true)
      }),
    )
  })

  it("wraps from the last tab back to the first when moving 'next'", () => {
    fc.assert(
      fc.property(fc.integer({ min: 2, max: 500 }), (n) => {
        expect(nextRovingIndex(n - 1, n, 'next')).toBe(0)
      }),
    )
  })

  it("wraps from the first tab back to the last when moving 'prev'", () => {
    fc.assert(
      fc.property(fc.integer({ min: 2, max: 500 }), (n) => {
        expect(nextRovingIndex(0, n, 'prev')).toBe(n - 1)
      }),
    )
  })

  it("moves forward by one for 'next' when not at the last tab", () => {
    fc.assert(
      fc.property(fc.integer({ min: 2, max: 500 }), (n) => {
        const idx = fc.sample(fc.integer({ min: 0, max: n - 2 }), 1)[0]
        expect(nextRovingIndex(idx, n, 'next')).toBe(idx + 1)
      }),
    )
  })

  it("moves backward by one for 'prev' when not at the first tab", () => {
    fc.assert(
      fc.property(fc.integer({ min: 2, max: 500 }), (n) => {
        const idx = fc.sample(fc.integer({ min: 1, max: n - 1 }), 1)[0]
        expect(nextRovingIndex(idx, n, 'prev')).toBe(idx - 1)
      }),
    )
  })

  it('returns -1 when count <= 0', () => {
    fc.assert(
      fc.property(fc.integer({ max: 0 }), current, direction, (n, cur, dir) => {
        expect(nextRovingIndex(cur, n, dir)).toBe(-1)
      }),
    )
  })

  it('returns 0 when count === 1', () => {
    fc.assert(
      fc.property(current, direction, (cur, dir) => {
        expect(nextRovingIndex(cur, 1, dir)).toBe(0)
      }),
    )
  })
})

describe('moveRovingIndex (Property 25: roving tabindex wrap-around)', () => {
  const current = fc.integer({ min: -1000, max: 1000 })

  it('wraps at both ends for horizontal arrow keys', () => {
    fc.assert(
      fc.property(fc.integer({ min: 2, max: 500 }), (n) => {
        expect(moveRovingIndex(n - 1, n, 'ArrowRight', 'horizontal')).toBe(0)
        expect(moveRovingIndex(0, n, 'ArrowLeft', 'horizontal')).toBe(n - 1)
      }),
    )
  })

  it('wraps at both ends for vertical arrow keys', () => {
    fc.assert(
      fc.property(fc.integer({ min: 2, max: 500 }), (n) => {
        expect(moveRovingIndex(n - 1, n, 'ArrowDown', 'vertical')).toBe(0)
        expect(moveRovingIndex(0, n, 'ArrowUp', 'vertical')).toBe(n - 1)
      }),
    )
  })

  it('returns a valid index within [0, count) for any navigation key', () => {
    // Pair each key with an orientation in which it actually navigates:
    // horizontal uses ArrowLeft/ArrowRight, vertical uses ArrowUp/ArrowDown,
    // and Home/End work in both orientations.
    const navCase = fc.constantFrom<[string, 'horizontal' | 'vertical']>(
      ['ArrowRight', 'horizontal'],
      ['ArrowLeft', 'horizontal'],
      ['ArrowDown', 'vertical'],
      ['ArrowUp', 'vertical'],
      ['Home', 'horizontal'],
      ['End', 'vertical'],
    )
    fc.assert(
      fc.property(current, fc.integer({ min: 1, max: 500 }), navCase, (cur, n, [key, ori]) => {
        const result = moveRovingIndex(cur, n, key, ori)
        // A navigation key paired with a compatible orientation always yields
        // a valid index rather than null.
        expect(result).not.toBeNull()
        expect(result as number).toBeGreaterThanOrEqual(0)
        expect(result as number).toBeLessThan(n)
      }),
    )
  })

  it('returns null for keys that do not affect navigation', () => {
    const nonNavKey = fc.constantFrom('Enter', ' ', 'Tab', 'Escape', 'a', 'PageUp')
    fc.assert(
      fc.property(current, fc.integer({ min: 1, max: 500 }), nonNavKey, (cur, n, key) => {
        expect(moveRovingIndex(cur, n, key)).toBeNull()
      }),
    )
  })
})
