import { describe, expect, it } from 'vitest'
import fc from 'fast-check'

import { FORM_MESSAGE_KEYS, validateNumeric } from './validation'

// Feature: frontend-refactor-2026-07, Property 24: 数值 min/max 约束
//
// For any finite numeric value and optional min/max bounds, `validateNumeric`
// must reject values outside the inclusive `[min, max]` range and accept values
// inside it: `value < min` or `value > max` returns a non-null FieldError, while
// `min <= value <= max` returns `null`. `step` is intentionally omitted here so
// step-grid alignment cannot interfere with the range contract.
//
// Validates: Requirements 10.9
describe('validateNumeric (Property 24: numeric min/max constraint)', () => {
  // Finite doubles only; exclude NaN/Infinity which have their own contract.
  const finite = fc.double({ noNaN: true, noDefaultInfinity: true })

  it('returns null when the value is within the inclusive [min, max] range', () => {
    fc.assert(
      // `t` in [0, 1] interpolates a value that lies inside [min, max].
      fc.property(finite, finite, fc.double({ min: 0, max: 1, noNaN: true }), (a, b, t) => {
        const min = Math.min(a, b)
        const max = Math.max(a, b)
        const value = min + (max - min) * t
        // Guard against float drift producing a value just outside the range.
        if (value < min || value > max) {
          return
        }
        expect(validateNumeric(value, min, max)).toBeNull()
      }),
    )
  })

  it('returns a belowMin error when value < min', () => {
    fc.assert(
      // `delta` is strictly positive so `value` is strictly below `min`.
      fc.property(finite, fc.double({ min: 1e-6, max: 1e9, noNaN: true }), (min, delta) => {
        const value = min - delta
        if (!(value < min)) {
          return // skip degenerate cases lost to float precision
        }
        const error = validateNumeric(value, min)
        expect(error).not.toBeNull()
        expect(error?.messageKey).toBe(FORM_MESSAGE_KEYS.numericBelowMin)
      }),
    )
  })

  it('returns an aboveMax error when value > max', () => {
    fc.assert(
      // `delta` is strictly positive so `value` is strictly above `max`.
      fc.property(finite, fc.double({ min: 1e-6, max: 1e9, noNaN: true }), (max, delta) => {
        const value = max + delta
        if (!(value > max)) {
          return // skip degenerate cases lost to float precision
        }
        const error = validateNumeric(value, undefined, max)
        expect(error).not.toBeNull()
        expect(error?.messageKey).toBe(FORM_MESSAGE_KEYS.numericAboveMax)
      }),
    )
  })

  it('honors both bounds together across the full input space', () => {
    fc.assert(
      fc.property(finite, finite, finite, (v, a, b) => {
        const min = Math.min(a, b)
        const max = Math.max(a, b)
        const error = validateNumeric(v, min, max)
        if (v < min) {
          expect(error?.messageKey).toBe(FORM_MESSAGE_KEYS.numericBelowMin)
        } else if (v > max) {
          expect(error?.messageKey).toBe(FORM_MESSAGE_KEYS.numericAboveMax)
        } else {
          expect(error).toBeNull()
        }
      }),
    )
  })
})
