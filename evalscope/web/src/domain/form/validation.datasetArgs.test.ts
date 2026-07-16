import { describe, expect, it } from 'vitest'
import fc from 'fast-check'

import { FORM_MESSAGE_KEYS, validateDatasetArgs } from './validation'

// Feature: frontend-refactor-2026-07, Property 23: Dataset_Args JSON 校验与 round-trip
//
// For any JSON value serialized via JSON.stringify and handed to
// validateDatasetArgs:
// - A JSON object (the expected structure) yields { ok: true } whose `value` is
//   semantically equivalent to the original object (round-trip preserved).
// - A valid JSON value that is NOT an object (array, null or a primitive) yields
//   { ok: false, messageKey: datasetArgsInvalidStructure }.
// - Arbitrary non-JSON text yields { ok: false, messageKey: datasetArgsInvalidJson }.
// Neither failure path mutates the input text (the function is pure).
//
// Validates: Requirements 10.6, 10.7
describe('validateDatasetArgs (Property 23: Dataset_Args JSON validation & round-trip)', () => {
  // A JSON object generator: string keys mapped to arbitrary JSON values.
  // `__proto__` is excluded because it cannot be represented as a plain own
  // property through the round-trip and would produce a spurious mismatch.
  const jsonObject = fc.dictionary(
    fc.string().filter((key) => key !== '__proto__'),
    fc.jsonValue(),
  )

  // Valid JSON values that are NOT objects: null, primitives and arrays. These
  // parse successfully but violate the expected object structure.
  const nonObjectJson = fc
    .jsonValue()
    .filter((value) => value === null || Array.isArray(value) || typeof value !== 'object')

  // Arbitrary text that is NOT parseable as JSON.
  const nonJsonText = fc.string().filter((text) => {
    try {
      JSON.parse(text)
      return false
    } catch {
      return true
    }
  })

  it('accepts any JSON object and round-trips its value', () => {
    fc.assert(
      fc.property(jsonObject, (obj) => {
        const text = JSON.stringify(obj)
        const result = validateDatasetArgs(text)
        expect(result.ok).toBe(true)
        // Round-trip: the parsed value is semantically equivalent to the input.
        // `toEqual` treats +0/-0 as equal, matching JSON semantics.
        if (result.ok) {
          expect(result.value).toEqual(obj)
        }
      }),
    )
  })

  it('rejects valid JSON that is not an object without mutating the input', () => {
    fc.assert(
      fc.property(nonObjectJson, (value) => {
        const text = JSON.stringify(value)
        const before = `${text}`
        const result = validateDatasetArgs(text)
        expect(result.ok).toBe(false)
        if (!result.ok) {
          expect(result.messageKey).toBe(FORM_MESSAGE_KEYS.datasetArgsInvalidStructure)
        }
        // The failure path leaves the input text unchanged.
        expect(text).toBe(before)
      }),
    )
  })

  it('rejects non-JSON text without mutating the input', () => {
    fc.assert(
      fc.property(nonJsonText, (text) => {
        const before = `${text}`
        const result = validateDatasetArgs(text)
        expect(result.ok).toBe(false)
        if (!result.ok) {
          expect(result.messageKey).toBe(FORM_MESSAGE_KEYS.datasetArgsInvalidJson)
        }
        // The failure path leaves the input text unchanged.
        expect(text).toBe(before)
      }),
    )
  })
})
