// Feature: frontend-refactor-2026-07, Property 26: Schema 校验 round-trip 与 typed error
//
// For any object that conforms to a domain schema, `schema.parse` (via
// `safeParse`) must succeed and return data that is semantically equivalent to
// the input (round-trip). For any object that breaks a required field or a
// field type, validation must fail rather than returning unvalidated data, and
// the API client's validated path must surface the failure as a typed
// `DomainError` with `kind='validation'` (never an uncaught exception, never
// the unvalidated body).
//
// Two representative domain schemas are exercised: `reportSummarySchema`
// (reports domain) and `perfRunSummarySchema` (performance domain). Both mix
// string, number, boolean and optional fields, so the property covers the
// common shapes of the API contract.
//
// Validates: Requirements 13.1, 13.2

import { describe, expect, it, vi } from 'vitest'
import fc from 'fast-check'
import type { ZodType } from 'zod'

import { apiValidated } from '../client'
import { DomainError, isDomainError } from '../errors'
import { perfRunSummarySchema, reportSummarySchema } from './index'

// ------------------------------------------------------------------ //
// Field metadata used to generate valid objects and to corrupt them   //
// ------------------------------------------------------------------ //

/** Primitive kind of a required schema field (drives wrong-type corruption). */
type FieldType = 'string' | 'number' | 'boolean'

interface FieldSpec {
  name: string
  type: FieldType
}

/**
 * Produce a value whose type is guaranteed wrong for `type`, so that a schema
 * expecting `type` must reject it.
 */
function wrongTypedValue(type: FieldType): unknown {
  switch (type) {
    case 'string':
      // A number is not a string.
      return 42
    case 'number':
      // A string is not a number.
      return 'not-a-number'
    case 'boolean':
      // A string is not a boolean.
      return 'not-a-boolean'
  }
}

// ------------------------------------------------------------------ //
// reportSummarySchema — required fields + one optional record field   //
// ------------------------------------------------------------------ //

const REPORT_SUMMARY_FIELDS: FieldSpec[] = [
  { name: 'name', type: 'string' },
  { name: 'model_name', type: 'string' },
  { name: 'dataset_name', type: 'string' },
  { name: 'score', type: 'number' },
  { name: 'num_samples', type: 'number' },
  { name: 'timestamp', type: 'string' },
]

/** Generate a fully-valid `reportSummarySchema` object (optional key varies). */
const reportSummaryArb: fc.Arbitrary<Record<string, unknown>> = fc.record(
  {
    name: fc.string(),
    model_name: fc.string(),
    dataset_name: fc.string(),
    score: fc.double({ noNaN: true, noDefaultInfinity: true }),
    num_samples: fc.nat(),
    timestamp: fc.string(),
    // Optional field: sometimes present, sometimes absent (requiredKeys below).
    dataset_scores: fc.dictionary(
      fc.string(),
      fc.double({ noNaN: true, noDefaultInfinity: true }),
      { maxKeys: 4 },
    ),
  },
  { requiredKeys: ['name', 'model_name', 'dataset_name', 'score', 'num_samples', 'timestamp'] },
)

// ------------------------------------------------------------------ //
// perfRunSummarySchema — strings, numbers, booleans, all required     //
// ------------------------------------------------------------------ //

const PERF_RUN_SUMMARY_FIELDS: FieldSpec[] = [
  { name: 'path', type: 'string' },
  { name: 'model', type: 'string' },
  { name: 'api_type', type: 'string' },
  { name: 'dataset', type: 'string' },
  { name: 'num_runs', type: 'number' },
  { name: 'total_requests', type: 'number' },
  { name: 'success_rate', type: 'number' },
  { name: 'best_rps', type: 'number' },
  { name: 'best_latency', type: 'number' },
  { name: 'is_embedding', type: 'boolean' },
  { name: 'has_html', type: 'boolean' },
  { name: 'timestamp', type: 'string' },
]

/** Generate a fully-valid `perfRunSummarySchema` object. */
const perfRunSummaryArb: fc.Arbitrary<Record<string, unknown>> = fc.record({
  path: fc.string(),
  model: fc.string(),
  api_type: fc.string(),
  dataset: fc.string(),
  num_runs: fc.nat(),
  total_requests: fc.nat(),
  success_rate: fc.double({ min: 0, max: 1, noNaN: true, noDefaultInfinity: true }),
  best_rps: fc.double({ noNaN: true, noDefaultInfinity: true }),
  best_latency: fc.double({ noNaN: true, noDefaultInfinity: true }),
  is_embedding: fc.boolean(),
  has_html: fc.boolean(),
  timestamp: fc.string(),
})

// ------------------------------------------------------------------ //
// Corruption generator: break exactly one required field              //
// ------------------------------------------------------------------ //

type CorruptionMode = 'delete' | 'null' | 'wrongType'

/**
 * Given a valid-object arbitrary and its required-field specs, produce an
 * arbitrary of objects each broken in exactly one required field. `null` and
 * `delete` break the required-field constraint; `wrongType` breaks the field
 * type. None of these schemas mark the targeted fields as nullable/optional, so
 * every mode is guaranteed to make the object invalid.
 */
function corruptedArb(
  validArb: fc.Arbitrary<Record<string, unknown>>,
  fields: FieldSpec[],
): fc.Arbitrary<Record<string, unknown>> {
  return validArb.chain((valid) =>
    fc
      .record({
        field: fc.constantFrom(...fields),
        mode: fc.constantFrom<CorruptionMode>('delete', 'null', 'wrongType'),
      })
      .map(({ field, mode }) => {
        const copy: Record<string, unknown> = { ...valid }
        if (mode === 'delete') {
          delete copy[field.name]
        } else if (mode === 'null') {
          copy[field.name] = null
        } else {
          copy[field.name] = wrongTypedValue(field.type)
        }
        return copy
      }),
  )
}

// ------------------------------------------------------------------ //
// Fake HTTP response for the API client path                          //
// ------------------------------------------------------------------ //

/** Build a minimal OK Response stub exposing only what the client reads. */
function okResponse(body: unknown): Response {
  return {
    ok: true,
    status: 200,
    statusText: 'OK',
    json: async () => body,
  } as unknown as Response
}

/** Stub `fetch` to resolve with an OK response carrying `body`. */
function stubFetchWith(body: unknown): void {
  vi.stubGlobal('fetch', vi.fn().mockResolvedValue(okResponse(body)))
}

// ------------------------------------------------------------------ //
// Tests                                                               //
// ------------------------------------------------------------------ //

const CASES: Array<{ name: string; schema: ZodType; validArb: fc.Arbitrary<Record<string, unknown>>; fields: FieldSpec[] }> = [
  { name: 'reportSummarySchema', schema: reportSummarySchema, validArb: reportSummaryArb, fields: REPORT_SUMMARY_FIELDS },
  { name: 'perfRunSummarySchema', schema: perfRunSummarySchema, validArb: perfRunSummaryArb, fields: PERF_RUN_SUMMARY_FIELDS },
]

describe('domain schema validation (Property 26: Schema 校验 round-trip 与 typed error)', () => {
  for (const { name, schema, validArb, fields } of CASES) {
    describe(name, () => {
      it('parses any conforming object and round-trips it to semantically equivalent data', () => {
        fc.assert(
          fc.property(validArb, (valid) => {
            const result = schema.safeParse(valid)
            expect(result.success).toBe(true)
            if (result.success) {
              // Round-trip: validated data is semantically equivalent to input.
              expect(result.data).toEqual(valid)
            }
          }),
        )
      })

      it('rejects any object with a broken required field or field type', () => {
        fc.assert(
          fc.property(corruptedArb(validArb, fields), (corrupted) => {
            const result = schema.safeParse(corrupted)
            // Validation must fail rather than pass the unvalidated object.
            expect(result.success).toBe(false)
          }),
        )
      })

      it('surfaces validation failures through apiValidated as DomainError(kind=validation)', async () => {
        await fc.assert(
          fc.asyncProperty(corruptedArb(validArb, fields), async (corrupted) => {
            stubFetchWith(corrupted)
            // apiValidated must reject with a typed DomainError, never resolve
            // with the unvalidated data and never throw synchronously.
            const promise = apiValidated('/api/v1/test', schema)
            await expect(promise).rejects.toBeInstanceOf(DomainError)
            await promise.catch((err: unknown) => {
              expect(isDomainError(err)).toBe(true)
              expect((err as DomainError).kind).toBe('validation')
            })
          }),
        )
      })

      it('resolves apiValidated with round-tripped data for a conforming response', async () => {
        await fc.assert(
          fc.asyncProperty(validArb, async (valid) => {
            stubFetchWith(valid)
            const data = await apiValidated('/api/v1/test', schema)
            expect(data).toEqual(valid)
          }),
        )
      })
    })
  }
})
