// Feature: frontend-refactor-2026-07, Property 8: 运行 display label 由 model 与 dataset 组成
//
// For any run identifier string, `buildDisplayLabel` must produce a `label`
// that is composed of the parsed model and dataset and always contains the
// parsed model. When a meaningful model can be parsed, the label must never
// collapse back to the raw timestamp prefix or the full run path.
//
// Validates: Requirements 5.6, 5.7, 8.9

import { describe, expect, it } from 'vitest'
import fc from 'fast-check'

import { DATASET_TOKEN, MODEL_TOKEN, REPORT_TOKEN, parseReportName } from '@/utils/reportParser'
import { buildDisplayLabel } from './compareModel'

/** Separator the production code places between the model and dataset parts. */
const LABEL_SEPARATOR = ' · '

/**
 * Timestamp prefix generator: digits + underscore only (e.g. `20260701_120000`).
 * Because it is purely numeric it can never coincide with a model name (models
 * start with a letter), which mirrors real run identifiers.
 */
const prefixArb = fc.stringMatching(/^\d{8}_\d{6}$/)

/**
 * Model name generator: starts with a letter, contains no encoding tokens
 * (`@@`, `::`, `, `) and no surrounding whitespace, so after trimming it is a
 * genuinely meaningful model name (e.g. `Qwen/Qwen2.5-0.5B-Instruct`).
 */
const modelArb = fc.stringMatching(/^[A-Za-z][A-Za-z0-9._/-]{0,30}$/)

/** Dataset name generator: no encoding tokens and no whitespace. */
const datasetArb = fc.stringMatching(/^[A-Za-z0-9._-]{1,20}$/)

/** Compose a well-formed run identifier from its parts. */
function makeRunName(prefix: string, model: string, datasets: string[]): string {
  return `${prefix}${REPORT_TOKEN}${model}${MODEL_TOKEN}${datasets.join(DATASET_TOKEN)}`
}

describe('buildDisplayLabel (Property 8: 运行 display label 由 model 与 dataset 组成)', () => {
  it('composes the label from model and dataset for well-formed run names', () => {
    const caseArb = fc.record({
      prefix: prefixArb,
      model: modelArb,
      datasets: fc.array(datasetArb, { minLength: 1, maxLength: 3 }),
    })

    fc.assert(
      fc.property(caseArb, ({ prefix, model, datasets }) => {
        const runName = makeRunName(prefix, model, datasets)
        const result = buildDisplayLabel(runName)

        const expectedDataset = datasets.join(DATASET_TOKEN)
        // Model and dataset are parsed back out faithfully.
        expect(result.model).toBe(model)
        expect(result.dataset).toBe(expectedDataset)

        // The label is composed of both the model and the dataset.
        expect(result.label).toBe(`${model}${LABEL_SEPARATOR}${expectedDataset}`)
        expect(result.label).toContain(result.model)
        expect(result.label).toContain(result.dataset)

        // A meaningful model never collapses to the raw prefix or full path.
        expect(result.label).not.toBe(prefix)
        expect(result.label).not.toBe(runName)
      }),
    )
  })

  it('uses the model alone as the label when a run has no dataset', () => {
    const caseArb = fc.record({ prefix: prefixArb, model: modelArb })

    fc.assert(
      fc.property(caseArb, ({ prefix, model }) => {
        const runName = makeRunName(prefix, model, [])
        const result = buildDisplayLabel(runName)

        expect(result.model).toBe(model)
        expect(result.dataset).toBe('')
        expect(result.label).toBe(model)
        expect(result.label).toContain(result.model)

        // Still derived from the model, not the raw prefix or full path.
        expect(result.label).not.toBe(prefix)
        expect(result.label).not.toBe(runName)
      }),
    )
  })

  it('always contains the parsed model and never falls back to the path for any string', () => {
    fc.assert(
      fc.property(fc.string(), (runName) => {
        const result = buildDisplayLabel(runName)
        const { prefix } = parseReportName(runName)

        // The label always contains the parsed model (trivially so when empty).
        expect(result.label).toContain(result.model)

        if (result.model.length > 0) {
          // A meaningful model was parsed: the label is derived from it and can
          // never equal the full run path (which still carries the encoding
          // tokens the label drops).
          expect(result.label).not.toBe(runName)

          // Nor does it collapse to the bare timestamp prefix, except in the
          // degenerate case where the model text is literally the prefix.
          if (result.model !== prefix) {
            expect(result.label).not.toBe(prefix)
          }
        }
      }),
    )
  })
})
