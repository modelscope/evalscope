// Feature: frontend-refactor-2026-07, compareModel (report compare) property tests.
//
// Merged property coverage for the report comparison model:
//   - Property 8: run display label is composed of model and dataset;
//   - Property 9: sorting/filtering preserves the selected set;
//   - Property 10: comparison selection cap.
//
// Validates: Requirements 5.6, 5.7, 5.8, 5.9, 8.9

import { describe, expect, it } from 'vitest'
import fc from 'fast-check'

import { DATASET_TOKEN, MODEL_TOKEN, REPORT_TOKEN, parseReportName } from '@/utils/reportParser'
import {
  addToSelection,
  buildDisplayLabel,
  MAX_COMPARE_SELECTION,
  preserveSelectionAcrossReorder,
} from './compareModel'

/* ─── Property 8: run display label composition ───────────────── */

describe('buildDisplayLabel (Property 8: run display label is composed of model and dataset)', () => {
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

/* ─── Property 9: selection preserved across reorder/filter ───── */

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

/* ─── Property 10: comparison selection cap ───────────────────── */

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
