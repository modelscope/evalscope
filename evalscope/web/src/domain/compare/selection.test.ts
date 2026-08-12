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

import { formatReportRef } from '@/domain/report/reportRef'
import {
  addToSelection,
  buildDisplayLabel,
  buildDisplayLabels,
  compatibilityReason,
  MAX_COMPARE_SLOTS,
  preserveSelectionAcrossReorder,
  togglePredictionSelection,
} from './selection'

/* Property 8: run display label composition */

describe('buildDisplayLabel (Property 8: run display label is composed of model and dataset)', () => {
  /** Separator the production code places between the model and dataset parts. */
  const LABEL_SEPARATOR = ' · '

  /** Run-id generator: digits + underscore only (e.g. `20260701_120000`), never a path separator. */
  const runIdArb = fc.stringMatching(/^\d{8}_\d{6}$/)

  /** Model-id generator: a single safe path segment (no `/`), as produced by `safe_filename`. */
  const modelArb = fc.stringMatching(/^[A-Za-z][A-Za-z0-9._-]{0,30}$/)

  /** Dataset name generator. */
  const datasetArb = fc.stringMatching(/^[A-Za-z0-9._-]{1,20}$/)

  it('composes the label from model and dataset for well-formed references', () => {
    const caseArb = fc.record({
      runId: runIdArb,
      model: modelArb,
      datasets: fc.array(datasetArb, { minLength: 1, maxLength: 3 }),
    })

    fc.assert(
      fc.property(caseArb, ({ runId, model, datasets }) => {
        const ref = formatReportRef({ runId, modelId: model })
        const result = buildDisplayLabel(ref, datasets)

        const expectedDataset = datasets.join(', ')
        // Model and dataset appear faithfully.
        expect(result.model).toBe(model)
        expect(result.dataset).toBe(expectedDataset)

        // The label is composed of both the model and the dataset.
        expect(result.label).toBe(`${model}${LABEL_SEPARATOR}${expectedDataset}`)
        expect(result.label).toContain(result.model)
        expect(result.label).toContain(result.dataset)

        // A meaningful model never collapses to the raw reference.
        expect(result.label).not.toBe(ref)
      }),
    )
  })

  it('uses the model alone as the label when a run has no dataset', () => {
    const caseArb = fc.record({ runId: runIdArb, model: modelArb })

    fc.assert(
      fc.property(caseArb, ({ runId, model }) => {
        const ref = formatReportRef({ runId, modelId: model })
        const result = buildDisplayLabel(ref, [])

        expect(result.model).toBe(model)
        expect(result.dataset).toBe('')
        expect(result.label).toBe(model)
        expect(result.label).toContain(result.model)

        // Still derived from the model, not the raw reference.
        expect(result.label).not.toBe(ref)
      }),
    )
  })

  it('always contains the parsed model for any reference', () => {
    fc.assert(
      fc.property(fc.string(), (ref) => {
        const result = buildDisplayLabel(ref, [])
        // The label always contains the parsed model (trivially so when empty).
        expect(result.label).toContain(result.model)
      }),
    )
  })
})

describe('buildDisplayLabels', () => {
  it('adds each run id when the same model is compared twice', () => {
    const first = formatReportRef({ runId: '20260810_100000', modelId: 'qwen-plus' })
    const second = formatReportRef({ runId: '20260810_110000', modelId: 'qwen-plus' })
    const datasetsByRef = { [first]: ['gsm8k'], [second]: ['gsm8k'] }

    expect(buildDisplayLabels([first, second], datasetsByRef)).toEqual({
      [first]: 'qwen-plus (20260810_100000) · gsm8k',
      [second]: 'qwen-plus (20260810_110000) · gsm8k',
    })
  })

  it('keeps the compact model and dataset label when model names are unique', () => {
    const run = formatReportRef({ runId: '20260810_100000', modelId: 'qwen-plus' })

    expect(buildDisplayLabels([run], { [run]: ['gsm8k'] })).toEqual({ [run]: 'qwen-plus · gsm8k' })
  })
})

describe('compatibilityReason', () => {
  it('returns null for fewer than two runs', () => {
    expect(compatibilityReason([])).toBeNull()
    expect(compatibilityReason([['gsm8k']])).toBeNull()
  })

  it('returns null when runs share a common dataset', () => {
    expect(compatibilityReason([['gsm8k', 'mmlu'], ['gsm8k']])).toBeNull()
  })

  it('flags runs that share no common dataset', () => {
    expect(compatibilityReason([['gsm8k'], ['mmlu']])).toBe('compare.noCommon')
  })
})

/* Property 9: selection preserved across reorder/filter */

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

/* Property 10: comparison selection */

describe('addToSelection (Property 10: compare selection is an unbounded set)', () => {
  // A set of unique run ids of size 0..8, so we exercise small and large
  // selections alike: selection is deliberately not capped.
  const uniqueSelection = fc.uniqueArray(fc.string(), { minLength: 0, maxLength: 8 })
  // A run id to add; drawn from the same string space so it may or may not
  // already be present in the selection.
  const runId = fc.string()

  it('MAX_COMPARE_SLOTS is 3', () => {
    expect(MAX_COMPARE_SLOTS).toBe(3)
  })

  it('de-duplicates when the run is already selected', () => {
    fc.assert(
      fc.property(uniqueSelection, (state) => {
        fc.pre(state.length > 0)
        // Pick an id that is already present in the selection.
        const existing = state[state.length - 1]
        const next = addToSelection(state, existing)
        // Already selected: selection is returned unchanged.
        expect(next).toBe(state)
        expect(next).toEqual(state)
      }),
    )
  })

  it('appends and grows by one when the run is new, regardless of size', () => {
    fc.assert(
      fc.property(uniqueSelection, runId, (state, id) => {
        fc.pre(!state.includes(id))
        const next = addToSelection(state, id)
        expect(next.length).toBe(state.length + 1)
        expect(next).toContain(id)
        // Result stays de-duplicated.
        expect(new Set(next).size).toBe(next.length)
      }),
    )
  })

  it('always yields a de-duplicated selection that keeps every prior run', () => {
    fc.assert(
      fc.property(uniqueSelection, runId, (state, id) => {
        const next = addToSelection(state, id)
        expect(new Set(next).size).toBe(next.length)
        // Adding never drops an already selected run.
        expect(next).toEqual(expect.arrayContaining(state))
      }),
    )
  })
})

describe('togglePredictionSelection', () => {
  it('adds and removes reports without affecting score selection', () => {
    expect(togglePredictionSelection(['a', 'b'], 'c')).toEqual(['a', 'b', 'c'])
    expect(togglePredictionSelection(['a', 'b', 'c'], 'b')).toEqual(['a', 'c'])
  })

  it('keeps the existing selection when three reports are already selected', () => {
    const selected = ['a', 'b', 'c']
    expect(togglePredictionSelection(selected, 'd')).toBe(selected)
  })
})
