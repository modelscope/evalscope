/**
 * Compare data model (no rendering).
 *
 * This module holds the pure data logic behind the Evaluation History /
 * Compare surfaces: how a run is labelled, how compare selection is bounded and
 * de-duplicated, and whether a set of runs can be meaningfully compared. It has
 * no React/DOM/network/clock dependencies so it can be exercised directly by
 * unit and property tests.
 */

import type { ReportData } from '@/api/types'
import { DATASET_TOKEN, parseReportName } from '@/utils/reportParser'

/** Separator placed between the model and dataset parts of a display label. */
const LABEL_SEPARATOR = ' · '

/**
 * A meaningful display label for a run, derived from its model and dataset(s)
 * rather than the raw timestamped path.
 */
export interface RunDisplayLabel {
  /** Parsed model name; empty string when no meaningful model could be parsed. */
  model: string
  /** Parsed dataset name(s), joined when a run spans multiple datasets. */
  dataset: string
  /**
   * Human-friendly label composed of model and dataset. When a meaningful model
   * is present the label always contains that model and is never equal to the
   * raw timestamp prefix or the full run path.
   */
  label: string
}

/**
 * Build a meaningful display label from a run identifier.
 *
 * Run identifiers follow the `{timestamp}@@{model}::{dataset1}, {dataset2}`
 * encoding (see `parseReportName`). When a model can be parsed, the label is
 * `"{model} · {dataset}"` (or just the model when no dataset is present). When
 * no model can be parsed, the label falls back to the raw run name so the caller
 * still has something to render.
 *
 * @param runName Raw run identifier string.
 * @returns The parsed model, dataset and composed label.
 */
export function buildDisplayLabel(runName: string): RunDisplayLabel {
  const { model, datasets } = parseReportName(runName)
  const dataset = datasets.join(DATASET_TOKEN)
  const trimmedModel = model.trim()

  if (trimmedModel.length === 0) {
    // No meaningful model could be parsed: fall back to the raw run name.
    return { model: '', dataset, label: runName }
  }

  const label = dataset.length > 0 ? `${trimmedModel}${LABEL_SEPARATOR}${dataset}` : trimmedModel
  return { model: trimmedModel, dataset, label }
}

/**
 * Number of runs the evaluation compare view can render side by side.
 *
 * DESIGN.md §Compare Slots defines exactly three brand-color slots, so the
 * `MODEL_PALETTE` in `ComparePage.tsx` must have this many entries. Selection
 * itself is unbounded: extra runs are simply not carried into the comparison,
 * which lets the list surfaces select (and delete) any number of runs.
 */
export const MAX_COMPARE_SLOTS = 3

/**
 * Add a run to the compare selection, de-duping.
 *
 * The selection is a set with a stable insertion order and is not capped: the
 * same selection also drives batch deletion, so bounding it here would limit
 * unrelated actions. Surfaces that can only render a fixed number of runs clamp
 * at navigation time instead (see `MAX_COMPARE_SLOTS`).
 *
 * @param state Current selection of run ids.
 * @param runId Run id to add.
 * @returns The next selection, unchanged when the run is already selected.
 */
export function addToSelection(state: string[], runId: string): string[] {
  if (state.includes(runId)) {
    // Already selected: de-duplicate by leaving the selection unchanged.
    return state
  }
  return [...state, runId]
}

/**
 * Preserve the full compare selection across a list reorder (sort/filter).
 *
 * The compare selection is a set of run ids that is conceptually independent of
 * the order in which runs happen to be listed. Sorting or filtering the list
 * only changes how runs are arranged (and which are currently visible); it must
 * never drop any selected run. This helper realises that contract:
 *
 *   - The returned selection is the *same set* as the input selection: no run is
 *     added or removed, and duplicates in the input are collapsed. As a set the
 *     output always equals the input (set identity across any reorder/filter).
 *   - Selected runs that appear in the reordered/filtered list are ordered to
 *     match that list, so the on-screen order of selected runs follows the sort.
 *   - Selected runs that are not present in the reordered list (e.g. hidden by a
 *     filter) are retained and appended, so filtering never loses a selection.
 *
 * @param selected Current selection of run ids (may contain duplicates).
 * @param reorderedList Run ids in their new (sorted/filtered) order.
 * @returns The preserved selection, reordered to follow the list but with an
 *   unchanged membership set.
 */
export function preserveSelectionAcrossReorder(selected: string[], reorderedList: string[]): string[] {
  // De-duplicate the incoming selection while keeping first-seen order as the
  // stable fallback ordering for runs missing from the reordered list.
  const uniqueSelected = [...new Set(selected)]
  const selectedSet = new Set(uniqueSelected)

  // Selected runs that are visible in the reordered list, in the list's order
  // (de-duplicated in case the list itself repeats an id).
  const seen = new Set<string>()
  const ordered: string[] = []
  for (const id of reorderedList) {
    if (selectedSet.has(id) && !seen.has(id)) {
      seen.add(id)
      ordered.push(id)
    }
  }

  // Selected runs absent from the reordered list (e.g. filtered out) are kept so
  // the selection set is never reduced by a sort/filter.
  const retained = uniqueSelected.filter((id) => !seen.has(id))

  return [...ordered, ...retained]
}

/** Localized message key returned when runs share no common dataset. */
const NO_COMMON_DATASET_KEY = 'compare.noCommon'

/**
 * Collect the set of dataset names a run covers.
 *
 * A run may carry its dataset either in the structured `dataset_name` field or
 * encoded in its `name` (for runs spanning multiple datasets), so both sources
 * are unioned.
 */
function runDatasets(run: ReportData): Set<string> {
  const datasets = new Set<string>()
  if (run.dataset_name) {
    datasets.add(run.dataset_name)
  }
  for (const ds of parseReportName(run.name).datasets) {
    const trimmed = ds.trim()
    if (trimmed.length > 0) {
      datasets.add(trimmed)
    }
  }
  return datasets
}

/**
 * Determine whether a set of runs can be meaningfully compared.
 *
 * Runs are compatible when they share at least one common dataset. When there is
 * no common dataset the runs cannot be aligned for comparison, so a localized
 * reason key is returned; the caller keeps the existing selection and surfaces
 * the reason. Fewer than two runs is not yet an incompatibility, so
 * `null` is returned.
 *
 * @param runs Runs selected for comparison.
 * @returns A localized reason key when incompatible, or `null` when compatible.
 */
export function compatibilityReason(runs: ReportData[]): string | null {
  if (runs.length < 2) {
    return null
  }

  const datasetSets = runs.map(runDatasets)
  const common = datasetSets.reduce((acc, set) => {
    return new Set([...acc].filter((ds) => set.has(ds)))
  })

  return common.size === 0 ? NO_COMMON_DATASET_KEY : null
}
