// Feature: frontend-refactor-2026-07, Property 21: symmetric config diff computation
//
// For any two runs' config key/value maps (best_config), buildCompareModel's
// configDiff must list exactly the keys that either take a different value on
// the two sides, or exist on only one side — and must never list a key that is
// present on both sides with an identical value. This is the symmetric
// difference over the two configs, keyed on (presence, value) and independent
// of the runs' timestamps, paths or metric rows.
//
// Validates: Requirements 9.13

import { describe, expect, it } from 'vitest'
import fc from 'fast-check'

import type { PerfDetailResponse } from '../../api/types'
import { buildCompareModel } from './compareModel'

/** How a single config key relates across the two runs. */
type KeyKind = 'same' | 'diff' | 'baseline-only' | 'candidate-only'

/** A generated config key, its relation kind and a base value. */
interface KeySpec {
  key: string
  kind: KeyKind
  value: string
}

/**
 * Generate a set of config keys with distinct names, each tagged with a
 * relation kind. Keys are made unique by name so the two derived configs stay
 * well-defined (no key collisions within a single side).
 */
const keySpecsArb: fc.Arbitrary<KeySpec[]> = fc.uniqueArray(
  fc.record({
    key: fc.stringMatching(/^[A-Za-z0-9 _-]{1,12}$/),
    kind: fc.constantFrom<KeyKind>('same', 'diff', 'baseline-only', 'candidate-only'),
    value: fc.stringMatching(/^[A-Za-z0-9.]{0,8}$/),
  }),
  { selector: (spec) => spec.key, maxLength: 12 },
)

/**
 * Build the baseline and candidate config maps plus the independently-computed
 * expected diff key set from a list of key specs.
 *
 * - `same` — both sides carry `value` (excluded from the diff);
 * - `diff` — baseline carries `value`, candidate carries a guaranteed-distinct
 *   `value + '~'` (both sides present, values differ → included);
 * - `baseline-only` / `candidate-only` — present on a single side (included).
 */
function buildConfigs(specs: KeySpec[]): {
  baselineConfig: Record<string, string>
  candidateConfig: Record<string, string>
  expectedDiffKeys: Set<string>
} {
  const baselineConfig: Record<string, string> = {}
  const candidateConfig: Record<string, string> = {}
  const expectedDiffKeys = new Set<string>()

  for (const { key, kind, value } of specs) {
    switch (kind) {
      case 'same':
        baselineConfig[key] = value
        candidateConfig[key] = value
        break
      case 'diff':
        baselineConfig[key] = value
        // `value + '~'` is always distinct from `value`.
        candidateConfig[key] = `${value}~`
        expectedDiffKeys.add(key)
        break
      case 'baseline-only':
        baselineConfig[key] = value
        expectedDiffKeys.add(key)
        break
      case 'candidate-only':
        candidateConfig[key] = value
        expectedDiffKeys.add(key)
        break
    }
  }

  return { baselineConfig, candidateConfig, expectedDiffKeys }
}

/** Build a minimal PerfDetailResponse carrying a config map and identity. */
function makeRun(path: string, bestConfig: Record<string, string>, generatedAt: string): PerfDetailResponse {
  return {
    path,
    model: 'model-a',
    api_type: 'openai_api',
    dataset: 'openqa',
    generated_at: generatedAt,
    basic_info: { 'Total requests': '100' },
    summary_columns: ['Metric', 'Value'],
    summary_rows: [['Number of requests', 100]],
    best_config: bestConfig,
    recommendations: [],
    num_runs: 1,
    is_embedding: false,
    has_html: true,
  }
}

describe('buildCompareModel — symmetric config diff (Property 21: symmetric config diff computation)', () => {
  it('lists exactly the keys that differ or exist on one side, never identical keys', () => {
    fc.assert(
      fc.property(keySpecsArb, (specs) => {
        const { baselineConfig, candidateConfig, expectedDiffKeys } = buildConfigs(specs)

        // Fix the baseline explicitly so the config diff is computed over the
        // known (baseline, candidate) pair regardless of timestamps.
        const baseline = makeRun('perf/baseline', baselineConfig, '2020-01-01T00:00:00.000Z')
        const candidate = makeRun('perf/candidate', candidateConfig, '2021-01-01T00:00:00.000Z')

        const model = buildCompareModel([baseline, candidate], baseline.path)

        const diffKeys = new Set(model.configDiff.map((entry) => entry.key))

        // The emitted diff key set is exactly the symmetric difference.
        expect(diffKeys).toEqual(expectedDiffKeys)

        // No identical-on-both-sides key ever appears in the diff.
        for (const entry of model.configDiff) {
          const inBaseline = Object.prototype.hasOwnProperty.call(baselineConfig, entry.key)
          const inCandidate = Object.prototype.hasOwnProperty.call(candidateConfig, entry.key)
          const identical = inBaseline && inCandidate && baselineConfig[entry.key] === candidateConfig[entry.key]
          expect(identical).toBe(false)

          // Each entry reflects the true per-side values ('' when absent).
          expect(entry.baseline).toBe(inBaseline ? baselineConfig[entry.key] : '')
          expect(entry.candidate).toBe(inCandidate ? candidateConfig[entry.key] : '')
        }
      }),
    )
  })
})
