// Feature: frontend-refactor-2026-07, buildCompareModel (perf) property tests.
//
// Merged property coverage for the perf comparison model. Each property keeps
// its own fixtures scoped inside its describe block:
//   - Property 15: delta summary field completeness;
//   - Property 16: delta sign matches the metric direction;
//   - Property 17: default baseline is the oldest run;
//   - Property 18: low-sample tier boundaries;
//   - Property 20: missing-data metrics de-emphasized, computable ones kept;
//   - Property 21: symmetric config diff computation.
//
// Validates: Requirements 9.1, 9.2, 9.4, 9.5, 9.6, 9.7, 9.8, 9.13, 9.14

import { describe, expect, it } from 'vitest'
import fc from 'fast-check'

import type { PerfDetailResponse } from '../../api/types'
import type { FormattedMetric } from '../metric/metricFormat'
import {
  buildCompareModel,
  classifySampleSize,
  type DeltaVerdict,
  type MetricDelta,
  type SampleTier,
} from './compareModel'

/* ─── Property 21: symmetric config diff computation ──────────── */

describe('buildCompareModel — symmetric config diff (Property 21: symmetric config diff computation)', () => {
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

/* ─── Property 17: default baseline is the oldest run ─────────── */

describe('buildCompareModel default baseline (Property 17: default baseline is the oldest run)', () => {
  /** Build a minimal `PerfDetailResponse` carrying just the fields the model reads. */
  function makeRun(
    path: string,
    generatedAt: string,
    summaryRows: (string | number)[][],
  ): PerfDetailResponse {
    return {
      path,
      model: 'test-model',
      api_type: 'openai_api',
      dataset: 'shared-workload',
      generated_at: generatedAt,
      basic_info: {},
      summary_columns: ['Metric', 'Value'],
      summary_rows: summaryRows,
      best_config: {},
      recommendations: [],
      num_runs: 1,
      is_embedding: false,
      has_html: false,
    }
  }

  /** A handful of random summary rows so runs differ in their metric payloads. */
  const summaryRowsArb = fc.array(
    fc.tuple(
      fc.string({ minLength: 1, maxLength: 12 }),
      fc.double({ min: -1e6, max: 1e6, noNaN: true, noDefaultInfinity: true }),
    ),
    { minLength: 0, maxLength: 5 },
  )

  /**
   * Epoch-millis range that stays inside the safe `Date` window, so
   * `new Date(ms).toISOString()` always yields a parseable ISO-8601 timestamp.
   */
  const epochMillisArb = fc.integer({ min: 0, max: 4102444800000 }) // 1970-01-01 .. 2100-01-01

  /**
   * A comparison set of 2+ runs, each with a distinct `path` and a distinct
   * `generated_at` ISO timestamp (drawn from unique epoch-millis values) plus
   * random summary rows.
   */
  const runsArb = fc
    .uniqueArray(epochMillisArb, { minLength: 2, maxLength: 8 })
    .chain((millis) =>
      fc.tuple(...millis.map(() => summaryRowsArb)).map((rowsPerRun) =>
        millis.map((ms, index) =>
          makeRun(`runs/run-${index}`, new Date(ms).toISOString(), rowsPerRun[index]),
        ),
      ),
    )

  it('selects the oldest run as the default baseline when none is specified', () => {
    fc.assert(
      fc.property(runsArb, (runs) => {
        // Empty baselineId ⇒ the model must fall back to the default (oldest) run.
        const model = buildCompareModel(runs, '')

        const chosen = runs.find((run) => run.path === model.baselineId)
        expect(chosen).toBeDefined()

        const chosenTime = Date.parse((chosen as PerfDetailResponse).generated_at)
        // The chosen baseline's timestamp is not later than any other run's.
        for (const run of runs) {
          expect(chosenTime).toBeLessThanOrEqual(Date.parse(run.generated_at))
        }
      }),
    )
  })
})

/* ─── Property 16: delta sign matches the metric direction ────── */

describe('buildCompareModel verdict direction (Property 16: Delta sign matches the metric direction)', () => {
  /**
   * Metric label whose `resolvePerfMetricSpec` direction is `higher-is-better`
   * (no latency/ttft/tpot/time/delay/duration keyword).
   */
  const HIGHER_IS_BETTER_METRIC = 'Output throughput (tokens/s)'
  /** Metric label whose inferred direction is `lower-is-better` (contains "latency"). */
  const LOWER_IS_BETTER_METRIC = 'Average latency (s)'

  const BASELINE_PATH = 'runs/baseline'
  const CANDIDATE_PATH = 'runs/candidate'

  /** Build a minimal `PerfDetailResponse` carrying just the fields the model reads. */
  function makeRun(
    path: string,
    generatedAt: string,
    metricValues: Record<string, number>,
  ): PerfDetailResponse {
    return {
      path,
      model: 'test-model',
      api_type: 'openai_api',
      dataset: 'shared-workload',
      generated_at: generatedAt,
      basic_info: {},
      summary_columns: ['Metric', 'Value'],
      summary_rows: Object.entries(metricValues).map(([key, value]) => [key, value]),
      best_config: {},
      recommendations: [],
      num_runs: 1,
      is_embedding: false,
      has_html: false,
    }
  }

  /** The direction rule under test, expressed independently of the SUT. */
  function expectedVerdict(baseline: number, candidate: number, higherIsBetter: boolean): DeltaVerdict {
    if (candidate === baseline) return 'neutral'
    const candidateIsHigher = candidate > baseline
    return candidateIsHigher === higherIsBetter ? 'improvement' : 'regression'
  }

  /**
   * A baseline value plus a candidate that is deliberately equal to, greater
   * than, or less than the baseline. Constructing the candidate from the baseline
   * (rather than drawing two independent values) guarantees all three relations
   * are exercised, and building `>` / `<` cases with a strictly positive gap
   * keeps them from collapsing back into equality under floating point.
   */
  const valuePairArb = fc
    .record({
      baseline: fc.double({ min: -1e6, max: 1e6, noNaN: true, noDefaultInfinity: true }),
      relation: fc.constantFrom<'equal' | 'greater' | 'less'>('equal', 'greater', 'less'),
      gap: fc.double({ min: 1e-3, max: 1e6, noNaN: true, noDefaultInfinity: true }),
    })
    .map(({ baseline, relation, gap }) => {
      if (relation === 'equal') return { baseline, candidate: baseline }
      if (relation === 'greater') return { baseline, candidate: baseline + gap }
      return { baseline, candidate: baseline - gap }
    })

  /** Locate the delta for a metric key in the built model. */
  function verdictFor(runs: PerfDetailResponse[], metricKey: string): DeltaVerdict {
    const model = buildCompareModel(runs, BASELINE_PATH)
    const delta = model.deltas.find((d) => d.metricKey === metricKey)
    if (!delta) throw new Error(`missing delta for ${metricKey}`)
    return delta.verdict
  }

  it('follows higher-is-better for a throughput metric', () => {
    fc.assert(
      fc.property(valuePairArb, ({ baseline, candidate }) => {
        const runs = [
          makeRun(BASELINE_PATH, '2026-06-01T00:00:00.000Z', { [HIGHER_IS_BETTER_METRIC]: baseline }),
          makeRun(CANDIDATE_PATH, '2026-07-01T00:00:00.000Z', { [HIGHER_IS_BETTER_METRIC]: candidate }),
        ]
        expect(verdictFor(runs, HIGHER_IS_BETTER_METRIC)).toBe(
          expectedVerdict(baseline, candidate, true),
        )
      }),
    )
  })

  it('follows lower-is-better for a latency metric', () => {
    fc.assert(
      fc.property(valuePairArb, ({ baseline, candidate }) => {
        const runs = [
          makeRun(BASELINE_PATH, '2026-06-01T00:00:00.000Z', { [LOWER_IS_BETTER_METRIC]: baseline }),
          makeRun(CANDIDATE_PATH, '2026-07-01T00:00:00.000Z', { [LOWER_IS_BETTER_METRIC]: candidate }),
        ]
        expect(verdictFor(runs, LOWER_IS_BETTER_METRIC)).toBe(
          expectedVerdict(baseline, candidate, false),
        )
      }),
    )
  })

  it('yields neutral on equal values and opposite verdicts across directions on unequal values', () => {
    fc.assert(
      fc.property(valuePairArb, ({ baseline, candidate }) => {
        const runs = [
          makeRun(BASELINE_PATH, '2026-06-01T00:00:00.000Z', {
            [HIGHER_IS_BETTER_METRIC]: baseline,
            [LOWER_IS_BETTER_METRIC]: baseline,
          }),
          makeRun(CANDIDATE_PATH, '2026-07-01T00:00:00.000Z', {
            [HIGHER_IS_BETTER_METRIC]: candidate,
            [LOWER_IS_BETTER_METRIC]: candidate,
          }),
        ]
        const model = buildCompareModel(runs, BASELINE_PATH)
        const higher = model.deltas.find((d) => d.metricKey === HIGHER_IS_BETTER_METRIC)?.verdict
        const lower = model.deltas.find((d) => d.metricKey === LOWER_IS_BETTER_METRIC)?.verdict

        if (candidate === baseline) {
          // Equal values are neutral regardless of direction.
          expect(higher).toBe('neutral')
          expect(lower).toBe('neutral')
        } else {
          // Same values, opposite directions ⇒ opposite verdicts.
          expect(higher).not.toBe('neutral')
          expect(lower).not.toBe('neutral')
          expect(higher).not.toBe(lower)
        }
      }),
    )
  })
})

/* ─── Property 15: delta summary field completeness ───────────── */

describe('buildCompareModel — delta field completeness (Property 15: Delta summary field completeness)', () => {
  /** The four verdicts buildCompareModel may emit. */
  const VERDICTS: DeltaVerdict[] = ['improvement', 'regression', 'neutral', 'incomputable']

  /**
   * Assert a value is a fully-formed FormattedMetric: an object exposing string
   * `primary` and `raw` fields (Property 15 requires these on every delta cell).
   */
  function assertFormattedMetric(field: unknown): asserts field is FormattedMetric {
    if (typeof field !== 'object' || field === null) {
      throw new Error('Expected a formatted metric object')
    }
    const metric = field as Record<string, unknown>
    if (typeof metric.primary !== 'string' || typeof metric.raw !== 'string') {
      throw new Error('Expected formatted metric primary/raw strings')
    }
  }

  /** Assert a single MetricDelta carries all four summary fields plus a verdict. */
  function assertDeltaComplete(delta: MetricDelta): void {
    if (typeof delta.metricKey !== 'string') throw new Error('Expected a metric key')
    assertFormattedMetric(delta.baseline)
    assertFormattedMetric(delta.candidate)
    assertFormattedMetric(delta.absoluteDelta)
    assertFormattedMetric(delta.percentDelta)
    if (!VERDICTS.includes(delta.verdict)) throw new Error(`Unexpected verdict: ${delta.verdict}`)
  }

  // A pool of representative perf metric names spanning lower-is-better latency
  // metrics, higher-is-better throughput metrics and a bounded success rate, so
  // generated runs mix directions and boundedness.
  const METRIC_NAME_POOL = [
    'Average latency (s)',
    'Average TTFT (s)',
    'Average TPOT (s)',
    'Output throughput (tokens/s)',
    'Request throughput (req/s)',
    'Success rate',
    'Number of requests',
    'Concurrency',
    'P99 latency (s)',
    'Total tokens',
  ]

  /** A single summary row: `[metricName, value]`. */
  const summaryRowArb: fc.Arbitrary<(string | number)[]> = fc.tuple(
    fc.constantFrom(...METRIC_NAME_POOL),
    fc.double({ min: 0, max: 100000, noNaN: true, noDefaultInfinity: true }),
  )

  /**
   * Generate a set of summary rows with at least one metric. Duplicate metric
   * names are collapsed by buildCompareModel (first occurrence wins), so the
   * generator does not need to dedupe.
   */
  const summaryRowsArb: fc.Arbitrary<(string | number)[][]> = fc.array(summaryRowArb, {
    minLength: 1,
    maxLength: 8,
  })

  /** ISO-8601 timestamp generator spanning a range of distinct instants. */
  const timestampArb: fc.Arbitrary<string> = fc
    .integer({ min: 0, max: 3_000_000_000 })
    .map((secs) => new Date(secs * 1000).toISOString())

  /** Generate a full PerfDetailResponse with a random metric set and identity. */
  function perfRunArb(index: number): fc.Arbitrary<PerfDetailResponse> {
    return fc.record({
      path: fc
        .array(fc.constantFrom(...'abcdefghijklmnopqrstuvwxyz0123456789_-'), { minLength: 1, maxLength: 12 })
        .map((parts) => `perf/${parts.join('')}-${index}`),
      model: fc.constantFrom('model-a', 'model-b', 'model-c'),
      api_type: fc.constant('openai_api'),
      dataset: fc.constantFrom('openqa', 'longalpaca', 'sharegpt'),
      generated_at: timestampArb,
      basic_info: fc.constant<Record<string, string>>({ 'Total requests': '100' }),
      summary_columns: fc.constant(['Metric', 'Value']),
      summary_rows: summaryRowsArb,
      best_config: fc.dictionary(
        fc.constantFrom('Concurrency', 'Rate', 'Requests', 'Max tokens'),
        fc.oneof(fc.integer({ min: 0, max: 1000 }), fc.double({ min: 0, max: 1000, noNaN: true })).map(String),
        { maxKeys: 4 },
      ),
      recommendations: fc.constant<string[]>([]),
      num_runs: fc.constant(1),
      is_embedding: fc.constant(false),
      has_html: fc.constant(true),
    })
  }

  /** Generate 2+ runs, each with an independent path so ids stay distinct. */
  const runsArb: fc.Arbitrary<PerfDetailResponse[]> = fc
    .integer({ min: 2, max: 5 })
    .chain((count) => fc.tuple(...Array.from({ length: count }, (_, i) => perfRunArb(i))))

  it('extracts metrics and sample counts from the real archive wide-table shape', () => {
    const columns = [
      'Conc.', 'Rate', 'RPS', 'Avg Lat.(s)', 'P99 Lat.(s)', 'Avg TTFT(ms)',
      'P99 TTFT(ms)', 'Avg TPOT(ms)', 'P99 TPOT(ms)', 'Gen. tok/s', 'Success Rate',
    ]
    const makeWideRun = (
      path: string,
      generatedAt: string,
      requests: number,
      row: (string | number)[],
    ): PerfDetailResponse => ({
      path,
      model: 'qwen-plus',
      api_type: 'openai',
      dataset: 'openqa',
      generated_at: generatedAt,
      basic_info: { 'Total Requests': String(requests) },
      summary_columns: columns,
      summary_rows: [row],
      best_config: {},
      recommendations: [],
      num_runs: 1,
      is_embedding: false,
      has_html: true,
    })
    const baseline = makeWideRun(
      'run-p1', '2026-07-15T15:37:19', 4,
      ['1', 'INF', '1.0326', '0.968', '1.080', '467.79', '597.30', '16.13', '17.10', '33.04', '100.0%'],
    )
    const candidate = makeWideRun(
      'run-p2', '2026-07-15T15:39:04', 6,
      ['1', 'INF', '1.9256', '1.027', '1.210', '493.17', '656.15', '17.21', '17.77', '61.62', '100.0%'],
    )

    const model = buildCompareModel([candidate, baseline], '')
    const rps = model.deltas.find((delta) => delta.metricKey === 'rps')
    const latency = model.deltas.find((delta) => delta.metricKey === 'latency')
    const success = model.deltas.find((delta) => delta.metricKey === 'success_rate')

    expect(model.sampleCounts).toEqual({ 'run-p1': 4, 'run-p2': 6 })
    expect(rps?.baseline.primary).toBe('1.03 req/s')
    expect(rps?.candidate.primary).toBe('1.93 req/s')
    expect(rps?.verdict).toBe('improvement')
    expect(latency?.verdict).toBe('regression')
    expect(success?.absoluteDelta.primary).toBe('0.0 pp')
    expect(model.configDiff).toEqual([{ key: 'Number of requests', baseline: '4', candidate: '6' }])
    expect(model.workloadMismatch).toBe(false)
  })

  it('does not combine or compare metrics from different wide-table workload rows', () => {
    const columns = ['Conc.', 'Rate', 'RPS', 'Avg Lat.(s)']
    const makeRun = (path: string, generatedAt: string, rows: (string | number)[][]): PerfDetailResponse => ({
      path,
      model: 'qwen-plus',
      api_type: 'openai',
      dataset: 'openqa',
      generated_at: generatedAt,
      basic_info: { 'Total Requests': '20' },
      summary_columns: columns,
      summary_rows: rows,
      best_config: {},
      recommendations: [],
      num_runs: 1,
      is_embedding: false,
      has_html: true,
    })
    const baseline = makeRun('baseline', '2026-07-15T15:37:19', [
      ['1', 'INF', 10, 1],
      ['2', 'INF', 20, 2],
    ])
    const candidate = makeRun('candidate', '2026-07-15T15:39:04', [
      ['1', 'INF', 11, 0.9],
      ['2', 'INF', 21, 2.1],
    ])

    const model = buildCompareModel([baseline, candidate], '')
    expect(model.workloadMismatch).toBe(false)
    expect(model.deltas.find((delta) => delta.metricKey === 'rps')?.baseline.raw).toBe('10.00')
    expect(model.deltas.find((delta) => delta.metricKey === 'latency')?.baseline.raw).toBe('1.00')

    const mismatched = makeRun('mismatch', '2026-07-15T15:40:04', [['8', 'INF', 80, 8]])
    const mismatchModel = buildCompareModel([baseline, mismatched], '')
    expect(mismatchModel.workloadMismatch).toBe(true)
    expect(mismatchModel.deltas).toEqual([])
  })

  it('keeps every delta field complete for default and explicit baselines', () => {
    fc.assert(
      fc.property(runsArb, fc.nat(), (runs, baselineIndex) => {
        const baselineId = runs[baselineIndex % runs.length].path
        const models = [buildCompareModel(runs, ''), buildCompareModel(runs, baselineId)]

        for (const model of models) {
          if (model.deltas.length === 0) throw new Error('Expected at least one metric delta')
          for (const delta of model.deltas) {
            assertDeltaComplete(delta)
          }
        }
        expect(models[1].baselineId).toBe(baselineId)
      }),
      { numRuns: 40 },
    )
  })
})

/* ─── Property 20: missing-data de-emphasis ───────────────────── */

describe('buildCompareModel — missing-data de-emphasis (Property 20: missing-data metrics are de-emphasized while computable metrics are preserved)', () => {
  /** Non-numeric cell tokens: toNumeric() returns null for each (missing value). */
  const NON_NUMERIC_TOKENS = ['N/A', '-', 'error', 'null', '']

  /**
   * Per-metric scenario describing how a metric appears across the two runs.
   * Every scenario except `bothValid` leaves the metric missing on one side, so
   * its delta must be incomputable.
   */
  type Scenario =
    | 'bothValid' // valid numeric on both sides → computable
    | 'missingBaseline' // row absent from baseline run → incomputable
    | 'missingCandidate' // row absent from candidate run → incomputable
    | 'nonNumericBaseline' // baseline cell non-numeric → incomputable
    | 'nonNumericCandidate' // candidate cell non-numeric → incomputable

  const MISSING_SCENARIOS: Scenario[] = [
    'missingBaseline',
    'missingCandidate',
    'nonNumericBaseline',
    'nonNumericCandidate',
  ]

  /** A single generated metric case, plus its expected computability. */
  interface MetricCase {
    name: string
    scenario: Scenario
    baselineValue: number
    candidateValue: number
    nonNumericToken: string
  }

  // Fixed, distinct run identities. The baseline run is strictly older so the
  // default baseline selection resolves to `baselineRun` deterministically.
  const BASELINE_TIMESTAMP = '2020-01-01T00:00:00.000Z'
  const CANDIDATE_TIMESTAMP = '2021-01-01T00:00:00.000Z'
  const BASELINE_PATH = 'perf/baseline'
  const CANDIDATE_PATH = 'perf/candidate'

  // Realistic perf metric names spanning both directions and boundedness so spec
  // resolution is exercised while the property is checked.
  const METRIC_NAME_POOL = [
    'Average latency (s)',
    'Average TTFT (s)',
    'Average TPOT (s)',
    'Output throughput (tokens/s)',
    'Request throughput (req/s)',
    'Success rate',
    'P99 latency (s)',
    'Total tokens',
  ]

  /** Positive, non-zero values so a computable percent delta is always defined. */
  const metricValueArb: fc.Arbitrary<number> = fc.double({
    min: 1,
    max: 100000,
    noNaN: true,
    noDefaultInfinity: true,
  })

  /** Build a metric case for a given (unique) name with a randomly chosen scenario. */
  function metricCaseArb(name: string, scenario: fc.Arbitrary<Scenario>): fc.Arbitrary<MetricCase> {
    return fc.record({
      name: fc.constant(name),
      scenario,
      baselineValue: metricValueArb,
      candidateValue: metricValueArb,
      nonNumericToken: fc.constantFrom(...NON_NUMERIC_TOKENS),
    })
  }

  /**
   * Generate a set of metric cases that always includes at least one `bothValid`
   * metric (a computable delta to retain) and at least one missing-data metric (an
   * incomputable delta to de-emphasize), plus a random mix in between. Names stay
   * unique so each metric yields exactly one delta.
   */
  const metricCasesArb: fc.Arbitrary<MetricCase[]> = fc
    .uniqueArray(fc.constantFrom(...METRIC_NAME_POOL), {
      minLength: 2,
      maxLength: METRIC_NAME_POOL.length,
    })
    .chain((names) =>
      fc.tuple(
        // First name: guaranteed computable metric.
        metricCaseArb(names[0], fc.constant<Scenario>('bothValid')),
        // Second name: guaranteed missing-data metric.
        metricCaseArb(names[1], fc.constantFrom(...MISSING_SCENARIOS)),
        // Remaining names: any scenario.
        fc.tuple(
          ...names.slice(2).map((name) =>
            metricCaseArb(
              name,
              fc.constantFrom<Scenario>('bothValid', ...MISSING_SCENARIOS),
            ),
          ),
        ),
      ),
    )
    .map(([both, missing, rest]) => [both, missing, ...rest])

  /** The baseline-side cell for a metric case, or undefined when the row is absent. */
  function baselineCell(mc: MetricCase): string | number | undefined {
    if (mc.scenario === 'missingBaseline') return undefined
    if (mc.scenario === 'nonNumericBaseline') return mc.nonNumericToken
    return mc.baselineValue
  }

  /** The candidate-side cell for a metric case, or undefined when the row is absent. */
  function candidateCell(mc: MetricCase): string | number | undefined {
    if (mc.scenario === 'missingCandidate') return undefined
    if (mc.scenario === 'nonNumericCandidate') return mc.nonNumericToken
    return mc.candidateValue
  }

  /** Assemble summary rows for one side from the metric cases, dropping absent rows. */
  function toSummaryRows(
    cases: MetricCase[],
    pick: (mc: MetricCase) => string | number | undefined,
  ): (string | number)[][] {
    const rows: (string | number)[][] = []
    for (const mc of cases) {
      const cell = pick(mc)
      if (cell === undefined) continue
      rows.push([mc.name, cell])
    }
    return rows
  }

  /** Build a PerfDetailResponse with a fixed identity and the given summary rows. */
  function makeRun(
    path: string,
    generatedAt: string,
    summaryRows: (string | number)[][],
  ): PerfDetailResponse {
    return {
      path,
      model: 'model-a',
      api_type: 'openai_api',
      dataset: 'openqa',
      generated_at: generatedAt,
      basic_info: { 'Total requests': '100' },
      summary_columns: ['Metric', 'Value'],
      summary_rows: summaryRows,
      best_config: {},
      recommendations: [],
      num_runs: 1,
      is_embedding: false,
      has_html: true,
    }
  }

  it('marks metrics missing on either side incomputable while keeping both-sided metrics computable', () => {
    fc.assert(
      fc.property(metricCasesArb, (cases) => {
        const baselineRun = makeRun(
          BASELINE_PATH,
          BASELINE_TIMESTAMP,
          toSummaryRows(cases, baselineCell),
        )
        const candidateRun = makeRun(
          CANDIDATE_PATH,
          CANDIDATE_TIMESTAMP,
          toSummaryRows(cases, candidateCell),
        )

        const model = buildCompareModel([baselineRun, candidateRun], '')

        // The older run is the baseline; the newer run is the candidate.
        expect(model.baselineId).toBe(BASELINE_PATH)
        expect(model.candidateId).toBe(CANDIDATE_PATH)

        const deltaByKey = new Map(model.deltas.map((delta) => [delta.metricKey, delta]))

        for (const mc of cases) {
          const delta = deltaByKey.get(mc.name)
          expect(delta).toBeDefined()
          if (!delta) continue

          if (mc.scenario === 'bothValid') {
            // Computable metric retained: verdict is a real direction and the
            // absolute delta is a fully-formed (non-missing) value.
            expect(delta.verdict).not.toBe('incomputable')
            expect(delta.absoluteDelta.isMissing).toBe(false)
            expect(delta.percentDelta.isMissing).toBe(false)
            expect(delta.baseline.isMissing).toBe(false)
            expect(delta.candidate.isMissing).toBe(false)
          } else {
            // Missing on one side → de-emphasized incomputable delta.
            expect(delta.verdict).toBe('incomputable')
            expect(delta.absoluteDelta.isMissing).toBe(true)
            expect(delta.percentDelta.isMissing).toBe(true)
          }
        }
      }),
    )
  })
})

/* ─── Property 18: low-sample tier boundaries ─────────────────── */

describe('classifySampleSize — low-sample tier boundaries (Property 18: low-sample tier boundaries)', () => {
  /** The two tier thresholds, mirrored here independently of the implementation. */
  const CRITICAL_THRESHOLD = 30
  const WARN_THRESHOLD = 100

  /**
   * Independent reference classifier for a non-negative sample count. Written
   * separately from the implementation so the property cross-checks behaviour
   * rather than restating the same code.
   */
  function expectedTier(n: number): SampleTier {
    if (n < CRITICAL_THRESHOLD) return 'critical'
    if (n < WARN_THRESHOLD) return 'warn'
    return 'ok'
  }

  it('matches the reference classifier for any non-negative sample count', () => {
    fc.assert(
      fc.property(fc.nat({ max: 1_000_000 }), (n) => {
        expect(classifySampleSize(n)).toBe(expectedTier(n))
      }),
    )
  })

  it('classifies the explicit boundary values 0/29/30/99/100/101', () => {
    expect(classifySampleSize(0)).toBe('critical')
    expect(classifySampleSize(29)).toBe('critical')
    expect(classifySampleSize(30)).toBe('warn')
    expect(classifySampleSize(99)).toBe('warn')
    expect(classifySampleSize(100)).toBe('ok')
    expect(classifySampleSize(101)).toBe('ok')
  })
})
