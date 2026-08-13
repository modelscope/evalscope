import { describe, expect, it } from 'vitest'
import type { ReportSummary } from '@/api/types'
import { buildTrendSeries } from './trendSeries'

const semantics = {
  semantic_id: 'quality.accuracy.ratio',
  metric_name: 'Accuracy',
  kind: 'quality' as const,
  direction: 'higher_is_better' as const,
  raw_unit: null,
  value_range: { min: 0, max: 1 },
  display_kind: 'percent' as const,
  display_multiplier: 100,
  display_unit: '%',
  display_precision: 1,
}

function report(runId: string, modelId: string, benchmark: string, score: number): ReportSummary {
  return {
    run_id: runId,
    model_id: modelId,
    model_name: modelId,
    dataset_name: benchmark,
    dataset_pretty_name: benchmark.toUpperCase(),
    num_samples: 10,
    timestamp: `2026-08-${runId.padStart(2, '0')}T10:00:00`,
    primary_metrics: [{
      dataset_name: benchmark,
      dataset_pretty_name: benchmark.toUpperCase(),
      identity: { name: 'accuracy', aggregation: 'mean', dimensions: {} },
      score,
      semantics,
    }],
  }
}

describe('buildTrendSeries', () => {
  it('compares repeated runs only within one model and benchmark', () => {
    const series = buildTrendSeries([
      report('1', 'model-a', 'gsm8k', 0.7),
      report('2', 'model-a', 'gsm8k', 0.8),
      report('3', 'model-b', 'gsm8k', 0.9),
      report('4', 'model-a', 'mmlu', 0.6),
    ])

    expect(series).toHaveLength(1)
    expect(series[0]).toMatchObject({ modelId: 'model-a', benchmark: 'gsm8k' })
    expect(series[0].points.map((point) => point.score)).toEqual([0.7, 0.8])
  })

  it('does not join a changed primary metric identity into the same line', () => {
    const reports = [report('1', 'model-a', 'gsm8k', 0.7), report('2', 'model-a', 'gsm8k', 0.8)]
    const changed = report('3', 'model-a', 'gsm8k', 0.9)
    changed.primary_metrics[0].identity = { name: 'f1', aggregation: 'mean', dimensions: {} }

    const series = buildTrendSeries([...reports, changed])

    expect(series).toHaveLength(1)
    expect(series[0].identity.name).toBe('accuracy')
    expect(series[0].points).toHaveLength(2)
  })
})
