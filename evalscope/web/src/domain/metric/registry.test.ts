import { describe, expect, it } from 'vitest'

import { MISSING_PLACEHOLDER } from './metricFormat'
import { DEFAULT_METRIC_SPEC } from './MetricDisplaySpec'
import {
  formatMetricByKey,
  formatScore,
  getMetricSpec,
  METRIC_REGISTRY,
  resolveMetricKey,
} from './registry'

// Identity translate: unit keys/labels resolve to themselves so assertions can
// check the literal unit strings and label keys carried by each spec.
const t = (key: string): string => key

describe('resolveMetricKey', () => {
  it('returns canonical keys unchanged', () => {
    expect(resolveMetricKey('accuracy')).toBe('accuracy')
    expect(resolveMetricKey('throughput')).toBe('throughput')
  })

  it('normalizes and resolves known aliases to canonical keys', () => {
    expect(resolveMetricKey('AverageAccuracy')).toBe('accuracy')
    expect(resolveMetricKey('pass@1')).toBe('pass_rate')
    expect(resolveMetricKey('Output TPS')).toBe('throughput')
    expect(resolveMetricKey('Avg Latency')).toBe('latency')
    expect(resolveMetricKey('Time to First Token')).toBe('ttft')
  })

  it('returns the normalized key for unknown metrics', () => {
    expect(resolveMetricKey('Some Weird Metric')).toBe('some_weird_metric')
  })
})

describe('getMetricSpec', () => {
  it('resolves a registered spec without fallback', () => {
    const { spec, isFallback } = getMetricSpec('accuracy')
    expect(isFallback).toBe(false)
    expect(spec).toBe(METRIC_REGISTRY.accuracy)
  })

  it('falls back to the default spec for unknown keys', () => {
    const { spec, isFallback } = getMetricSpec('nonexistent_metric')
    expect(isFallback).toBe(true)
    expect(spec).toBe(DEFAULT_METRIC_SPEC)
  })
})

describe('formatMetricByKey', () => {
  it('renders bounded evaluation metrics as a percentage with a 0-1 raw value', () => {
    const result = formatMetricByKey('accuracy', 0.923, t)
    expect(result.primary).toBe('92.3%')
    expect(result.raw).toBe('0.9230')
    expect(result.isMissing).toBe(false)
    expect(result.isSpecUndefined).toBe(false)
  })

  it('keeps the native unit for unbounded performance metrics (no percentage)', () => {
    const result = formatMetricByKey('throughput', 1234.5, t)
    expect(result.primary).toBe('1234.50 tokens/s')
    expect(result.unitLabel).toBe('tokens/s')
  })

  it('does not convert a value greater than 1 to a percentage for unbounded metrics', () => {
    const result = formatMetricByKey('latency', 12.5, t)
    expect(result.primary).toBe('12.50 s')
    expect(result.primary.includes('%')).toBe(false)
  })

  it('renders a distinct placeholder for missing values instead of 0', () => {
    expect(formatMetricByKey('accuracy', null, t).primary).toBe(MISSING_PLACEHOLDER)
    expect(formatMetricByKey('accuracy', undefined, t).primary).toBe(MISSING_PLACEHOLDER)
    expect(formatMetricByKey('accuracy', NaN, t).isMissing).toBe(true)
    // A legitimate 0 is not treated as missing.
    expect(formatMetricByKey('accuracy', 0, t).isMissing).toBe(false)
    expect(formatMetricByKey('accuracy', 0, t).primary).toBe('0.0%')
  })

  it('flags an undefined display form for unknown metrics without inferring a percentage', () => {
    const result = formatMetricByKey('unknown_metric', 42, t)
    expect(result.isSpecUndefined).toBe(true)
    expect(result.primary).toBe('42.0000')
    expect(result.primary.includes('%')).toBe(false)
  })
})

describe('formatScore', () => {
  it('returns the primary display string for a metric key', () => {
    expect(formatScore('pass_rate', 0.5, t)).toBe('50.0%')
    expect(formatScore('rps', 10, t)).toBe('10.00 req/s')
  })
})
