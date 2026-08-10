/**
 * Static constraints on the metric domain.
 *
 * Feature: metric-semantics-governance, Property 42: the frontend contains no metric-name
 * inference. These assertions read the source tree rather than exercising behaviour, because the
 * property being protected is the *absence* of a mechanism: once a name table or a `metrics[0]`
 * shortcut reappears anywhere, metric semantics silently have two sources of truth again.
 */

import { describe, expect, it } from 'vitest'
import { readFileSync, readdirSync, statSync } from 'node:fs'
import { join, relative } from 'node:path'

import * as metricDomain from './index'

const SRC_ROOT = join(__dirname, '..', '..')

/** Files that legitimately mention the forbidden patterns: the tests describing them. */
const ALLOWED_SUFFIXES = ['.test.ts', '.test.tsx', '__arbitraries__.ts']

function sourceFiles(dir: string): string[] {
  const entries = readdirSync(dir)
  const files: string[] = []
  for (const entry of entries) {
    const full = join(dir, entry)
    if (statSync(full).isDirectory()) {
      if (entry === 'node_modules' || entry === 'test') continue
      files.push(...sourceFiles(full))
      continue
    }
    if (!/\.tsx?$/.test(entry)) continue
    if (ALLOWED_SUFFIXES.some((suffix) => entry.endsWith(suffix))) continue
    files.push(full)
  }
  return files
}

describe('metric domain public surface', () => {
  it('exports exactly the expected primitives', () => {
    // `getValuePosition` and `getBoundedQualityRatio` are deliberately separate: position sizes a
    // bar (never inverted), quality colours it (inverted for lower-is-better). Collapsing them is
    // what made a 4.3% WER draw a 95.7% full bar.
    // `formatDifference` is here rather than duplicated per feature, because "a difference of a
    // percentage is percentage points" is one rule and had already been written twice.
    // `MISSING_PLACEHOLDER` / `roundHalfUp` are internals of `formatMetric` and stay unexported.
    expect(Object.keys(metricDomain).sort()).toEqual([
      'formatDifference',
      'formatMetric',
      'getBoundedQualityRatio',
      'getComparisonVerdict',
      'getValuePosition',
    ])
  })

  it('has no metric name resolution helper', () => {
    for (const forbidden of ['resolveMetricKey', 'getMetricSpec', 'formatMetricByKey', 'formatScore']) {
      expect(metricDomain).not.toHaveProperty(forbidden)
    }
  })
})

describe('Property 42: no metric name inference in the frontend', () => {
  const files = sourceFiles(SRC_ROOT)

  it('finds source files to scan', () => {
    expect(files.length).toBeGreaterThan(20)
  })

  it('contains no metric alias table or name-keyed registry', () => {
    const offenders: string[] = []
    for (const file of files) {
      const text = readFileSync(file, 'utf-8')
      if (/METRIC_ALIASES|METRIC_REGISTRY|EVALUATION_METRIC_SPECS|PERFORMANCE_METRIC_SPECS/.test(text)) {
        offenders.push(relative(SRC_ROOT, file))
      }
    }
    expect(offenders).toEqual([])
  })

  it('never reads metrics[0] to pick a metric', () => {
    const offenders: string[] = []
    for (const file of files) {
      const text = readFileSync(file, 'utf-8')
      if (/metrics\[0\]/.test(text)) {
        offenders.push(relative(SRC_ROOT, file))
      }
    }
    expect(offenders).toEqual([])
  })

  it('no longer imports the deleted registry or display spec modules', () => {
    const offenders: string[] = []
    for (const file of files) {
      const text = readFileSync(file, 'utf-8')
      if (/domain\/metric\/(registry|MetricDisplaySpec)/.test(text)) {
        offenders.push(relative(SRC_ROOT, file))
      }
    }
    expect(offenders).toEqual([])
  })
})

describe('i18n dictionaries are not mixed up', () => {
  it('does not use the generated HTML report keys in the React app', () => {
    // `col.*` and `card.*` belong to evalscope/report/template/js/i18n_eval.js, the dictionary of
    // the standalone HTML report. Using one here renders the raw key on screen, which is silent:
    // nothing throws, the header simply reads `col.metric`.
    const offenders: string[] = []
    for (const file of sourceFiles(SRC_ROOT)) {
      const content = readFileSync(file, 'utf8')
      if (/\bt\(\s*['"](col|card)\./.test(content)) {
        offenders.push(relative(SRC_ROOT, file))
      }
    }
    expect(offenders).toEqual([])
  })
})
