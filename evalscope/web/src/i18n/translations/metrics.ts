import type { Dict } from './types'

/**
 * Metric-related UI strings.
 *
 * This dictionary deliberately holds no metric *names*: a metric's display name comes from the
 * backend `MetricSemantics` contract, so translating names here would reintroduce a second source
 * of truth. Only the surrounding UI wording lives here.
 */
export const en: Dict = {
  higherIsBetter: 'Higher is better',
  lowerIsBetter: 'Lower is better',
  diagnostics: 'Diagnostic metrics',
  value: 'Value',
  multiplePrimaryMetrics: '${count} primary metrics, cannot be merged',
  multipleMetrics: 'Multiple metrics',
}

export const zh: Dict = {
  higherIsBetter: '越高越好',
  lowerIsBetter: '越低越好',
  diagnostics: '诊断指标',
  value: '数值',
  multiplePrimaryMetrics: '${count} 个主指标，不可合并',
  multipleMetrics: '多个指标',
}
