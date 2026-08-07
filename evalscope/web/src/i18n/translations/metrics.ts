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
  noPrimaryMetric: 'No metric reported',
  inferredPrimary: 'This benchmark declares no primary metric; one was inferred to show a value',
  multipleMetrics: 'Multiple metrics',
}

export const zh: Dict = {
  higherIsBetter: '越高越好',
  lowerIsBetter: '越低越好',
  diagnostics: '诊断指标',
  value: '数值',
  multiplePrimaryMetrics: '${count} 个主指标，不可合并',
  noPrimaryMetric: '未报告指标',
  inferredPrimary: '该 benchmark 未声明主指标，此处推断了一个用于展示',
  multipleMetrics: '多个指标',
}
