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
  noPrimaryMetric: 'No metric reported',
}

export const zh: Dict = {
  higherIsBetter: '越高越好',
  lowerIsBetter: '越低越好',
  noPrimaryMetric: '未报告指标',
}
