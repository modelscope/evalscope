import type { Dict } from './types'

export const en: Dict = {
  scoreAxisLabel: 'Score (%)',
  scoreHistogramAriaLabel: 'Score distribution histogram',
  groupedBarAriaLabel: 'Grouped bar chart',
  radarAriaLabel: 'Radar chart',
  loading: 'Loading chart…',
  loadError: 'Failed to load chart',
  errorTimeout: 'The chart request timed out.',
  error4xx: 'The chart could not be found.',
  error5xx: 'The chart service encountered an error.',
  errorNetwork: 'A network error prevented the chart from loading.',
  retry: 'Retry',
  fallbackTitle: 'Data table',
  fallbackHint: 'Showing the same underlying data as a table.',
  singleValueAriaLabel: 'Single metric value',
  emptyTitle: 'No dimensions to visualize',
}

export const zh: Dict = {
  scoreAxisLabel: '得分 (%)',
  scoreHistogramAriaLabel: '得分分布直方图',
  groupedBarAriaLabel: '分组柱状图',
  radarAriaLabel: '雷达图',
  loading: '正在加载图表…',
  loadError: '图表加载失败',
  errorTimeout: '图表请求超时。',
  error4xx: '未找到图表。',
  error5xx: '图表服务发生错误。',
  errorNetwork: '网络错误导致图表无法加载。',
  retry: '重试',
  fallbackTitle: '数据表',
  fallbackHint: '以数据表形式展示相同的底层数据。',
  singleValueAriaLabel: '单项指标值',
  emptyTitle: '暂无可视化维度',
}
