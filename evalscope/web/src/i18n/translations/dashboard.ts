import type { Dict } from './types'

/**
 * Dashboard strings.
 *
 * No metric *names* live here: those come from the backend `MetricSemantics` contract, so putting a
 * translation of one in this file would create a second source of truth for what a metric is called.
 */
export const en: Dict = {
  // Zone 1 — what to do next
  startTitle: 'Start',
  actionRunEval: 'Run evaluation',
  actionRunPerf: 'Run benchmark',
  actionCompare: 'Compare models',
  actionBrowse: 'Browse benchmarks',
  repeatTitle: 'Repeat a recent configuration',
  repeatAction: 'Run again',

  // Zone 2 — results aggregated by what they measure
  resultsTitle: 'Results',
  totalsSummary: '${models} models · ${benchmarks} benchmarks · ${cells} combinations · ${runs} runs',
  sort_instability: 'Widest spread',
  sort_recency: 'Most recent',
  model: 'Model',
  benchmark: 'Benchmark',
  latest: 'Latest',
  trend: 'Trend',
  spread: 'Spread',
  runsCol: 'Runs',
  expandRow: 'Show history',
  collapseRow: 'Hide history',
  trendLabel: 'History of ${metric} over ${runs} runs',
  trendDetailLabel: 'Every recorded run; select one to open it',
  statLatest: 'Latest',
  statMean: 'Mean',
  statRange: 'Range',
  statSpread: 'Spread',
  statStddev: 'Std. dev.',
  statRuns: 'Runs',
  singleRunHint: 'Measured once, so there is nothing to compare it against yet.',
  outOfRangeHint:
    'Some runs recorded values outside this metric declared range, so they are shown as recorded '
    + 'and not converted. The benchmark most likely changed the scale it reports.',
  openLatest: 'Open the latest run',

  // Zone 3 — did the run I just started finish
  activityTitle: 'Recent activity',
  viewAll: 'All evaluations',

  // Shared
  noReportsHint: 'Enter an output directory and click Scan to discover reports',
  welcomeTitle: 'Welcome to EvalScope',
  welcomeDesc: 'Enter an output directory path and scan to get started',
  filter_eval: 'Evaluation',
  filter_perf: 'Benchmark',
}

export const zh: Dict = {
  startTitle: '开始',
  actionRunEval: '跑评测',
  actionRunPerf: '跑压测',
  actionCompare: '对比模型',
  actionBrowse: '浏览 Benchmark',
  repeatTitle: '重跑最近的配置',
  repeatAction: '再跑一次',

  resultsTitle: '结果总览',
  totalsSummary: '${models} 个模型 · ${benchmarks} 个 benchmark · ${cells} 个组合 · ${runs} 次运行',
  sort_instability: '波动最大',
  sort_recency: '最近运行',
  model: '模型',
  benchmark: 'Benchmark',
  latest: '最新',
  trend: '趋势',
  spread: '波动',
  runsCol: '次数',
  expandRow: '展开历史',
  collapseRow: '收起历史',
  trendLabel: '${metric} 在 ${runs} 次运行中的历史',
  trendDetailLabel: '全部运行记录，点击任一条打开',
  statLatest: '最新',
  statMean: '均值',
  statRange: '区间',
  statSpread: '波动',
  statStddev: '标准差',
  statRuns: '次数',
  singleRunHint: '只跑过一次，暂无可对比的历史。',
  outOfRangeHint: '部分运行记录的数值超出了该指标声明的量程，因此按原样显示、不做换算。通常是该 benchmark 改变了上报的刻度。',
  openLatest: '打开最近一次运行',

  activityTitle: '最近活动',
  viewAll: '全部评测',

  noReportsHint: '输入输出目录路径并点击扫描来发现报告',
  welcomeTitle: '欢迎使用 EvalScope',
  welcomeDesc: '输入输出目录路径并扫描即可开始',
  filter_eval: '评测',
  filter_perf: '压测',
}
