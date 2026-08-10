import type { Dict } from './types'

/**
 * Dashboard strings.
 *
 * No metric *names* live here: those come from the backend `MetricSemantics` contract, so putting a
 * translation of one in this file would create a second source of truth for what a metric is called.
 */
export const en: Dict = {
  // Zone 1 — how much has been recorded here
  totalEvaluations: 'Total Evaluations',
  totalPerfRuns: 'Performance Runs',
  modelsEvaluated: 'Models Evaluated',
  latestRun: 'Latest Run',
  neverText: '—',

  // Zone 2 — results aggregated by what they measure
  tabAll: 'All',
  tabEval: 'Eval',
  tabPerf: 'Perf',
  model: 'Model',
  benchmark: 'Benchmark',
  latest: 'Latest',
  trend: 'Trend',
  runsCol: 'Runs',
  lastRun: 'Last run',
  sortBy: 'Sort by ${column}',
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

  // Shared
  noReportsHint: 'Enter an output directory and click Scan to discover reports',
  welcomeTitle: 'Welcome to EvalScope',
  welcomeDesc: 'Enter an output directory path and scan to get started',
  filter_eval: 'Evaluation',
  filter_perf: 'Benchmark',
}

export const zh: Dict = {
  totalEvaluations: '评测总数',
  totalPerfRuns: '性能压测',
  modelsEvaluated: '已评估模型',
  latestRun: '最近运行',
  neverText: '—',

  tabAll: '全部',
  tabEval: '评测',
  tabPerf: '压测',
  model: '模型',
  benchmark: 'Benchmark',
  latest: '最新',
  trend: '趋势',
  runsCol: '次数',
  lastRun: '最后运行',
  sortBy: '按${column}排序',
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

  noReportsHint: '输入输出目录路径并点击扫描来发现报告',
  welcomeTitle: '欢迎使用 EvalScope',
  welcomeDesc: '输入输出目录路径并扫描即可开始',
  filter_eval: '评测',
  filter_perf: '压测',
}
