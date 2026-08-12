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
  today: 'Today',
  neverText: '—',

  // Zone 2 — bounded recent-run feed
  tabAll: 'All',
  tabEval: 'Eval',
  tabPerf: 'Perf',
  searchPlaceholder: 'Search model or dataset',
  samples: 'samples',
  runs: 'runs',
  datasets: 'datasets',

  // Shared
  noReportsHint: 'Enter an output directory and click Scan to discover reports',
  welcomeTitle: 'Welcome to EvalScope',
  welcomeDesc: 'Enter an output directory path and scan to get started',
}

export const zh: Dict = {
  totalEvaluations: '评测总数',
  totalPerfRuns: '性能压测',
  modelsEvaluated: '已评估模型',
  latestRun: '最近运行',
  today: '今天',
  neverText: '—',

  tabAll: '全部',
  tabEval: '评测',
  tabPerf: '压测',
  searchPlaceholder: '搜索模型或数据集',
  samples: '条样本',
  runs: '次运行',
  datasets: '个数据集',

  noReportsHint: '输入输出目录路径并点击扫描来发现报告',
  welcomeTitle: '欢迎使用 EvalScope',
  welcomeDesc: '输入输出目录路径并扫描即可开始',
}
