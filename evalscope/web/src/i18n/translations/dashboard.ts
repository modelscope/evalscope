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

  // Zone 3 — bounded recent-run feed
  tabAll: 'All',
  tabEval: 'Eval',
  tabPerf: 'Perf',
  searchPlaceholder: 'Search model or dataset',
  samples: 'samples',
  runs: 'runs',
  datasets: 'datasets',

  // Zone 2 — scoped evaluation history
  trendTitle: 'Evaluation Trend',
  trendDescription: 'Compares the primary metric only across runs of the same model and benchmark.',
  trendModel: 'Trend model',
  trendBenchmark: 'Trend benchmark',
  trendEmpty: 'Run the same model and benchmark at least twice to see a trend.',
  trendRunCount: '${count} runs',
  trendPrevious: '${delta} vs previous',

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

  trendTitle: '评测趋势',
  trendDescription: '仅比较同一模型、同一基准多次运行的主指标。',
  trendModel: '趋势模型',
  trendBenchmark: '趋势基准',
  trendEmpty: '同一模型和基准至少运行两次后才会显示趋势。',
  trendRunCount: '${count} 次运行',
  trendPrevious: '较上次 ${delta}',

  noReportsHint: '输入输出目录路径并点击扫描来发现报告',
  welcomeTitle: '欢迎使用 EvalScope',
  welcomeDesc: '输入输出目录路径并扫描即可开始',
}
