import type { Dict } from './types'

export const en: Dict = {
  title: 'Performance Testing',
  config: 'Performance Config',
  apiType: 'API Type',
  parallel: 'Concurrency (comma-separated)',
  number: 'Total Requests (comma-separated)',
  rate: 'Rate Limit (req/s)',
  maxTokens: 'Max Tokens',
  minTokens: 'Min Tokens',
  dataset: 'Test Dataset',
  maxPromptLen: 'Max Prompt Length',
  minPromptLen: 'Min Prompt Length',
  startPerf: 'Start Performance Test',
  status: 'Status & Logs',
  ready: 'Status: Ready',
}

export const zh: Dict = {
  title: '性能测试',
  config: '性能测试配置',
  apiType: 'API类型',
  parallel: '并发数 (逗号分隔)',
  number: '总请求数 (逗号分隔)',
  rate: '速率限制 (请求/秒)',
  maxTokens: 'Max Tokens',
  minTokens: 'Min Tokens',
  dataset: '测试数据集',
  maxPromptLen: '最大Prompt长度',
  minPromptLen: '最小Prompt长度',
  startPerf: '开始性能测试',
  status: '运行状态与日志',
  ready: '当前状态: 准备就绪',
}
