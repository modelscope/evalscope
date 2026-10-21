import type { Dict } from './types'

export const en: Dict = {
  evalTab: 'Evaluation',
  perfTab: 'Performance',
  alias: 'Alias (optional)',
  activeTitle: 'Running Tasks',
  activeAdopt: 'Monitor',
  activeKindEval: 'Evaluation',
  activeKindPerf: 'Performance',
  failedWithLogHint: 'Task failed. See the task log for details.',
  progressUnreachable:
    'Progress updates are failing — the task may have finished or been lost. Stop clears this panel.',
}

export const zh: Dict = {
  evalTab: '模型评估',
  perfTab: '性能测试',
  alias: '别名（可选）',
  activeTitle: '正在运行的任务',
  activeAdopt: '监控',
  activeKindEval: '评估',
  activeKindPerf: '性能',
  failedWithLogHint: '任务失败，详情见任务日志。',
  progressUnreachable: '进度更新持续失败，任务可能已结束或丢失。点击停止清除此面板。',
}
