import type { Dict } from './types'

export const en: Dict = {
  modelGenerate: 'Model Generate',
  toolCall: 'Tool Call',
  toolResult: 'Tool Result',
  envExec: 'Env Exec',
  error: 'Error',
  submit: 'Submit',
  step: 'Step',
  toolCallsCount: '${n} tool calls',
  stdout: 'stdout',
  arguments: 'Arguments',
  expandDetails: 'Show details',
  nudge: 'System Nudge',
  runStart: 'Run Start',
  runEnd: 'Run End',
  loopMessage: {
    max_steps_exceeded: 'Maximum steps exceeded',
    model_context_overflow: 'Model context window exceeded',
    no_tool_call_reminder: 'No tool was called — system reminder injected',
  },
}

export const zh: Dict = {
  modelGenerate: '模型生成',
  toolCall: '工具调用',
  toolResult: '工具结果',
  envExec: '环境执行',
  error: '错误',
  submit: '提交',
  step: '步骤',
  toolCallsCount: '${n} 个工具调用',
  stdout: '输出',
  arguments: '参数',
  expandDetails: '展开详情',
  nudge: '系统提醒',
  runStart: '运行开始',
  runEnd: '运行结束',
  loopMessage: {
    max_steps_exceeded: '达到最大步数上限',
    model_context_overflow: '超出模型上下文窗口',
    no_tool_call_reminder: '未调用工具——系统已注入提醒',
  },
}
