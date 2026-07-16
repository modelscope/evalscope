import type { PredictionRow } from '@/api/types'

export type ChatPresentation = 'traced' | 'structured' | 'legacy'

type ChatPresentationInput = Pick<PredictionRow, 'Messages' | 'AgentTrace'>

/** Select the established prediction presentation without coupling it to React. */
export function selectChatPresentation(prediction: ChatPresentationInput): ChatPresentation {
  const hasMessages = Boolean(prediction.Messages?.length)
  const hasTrace = Boolean(prediction.AgentTrace?.events?.length)

  if (hasMessages && hasTrace) return 'traced'
  if (hasMessages) return 'structured'
  return 'legacy'
}
