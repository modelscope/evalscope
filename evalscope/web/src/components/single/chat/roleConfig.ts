import React from 'react'
import { User, Bot, Shield, Wrench } from 'lucide-react'
import { bubbleAccent, type BubbleRole } from '@/components/ui/ChatBubble'

export type Role = 'system' | 'user' | 'assistant' | 'tool'

export interface RolePalette {
  icon: React.FC<{ size?: number; style?: React.CSSProperties; className?: string }>
  /** Accent color for the icon stroke and label text. */
  labelColor: string
  label: string
}

/** Map the chat-domain `Role` to the canonical `BubbleRole` token family in ChatBubble. */
export function roleToBubble(role: Role): BubbleRole {
  switch (role) {
    case 'assistant': return 'bot'
    case 'user': return 'user'
    case 'tool': return 'tool'
    case 'system':
    default: return 'system'
  }
}

export function rolePalette(role: Role, t: (k: string) => string): RolePalette {
  switch (role) {
    case 'user':
      return { icon: User, labelColor: bubbleAccent('user'), label: 'User' }
    case 'assistant':
      return { icon: Bot, labelColor: bubbleAccent('bot'), label: 'Assistant' }
    case 'tool':
      return { icon: Wrench, labelColor: bubbleAccent('tool'), label: t('prediction.toolResult') }
    case 'system':
    default:
      return { icon: Shield, labelColor: 'var(--text-muted)', label: t('prediction.systemPrompt') }
  }
}
