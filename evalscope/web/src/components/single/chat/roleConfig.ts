import React from 'react'
import { User, Bot, Shield, Wrench } from 'lucide-react'

export type Role = 'system' | 'user' | 'assistant' | 'tool'

export interface RolePalette {
  icon: React.FC<{ size?: number; style?: React.CSSProperties; className?: string }>
  barColor: string
  tintBg: string
  tintBgHl: string
  borderHl: string
  labelColor: string
  label: string
}

export function rolePalette(role: Role, t: (k: string) => string): RolePalette {
  switch (role) {
    case 'user':
      return {
        icon: User,
        barColor: 'var(--bubble-user-color)',
        tintBg: 'var(--bubble-user-bg)',
        tintBgHl: 'var(--bubble-user-bg-hl)',
        borderHl: 'var(--bubble-user-border-hl)',
        labelColor: 'var(--bubble-user-color)',
        label: 'User',
      }
    case 'assistant':
      return {
        icon: Bot,
        barColor: 'var(--bubble-bot-color)',
        tintBg: 'transparent',
        tintBgHl: 'var(--bubble-bot-bg-hl)',
        borderHl: 'var(--bubble-bot-border-hl)',
        labelColor: 'var(--bubble-bot-color)',
        label: 'Assistant',
      }
    case 'tool':
      return {
        icon: Wrench,
        barColor: 'var(--bubble-tool-color)',
        tintBg: 'var(--bubble-tool-bg)',
        tintBgHl: 'var(--bubble-tool-bg)',
        borderHl: 'var(--bubble-tool-border)',
        labelColor: 'var(--bubble-tool-color)',
        label: t('prediction.toolResult'),
      }
    case 'system':
    default:
      return {
        icon: Shield,
        barColor: 'var(--text-muted)',
        tintBg: 'var(--bubble-system-bg)',
        tintBgHl: 'var(--bubble-system-bg)',
        borderHl: 'var(--bubble-system-border)',
        labelColor: 'var(--text-muted)',
        label: t('prediction.systemPrompt'),
      }
  }
}
