import type { CSSProperties, ReactNode } from 'react'
import { cn } from '@/lib/utils'

export type BubbleRole = 'user' | 'bot' | 'tool' | 'reasoning' | 'system'

interface ChatBubbleProps {
  role: BubbleRole
  children: ReactNode
  /**
   * Highlight (hover / selected) state — swaps `bg` / `border` to the `*-hl` variants.
   * Pass a stable value (e.g. on `.is-active` or a selected-id match) — atom does not own hover.
   */
  highlighted?: boolean
  /**
   * Visual variant:
   *   - 'bar'  — left vertical bar accent (the default EvalScope chat row look)
   *   - 'card' — full bordered container per DESIGN.md `{components.chat-bubble}` spec
   * Defaults to 'bar' to preserve in-place visuals; opt into 'card' for the new spec.
   */
  variant?: 'bar' | 'card'
  className?: string
  style?: CSSProperties
}

const ROLE_TOKENS: Record<BubbleRole, { bg: string; bgHl: string; border: string; borderHl: string }> = {
  user: {
    bg: 'var(--bubble-user-bg)',
    bgHl: 'var(--bubble-user-bg-hl)',
    border: 'var(--bubble-user-border)',
    borderHl: 'var(--bubble-user-border-hl)',
  },
  bot: {
    bg: 'var(--bubble-bot-bg)',
    bgHl: 'var(--bubble-bot-bg-hl)',
    border: 'var(--bubble-bot-border)',
    borderHl: 'var(--bubble-bot-border-hl)',
  },
  tool: {
    bg: 'var(--bubble-tool-bg)',
    bgHl: 'var(--bubble-tool-bg-hl)',
    border: 'var(--bubble-tool-border)',
    borderHl: 'var(--bubble-tool-border-hl)',
  },
  reasoning: {
    bg: 'var(--bubble-reasoning-bg)',
    bgHl: 'var(--bubble-reasoning-bg-hl)',
    border: 'var(--bubble-reasoning-border)',
    borderHl: 'var(--bubble-reasoning-border-hl)',
  },
  system: {
    bg: 'var(--bubble-system-bg)',
    bgHl: 'var(--bubble-system-bg-hl)',
    border: 'var(--bubble-system-border)',
    borderHl: 'var(--bubble-system-border-hl)',
  },
}

const ROLE_ACCENT: Record<BubbleRole, string> = {
  user: 'var(--bubble-user-color)',
  bot: 'var(--bubble-bot-color)',
  tool: 'var(--bubble-tool-color)',
  reasoning: 'var(--bubble-reasoning-color)',
  system: 'var(--bubble-system-color)',
}

/**
 * Chat bubble — DESIGN.md `{components.chat-bubble}`.
 * Encapsulates the 5 role × 7 token bubble system so chat-domain code never reaches into
 * `--bubble-*` tokens directly. The `style` prop is forwarded for legitimate dynamic needs
 * (e.g. animation delays); do not pass `background` / `border` overrides through it.
 */
export default function ChatBubble({
  role,
  children,
  highlighted = false,
  variant = 'bar',
  className,
  style,
}: ChatBubbleProps) {
  const tokens = ROLE_TOKENS[role]
  const accent = ROLE_ACCENT[role]

  if (variant === 'card') {
    return (
      <div
        className={cn('rounded-[var(--radius)] border transition-colors', className)}
        style={{
          background: highlighted ? tokens.bgHl : tokens.bg,
          borderColor: highlighted ? tokens.borderHl : tokens.border,
          ...style,
        }}
      >
        {children}
      </div>
    )
  }

  return (
    <div
      className={cn('rounded-[var(--radius-sm)] transition-colors', className)}
      style={{
        background: highlighted ? tokens.bgHl : tokens.bg,
        borderLeft: `3px solid ${highlighted ? tokens.borderHl : accent}`,
        ...style,
      }}
    >
      {children}
    </div>
  )
}

/** Helper for chat-domain code that still needs the role's accent color (icon stroke, label text). */
export function bubbleAccent(role: BubbleRole): string {
  return ROLE_ACCENT[role]
}

/** Helper for chat-domain code that needs the role's border token (decorative side bars). */
export function bubbleBorder(role: BubbleRole, highlighted = false): string {
  return highlighted ? ROLE_TOKENS[role].borderHl : ROLE_TOKENS[role].border
}
