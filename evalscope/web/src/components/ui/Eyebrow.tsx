import type { ElementType, ReactNode } from 'react'
import { cn } from '@/lib/utils'

interface EyebrowProps {
  children: ReactNode
  as?: ElementType
  className?: string
}

/**
 * Section eyebrow — the brand's signature uppercase micro-label.
 * Maps to DESIGN.md `{typography.label-xs}` — 12px / semibold / uppercase / tracking-wider / text-muted.
 * Use for card section headers, form labels, group titles. Never for body or display.
 */
export default function Eyebrow({ children, as: Tag = 'h3', className }: EyebrowProps) {
  return <Tag className={cn('type-label-xs', className)}>{children}</Tag>
}
