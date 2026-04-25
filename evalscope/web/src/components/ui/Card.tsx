import { useState, type ReactNode } from 'react'
import { cn } from '@/lib/utils'
import { ChevronDown } from 'lucide-react'

interface CardProps {
  children: ReactNode
  className?: string
  title?: string
  badge?: ReactNode
  collapsible?: boolean
}

export default function Card({ children, className, title, badge, collapsible }: CardProps) {
  const [collapsed, setCollapsed] = useState(false)

  return (
    <div
      className={cn(
        'rounded-[var(--radius)] border border-[var(--border)] bg-[var(--bg-card)] shadow-[var(--shadow-sm)]',
        className,
      )}
    >
      {title && (
        <div
          className={cn(
            'flex items-center justify-between px-5 py-3 border-b border-[var(--border)]',
            collapsible && 'cursor-pointer select-none',
          )}
          onClick={collapsible ? () => setCollapsed((c) => !c) : undefined}
        >
          <div className="flex items-center gap-2">
            <h3 className="text-xs font-semibold uppercase tracking-wider text-[var(--text-muted)]">
              {title}
            </h3>
            {badge}
          </div>
          {collapsible && (
            <ChevronDown
              size={14}
              className={cn(
                'text-[var(--text-dim)] transition-transform duration-200',
                collapsed && '-rotate-90',
              )}
            />
          )}
        </div>
      )}
      {!collapsed && <div className="p-5">{children}</div>}
    </div>
  )
}
