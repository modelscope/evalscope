import { Link } from 'react-router-dom'
import { ChevronRight } from 'lucide-react'
import { cn } from '@/lib/utils'

interface BreadcrumbItem {
  label: string
  href?: string
}

interface BreadcrumbProps {
  items: BreadcrumbItem[]
  className?: string
}

export default function Breadcrumb({ items, className }: BreadcrumbProps) {
  return (
    <nav className={cn('flex items-center gap-1.5 text-sm', className)}>
      {items.map((item, i) => {
        const isLast = i === items.length - 1
        return (
          <span key={i} className="flex items-center gap-1.5">
            {i > 0 && <ChevronRight size={12} className="text-[var(--text-dim)]" />}
            {item.href && !isLast ? (
              <Link
                to={item.href}
                className="text-[var(--text-muted)] hover:text-[var(--text)] transition-colors duration-150"
              >
                {item.label}
              </Link>
            ) : (
              <span className={isLast ? 'text-[var(--text)]' : 'text-[var(--text-muted)]'}>
                {item.label}
              </span>
            )}
          </span>
        )
      })}
    </nav>
  )
}
