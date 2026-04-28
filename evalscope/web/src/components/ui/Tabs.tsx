import { cn } from '@/lib/utils'

interface Tab {
  key: string
  label: string
}

interface TabsProps {
  tabs: Tab[]
  activeKey: string
  onChange: (key: string) => void
  className?: string
}

export default function Tabs({ tabs, activeKey, onChange, className }: TabsProps) {
  return (
    <div
      className={cn(
        'inline-flex items-center gap-1 p-1 rounded-[var(--radius)] bg-[var(--bg-deep)] border border-[var(--border)]',
        className,
      )}
    >
      {tabs.map((tab) => (
        <button
          key={tab.key}
          onClick={() => onChange(tab.key)}
          className={cn(
            'px-4 py-1.5 text-sm font-medium rounded-[var(--radius-sm)] transition-all duration-[var(--transition)] cursor-pointer',
            activeKey === tab.key
              ? 'bg-[var(--accent)] text-white shadow-[0_0_12px_rgba(129,109,248,0.2)]'
              : 'text-[var(--text-muted)] hover:text-[var(--text)] bg-[var(--bg-card)] hover:bg-[var(--bg-card2)]',
          )}
        >
          {tab.label}
        </button>
      ))}
    </div>
  )
}
