import { cn } from '@/lib/utils'

interface Props {
  datasets: string[]
  active: string
  onChange: (ds: string) => void
}

export default function DatasetNav({ datasets, active, onChange }: Props) {
  return (
    <nav
      className="flex flex-col gap-0.5 py-3 pr-3"
      style={{
        width: 200,
        flexShrink: 0,
        borderRight: '1px solid var(--border)',
        position: 'sticky',
        top: 0,
        alignSelf: 'flex-start',
        maxHeight: '80vh',
        overflowY: 'auto',
      }}
    >
      <div className="px-3 pb-2 text-[10px] font-semibold uppercase tracking-wider text-[var(--text-dim)]">
        Datasets
      </div>
      {datasets.map((ds) => {
        const isActive = ds === active
        return (
          <button
            key={ds}
            onClick={() => onChange(ds)}
            className={cn(
              'flex items-center gap-2 px-3 py-2 text-sm rounded-[var(--radius-sm)] text-left transition-all duration-150 cursor-pointer',
              isActive
                ? 'bg-[var(--accent-dim)] text-[var(--accent)] font-medium'
                : 'text-[var(--text-muted)] hover:bg-[var(--bg-card2)] hover:text-[var(--text)]',
            )}
          >
            <span
              className={cn(
                'w-1.5 h-1.5 rounded-full flex-shrink-0 transition-all',
                isActive ? 'bg-[var(--accent)] shadow-[0_0_6px_var(--accent)]' : 'bg-[var(--text-dim)]',
              )}
            />
            <span className="truncate">{ds}</span>
          </button>
        )
      })}
    </nav>
  )
}
