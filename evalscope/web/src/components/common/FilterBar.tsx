interface Props {
  modes: string[]
  active: string
  onChange: (m: string) => void
}

export default function FilterBar({ modes, active, onChange }: Props) {
  return (
    <div className="flex flex-wrap gap-1">
      {modes.map((m) => (
        <button
          key={m}
          onClick={() => onChange(m)}
          className={`px-3 py-1 text-xs rounded-full transition-colors ${
            active === m
              ? 'bg-[var(--color-primary)] text-white'
              : 'bg-[var(--color-surface)] text-[var(--color-ink-muted)] hover:bg-[var(--color-surface-hover)] border border-[var(--color-border)]'
          }`}
        >
          {m}
        </button>
      ))}
    </div>
  )
}
