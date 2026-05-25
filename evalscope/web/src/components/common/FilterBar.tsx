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
              ? 'bg-[var(--accent)] text-white'
              : 'bg-[var(--bg-card)] text-[var(--text-muted)] hover:bg-[var(--bg-card2)] border border-[var(--border)]'
          }`}
        >
          {m}
        </button>
      ))}
    </div>
  )
}
