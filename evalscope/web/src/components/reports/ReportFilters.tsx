import { useCallback, useRef, useState } from 'react'
import { ArrowUpDown, ChevronDown } from 'lucide-react'
import { cn } from '@/lib/utils'
import { useLocale } from '@/contexts/LocaleContext'
import SearchInput from '@/components/ui/SearchInput'
import FilterChip from '@/components/ui/FilterChip'
import Button from '@/components/ui/Button'

export interface ReportFilters {
  search: string
  models: string[]
  datasets: string[]
  scoreMin: number
  scoreMax: number
  sortBy: 'score' | 'model' | 'dataset' | 'time'
  sortOrder: 'asc' | 'desc'
}

interface ReportFiltersProps {
  filters: ReportFilters
  availableModels: string[]
  availableDatasets: string[]
  onChange: (filters: ReportFilters) => void
}

// Multi-select dropdown with checkboxes
function MultiSelectDropdown({
  label,
  options,
  selected,
  onChange,
}: {
  label: string
  options: string[]
  selected: string[]
  onChange: (selected: string[]) => void
}) {
  const [open, setOpen] = useState(false)
  const ref = useRef<HTMLDivElement>(null)

  const toggle = (val: string) => {
    if (selected.includes(val)) onChange(selected.filter((s) => s !== val))
    else onChange([...selected, val])
  }

  return (
    <div ref={ref} className="relative">
      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        className={cn(
          'flex items-center gap-1.5 px-3 py-2 text-sm rounded-[var(--radius-sm)]',
          'bg-[var(--bg-deep)] border border-[var(--border)] text-[var(--text)]',
          'hover:border-[var(--border-md)] transition-all duration-[var(--transition)]',
          'cursor-pointer',
          selected.length > 0 && 'border-[var(--accent-dim)]',
        )}
      >
        <span className="truncate max-w-[120px]">
          {selected.length > 0 ? `${label} (${selected.length})` : label}
        </span>
        <ChevronDown size={14} className={cn('text-[var(--text-dim)] transition-transform', open && 'rotate-180')} />
      </button>
      {open && (
        <>
          <div className="fixed inset-0 z-10" onClick={() => setOpen(false)} />
          <div className="absolute z-20 top-full mt-1 left-0 min-w-[200px] max-h-[240px] overflow-y-auto rounded-[var(--radius)] bg-[var(--bg-card)] border border-[var(--border)] shadow-[var(--shadow-lg)] py-1">
            {options.length === 0 ? (
              <div className="px-3 py-2 text-xs text-[var(--text-dim)]">—</div>
            ) : (
              options.map((opt) => (
                <label
                  key={opt}
                  className="flex items-center gap-2 px-3 py-1.5 text-sm text-[var(--text)] hover:bg-[var(--bg-card2)] cursor-pointer transition-colors"
                >
                  <input
                    type="checkbox"
                    checked={selected.includes(opt)}
                    onChange={() => toggle(opt)}
                    className="accent-[var(--accent)] w-3.5 h-3.5"
                  />
                  <span className="truncate">{opt}</span>
                </label>
              ))
            )}
          </div>
        </>
      )}
    </div>
  )
}

export default function ReportFiltersBar({
  filters,
  availableModels,
  availableDatasets,
  onChange,
}: ReportFiltersProps) {
  const { t } = useLocale()

  const update = useCallback(
    (patch: Partial<ReportFilters>) => onChange({ ...filters, ...patch }),
    [filters, onChange],
  )

  const sortOptions: { value: ReportFilters['sortBy']; label: string }[] = [
    { value: 'time', label: t('reports.filters.time') },
    { value: 'score', label: t('reports.filters.score') },
    { value: 'model', label: t('reports.filters.model') },
    { value: 'dataset', label: t('reports.filters.dataset') },
  ]

  const activeFilters: { key: string; label: string; onRemove: () => void }[] = []
  filters.models.forEach((m) =>
    activeFilters.push({
      key: `model:${m}`,
      label: `model:${m}`,
      onRemove: () => update({ models: filters.models.filter((x) => x !== m) }),
    }),
  )
  filters.datasets.forEach((d) =>
    activeFilters.push({
      key: `dataset:${d}`,
      label: `dataset:${d}`,
      onRemove: () => update({ datasets: filters.datasets.filter((x) => x !== d) }),
    }),
  )
  if (filters.scoreMin > 0)
    activeFilters.push({
      key: 'scoreMin',
      label: `score≥${filters.scoreMin}`,
      onRemove: () => update({ scoreMin: 0 }),
    })
  if (filters.scoreMax < 1)
    activeFilters.push({
      key: 'scoreMax',
      label: `score≤${filters.scoreMax}`,
      onRemove: () => update({ scoreMax: 1 }),
    })

  return (
    <div className="flex flex-col gap-2">
      {/* Filter row */}
      <div className="flex flex-wrap items-center gap-2">
        <SearchInput
          value={filters.search}
          onChange={(v) => update({ search: v })}
          placeholder={t('reports.filters.search')}
          className="w-[220px]"
        />

        <MultiSelectDropdown
          label={t('reports.filters.model')}
          options={availableModels}
          selected={filters.models}
          onChange={(models) => update({ models })}
        />

        <MultiSelectDropdown
          label={t('reports.filters.dataset')}
          options={availableDatasets}
          selected={filters.datasets}
          onChange={(datasets) => update({ datasets })}
        />

        {/* Score range */}
        <div className="flex items-center gap-1 text-sm">
          <span className="text-[var(--text-muted)] text-xs">{t('reports.filters.score')}:</span>
          <input
            type="number"
            min={0}
            max={1}
            step={0.01}
            value={filters.scoreMin}
            onChange={(e) => update({ scoreMin: parseFloat(e.target.value) || 0 })}
            className="w-[60px] px-2 py-1.5 text-xs rounded-[var(--radius-sm)] bg-[var(--bg-deep)] border border-[var(--border)] text-[var(--text)] focus:outline-none focus:border-[var(--accent)]"
          />
          <span className="text-[var(--text-dim)]">—</span>
          <input
            type="number"
            min={0}
            max={1}
            step={0.01}
            value={filters.scoreMax}
            onChange={(e) => update({ scoreMax: parseFloat(e.target.value) || 1 })}
            className="w-[60px] px-2 py-1.5 text-xs rounded-[var(--radius-sm)] bg-[var(--bg-deep)] border border-[var(--border)] text-[var(--text)] focus:outline-none focus:border-[var(--accent)]"
          />
        </div>

        {/* Sort */}
        <div className="flex items-center gap-1 ml-auto">
          <span className="text-[var(--text-muted)] text-xs">{t('reports.filters.sortBy')}:</span>
          <select
            value={filters.sortBy}
            onChange={(e) => update({ sortBy: e.target.value as ReportFilters['sortBy'] })}
            className="appearance-none px-2 py-1.5 pr-6 text-xs rounded-[var(--radius-sm)] bg-[var(--bg-deep)] border border-[var(--border)] text-[var(--text)] focus:outline-none focus:border-[var(--accent)] cursor-pointer"
          >
            {sortOptions.map((o) => (
              <option key={o.value} value={o.value}>
                {o.label}
              </option>
            ))}
          </select>
          <Button
            variant="ghost"
            size="sm"
            onClick={() => update({ sortOrder: filters.sortOrder === 'asc' ? 'desc' : 'asc' })}
            className="!px-1.5"
            title={filters.sortOrder === 'asc' ? 'Ascending' : 'Descending'}
          >
            <ArrowUpDown size={14} className={filters.sortOrder === 'desc' ? 'rotate-180' : ''} />
          </Button>
        </div>
      </div>

      {/* Active filter chips */}
      {activeFilters.length > 0 && (
        <div className="flex flex-wrap items-center gap-1.5">
          {activeFilters.map((f) => (
            <FilterChip key={f.key} label={f.label} onRemove={f.onRemove} />
          ))}
        </div>
      )}
    </div>
  )
}
