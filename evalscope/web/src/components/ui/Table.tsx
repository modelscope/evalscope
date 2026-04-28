import { type ReactNode, useState, useMemo } from 'react'
import { ChevronUp, ChevronDown, ChevronsUpDown } from 'lucide-react'
import { cn } from '@/lib/utils'

type SortDir = 'asc' | 'desc'

interface Column<T> {
  key: string
  label: string
  render?: (row: T, index: number) => ReactNode
  /** Enable column sorting. Provide a comparator or leave true for default (string/number). */
  sortable?: boolean | ((a: T, b: T) => number)
}

interface TableProps<T> {
  columns: Column<T>[]
  data: T[]
  onRowClick?: (row: T, index: number) => void
  className?: string
  /** Default sort: { key, dir } */
  defaultSort?: { key: string; dir: SortDir }
}

/** Convert a 0–1 score to an HSL color string (red → yellow → green) */
export function scoreColor(score: number): string {
  const hue = Math.round(score * 120) // 0=red, 60=yellow, 120=green
  return `hsl(${hue}, 70%, 45%)`
}

export default function Table<T extends Record<string, unknown>>({
  columns,
  data,
  onRowClick,
  className,
  defaultSort,
}: TableProps<T>) {
  const [sortKey, setSortKey] = useState<string | null>(defaultSort?.key ?? null)
  const [sortDir, setSortDir] = useState<SortDir>(defaultSort?.dir ?? 'desc')

  const handleHeaderClick = (col: Column<T>) => {
    if (!col.sortable) return
    if (sortKey === col.key) {
      setSortDir((d) => (d === 'asc' ? 'desc' : 'asc'))
    } else {
      setSortKey(col.key)
      setSortDir('desc')
    }
  }

  const sortedData = useMemo(() => {
    if (!sortKey) return data
    const col = columns.find((c) => c.key === sortKey)
    if (!col?.sortable) return data

    const cmp: (a: T, b: T) => number =
      typeof col.sortable === 'function'
        ? col.sortable
        : (a, b) => {
            const av = a[sortKey]
            const bv = b[sortKey]
            if (typeof av === 'number' && typeof bv === 'number') return av - bv
            return String(av ?? '').localeCompare(String(bv ?? ''), undefined, { numeric: true })
          }

    const sorted = [...data].sort(cmp)
    return sortDir === 'asc' ? sorted : sorted.reverse()
  }, [data, sortKey, sortDir, columns])

  return (
    <div
      className={cn(
        'overflow-x-auto rounded-[var(--radius)] border border-[var(--border)] bg-[var(--bg-card)]',
        className,
      )}
    >
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-[var(--border)]">
            {columns.map((col) => {
              const isActive = sortKey === col.key
              const isSortable = !!col.sortable
              return (
                <th
                  key={col.key}
                  onClick={() => handleHeaderClick(col)}
                  className={cn(
                    'px-4 py-3 text-left text-[10px] font-semibold uppercase tracking-wider text-[var(--text-dim)] select-none',
                    isSortable && 'cursor-pointer hover:text-[var(--text)]',
                    isActive && 'text-[var(--accent)]',
                  )}
                >
                  <span className="inline-flex items-center gap-1">
                    {col.label}
                    {isSortable && (
                      <span className="opacity-60" style={{ flexShrink: 0 }}>
                        {isActive ? (
                          sortDir === 'asc' ? <ChevronUp size={12} /> : <ChevronDown size={12} />
                        ) : (
                          <ChevronsUpDown size={12} />
                        )}
                      </span>
                    )}
                  </span>
                </th>
              )
            })}
          </tr>
        </thead>
        <tbody>
          {sortedData.map((row, i) => (
            <tr
              key={i}
              onClick={onRowClick ? () => onRowClick(row, i) : undefined}
              className={cn(
                'border-b border-[var(--border)] last:border-b-0 transition-colors duration-150',
                onRowClick && 'cursor-pointer hover:bg-[var(--bg-card2)]',
              )}
            >
              {columns.map((col) => (
                <td key={col.key} className="px-4 py-3 text-[var(--text)]">
                  {col.render ? col.render(row, i) : (row[col.key] as ReactNode)}
                </td>
              ))}
            </tr>
          ))}
          {data.length === 0 && (
            <tr>
              <td
                colSpan={columns.length}
                className="px-4 py-8 text-center text-[var(--text-dim)]"
              >
                No data
              </td>
            </tr>
          )}
        </tbody>
      </table>
    </div>
  )
}
