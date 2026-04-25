import { type ReactNode } from 'react'
import { cn } from '@/lib/utils'

interface Column<T> {
  key: string
  label: string
  render?: (row: T, index: number) => ReactNode
}

interface TableProps<T> {
  columns: Column<T>[]
  data: T[]
  onRowClick?: (row: T, index: number) => void
  className?: string
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
}: TableProps<T>) {
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
            {columns.map((col) => (
              <th
                key={col.key}
                className="px-4 py-3 text-left text-[10px] font-semibold uppercase tracking-wider text-[var(--text-dim)]"
              >
                {col.label}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {data.map((row, i) => (
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
