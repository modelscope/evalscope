import { scoreBg } from '@/utils/colorScale'
import { formatScore } from '@/utils/formatUtils'

interface Props {
  columns: string[]
  data: Record<string, unknown>[]
  scoreColumns?: string[]
}

export default function DataTable({ columns, data, scoreColumns = [] }: Props) {
  if (!data.length) return null
  const scoreCols = new Set(scoreColumns.length ? scoreColumns : columns.filter((c) => c.toLowerCase().includes('score')))

  return (
    <div className="overflow-x-auto rounded-lg border border-[var(--color-border)]">
      <table className="w-full text-sm">
        <thead>
          <tr className="bg-[var(--color-surface)]">
            {columns.map((col) => (
              <th key={col} className="px-3 py-2 text-left font-medium text-[var(--color-ink-muted)] whitespace-nowrap">
                {col}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {data.map((row, i) => (
            <tr key={i} className="border-t border-[var(--color-border)] hover:bg-[var(--color-surface-hover)]">
              {columns.map((col) => {
                const val = row[col]
                const isScore = scoreCols.has(col) && typeof val === 'number'
                return (
                  <td
                    key={col}
                    className="px-3 py-1.5 whitespace-nowrap"
                    style={isScore ? { backgroundColor: scoreBg(val as number) } : undefined}
                  >
                    {isScore ? formatScore(val as number) : String(val ?? '')}
                  </td>
                )
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
