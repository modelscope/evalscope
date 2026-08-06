import { scoreBg } from '@/utils/colorScale'
import { formatMetric, getBoundedQualityRatio } from '@/domain/metric'
import type { MetricSemantics } from '@/domain/metric'

interface Props {
  columns: string[]
  data: Record<string, unknown>[]
  scoreColumns?: string[]
  /** Backend semantics of the score columns, used for formatting, colour scale and sorting. */
  semantics?: MetricSemantics | null
}

export default function DataTable({ columns, data, scoreColumns = [], semantics }: Props) {
  if (!data.length) return null
  const scoreCols = new Set(scoreColumns.length ? scoreColumns : columns.filter((c) => c.toLowerCase().includes('score')))

  return (
    <div className="overflow-x-auto rounded-lg border border-[var(--border)]">
      <table className="w-full text-sm">
        <thead>
          <tr className="bg-[var(--bg-card)]">
            {columns.map((col) => (
              <th key={col} className="px-3 py-2 text-left font-medium text-[var(--text-muted)] whitespace-nowrap">
                {col}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {data.map((row, i) => (
            <tr key={i} className="border-t border-[var(--border)] hover:bg-[var(--bg-card2)]">
              {columns.map((col) => {
                const val = row[col]
                const isScore = scoreCols.has(col) && typeof val === 'number'
                // Only a bounded quality metric gets a colour scale; a diagnostic or an
                // unbounded one would imply a verdict it does not carry.
                const ratio = isScore ? getBoundedQualityRatio(val as number, semantics) : null
                return (
                  <td
                    key={col}
                    className="px-3 py-1.5 whitespace-nowrap"
                    style={ratio === null ? undefined : { backgroundColor: scoreBg(ratio) }}
                  >
                    {isScore ? formatMetric(val as number, semantics).primary : String(val ?? '')}
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
