import { scoreBg } from '@/utils/colorScale'
import { useLocale } from '@/contexts/LocaleContext'
import { formatMetricByKey } from '@/domain/metric/registry'

interface Props {
  columns: string[]
  data: Record<string, unknown>[]
  scoreColumns?: string[]
}

export default function DataTable({ columns, data, scoreColumns = [] }: Props) {
  const { t } = useLocale()
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
                return (
                  <td
                    key={col}
                    className="px-3 py-1.5 whitespace-nowrap"
                    style={isScore ? { backgroundColor: scoreBg(val as number) } : undefined}
                  >
                    {isScore ? formatMetricByKey('score', val as number, t).primary : String(val ?? '')}
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
