import { useMemo } from 'react'
import { cn } from '@/lib/utils'

interface HeatmapCell {
  model: string
  dataset: string
  score: number | null
  reportName?: string
}

interface ScoreHeatmapProps {
  data: HeatmapCell[]
  onCellClick?: (cell: HeatmapCell) => void
  className?: string
}

/** Convert a 0–1 score to an HSL color string (red → yellow → green) */
function scoreColor(score: number): string {
  const hue = Math.round(score * 120) // 0=red, 60=yellow, 120=green
  return `hsl(${hue}, 70%, 45%)`
}

export default function ScoreHeatmap({ data, onCellClick, className }: ScoreHeatmapProps) {
  const { models, datasets, cellMap } = useMemo(() => {
    const modelSet = new Set<string>()
    const datasetSet = new Set<string>()
    const map = new Map<string, HeatmapCell>()

    for (const cell of data) {
      modelSet.add(cell.model)
      datasetSet.add(cell.dataset)
      map.set(`${cell.model}::${cell.dataset}`, cell)
    }

    return {
      models: Array.from(modelSet),
      datasets: Array.from(datasetSet),
      cellMap: map,
    }
  }, [data])

  if (models.length === 0 || datasets.length === 0) {
    return (
      <div
        className={cn(
          'flex items-center justify-center py-12 text-[var(--text-dim)] text-sm',
          className,
        )}
      >
        No data available
      </div>
    )
  }

  return (
    <div className={cn('overflow-x-auto', className)}>
      <table className="text-sm border-collapse">
        <thead>
          <tr>
            <th className="px-3 py-2 text-left text-[10px] font-semibold uppercase tracking-wider text-[var(--text-dim)] sticky left-0 bg-[var(--bg-card)] z-10">
              Model
            </th>
            {datasets.map((ds) => (
              <th
                key={ds}
                className="px-3 py-2 text-center text-[10px] font-semibold uppercase tracking-wider text-[var(--text-dim)]"
              >
                {ds}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {models.map((model) => (
            <tr key={model}>
              <td className="px-3 py-2 text-xs font-medium text-[var(--text-muted)] whitespace-nowrap sticky left-0 bg-[var(--bg-card)] z-10 border-r border-[var(--border)]">
                {model}
              </td>
              {datasets.map((ds) => {
                const cell = cellMap.get(`${model}::${ds}`)
                const score = cell?.score
                const hasScore = score != null

                return (
                  <td key={ds} className="px-1 py-1">
                    <button
                      onClick={hasScore && onCellClick ? () => onCellClick(cell!) : undefined}
                      disabled={!hasScore}
                      className={cn(
                        'w-full min-w-[56px] py-2 rounded-[var(--radius-xs)] text-xs font-mono font-medium text-center transition-all duration-150',
                        hasScore
                          ? 'text-white cursor-pointer hover:scale-105 hover:shadow-[var(--shadow-sm)]'
                          : 'text-[var(--text-dim)] bg-[var(--bg-deep)] cursor-default',
                      )}
                      style={
                        hasScore
                          ? { backgroundColor: scoreColor(score!) }
                          : undefined
                      }
                    >
                      {hasScore ? (score! * 100).toFixed(1) : '—'}
                    </button>
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
