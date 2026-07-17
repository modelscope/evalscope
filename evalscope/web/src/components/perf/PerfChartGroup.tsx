import Card from '@/components/ui/Card'
import Skeleton from '@/components/ui/Skeleton'
import PlotlyChart from '@/components/charts/PlotlyChart'
import type { DataTableModel } from '@/components/common/DataTableFallback'
import { CHART_TITLES } from '@/utils/perf'

interface PerfChartGroupProps {
  title: string
  charts: readonly string[]
  fallbackTable: DataTableModel
  getChartUrl: (chart: string) => string
  loading?: boolean
}

export default function PerfChartGroup({
  title,
  charts,
  fallbackTable,
  getChartUrl,
  loading = false,
}: PerfChartGroupProps) {
  return (
    <Card title={title}>
      {loading ? (
        <Skeleton width="100%" height={340} />
      ) : (
        <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
          {charts.map((chart) => (
            <PlotlyChart
              key={chart}
              src={getChartUrl(chart)}
              fallbackTable={fallbackTable}
              title={CHART_TITLES[chart]}
              height={340}
            />
          ))}
        </div>
      )}
    </Card>
  )
}
