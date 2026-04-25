import { useMemo } from 'react'
import { useReports } from '@/contexts/ReportsContext'
import { useLocale } from '@/contexts/LocaleContext'
import type { ReportData } from '@/api/types'
import { getChartUrl } from '@/api/reports'
import ChartEmbed from '@/components/charts/ChartEmbed'
import GroupedBarChart from '@/components/charts/GroupedBarChart'
import DataTable from '@/components/common/DataTable'
import EmptyState from '@/components/common/EmptyState'
import { parseReportName } from '@/utils/reportParser'

interface Props {
  reports: ReportData[]
}

export default function ModelsOverview({ reports }: Props) {
  const { t } = useLocale()
  const { rootPath, selectedReports } = useReports()

  const { tableColumns, tableData, groupedChartData } = useMemo(() => {
    if (!reports.length) return { tableColumns: [], tableData: [], groupedChartData: { datasets: [], models: [] } }

    // Group by model
    const byModel: Record<string, Record<string, number>> = {}
    for (const r of reports) {
      if (!byModel[r.model_name]) byModel[r.model_name] = {}
      byModel[r.model_name][r.dataset_name] = r.score
    }

    const modelNames = Object.keys(byModel)
    // Find common datasets
    const dsLists = modelNames.map((m) => new Set(Object.keys(byModel[m])))
    const common = [...dsLists.reduce((a, b) => new Set([...a].filter((x) => b.has(x))))]
    common.sort()

    // Pivot table: model x dataset
    const tableColumns = ['Model', ...common]
    const tableData = modelNames.map((name) => {
      const row: Record<string, unknown> = { Model: name }
      for (const ds of common) row[ds] = byModel[name][ds] ?? 0
      return row
    })

    // Grouped bar chart data: use common datasets, short model labels
    const groupedModels = modelNames.map((name) => ({
      name: parseReportName(name).model || name,
      scores: Object.fromEntries(common.map((ds) => [ds, byModel[name][ds] ?? 0])),
    }))

    return {
      tableColumns,
      tableData,
      groupedChartData: { datasets: common, models: groupedModels },
    }
  }, [reports])

  if (!reports.length) return <EmptyState />

  return (
    <div className="flex flex-col gap-6">
      {/* Frontend-rendered grouped bar chart */}
      {groupedChartData.datasets.length > 0 && (
        <section>
          <h4 className="text-sm font-medium mb-3 text-[var(--color-ink-muted)]">{t('multi.groupedBarChart')}</h4>
          <div className="glass-card rounded-xl p-4">
            <GroupedBarChart
              datasets={groupedChartData.datasets}
              models={groupedChartData.models}
              height={300}
            />
          </div>
        </section>
      )}

      <section>
        <h4 className="text-sm font-medium mb-2 text-[var(--color-ink-muted)]">Model Comparison Radar</h4>
        <div className="bg-[var(--color-surface)] rounded-lg p-2 border border-[var(--color-border)]">
          <ChartEmbed
            src={getChartUrl(rootPath, 'radar', { reportNames: selectedReports })}
            height={450}
          />
        </div>
      </section>
      <section>
        <h4 className="text-sm font-medium mb-2 text-[var(--color-ink-muted)]">Comparison Scores</h4>
        <DataTable columns={tableColumns} data={tableData} scoreColumns={tableColumns.slice(1)} />
      </section>
    </div>
  )
}
