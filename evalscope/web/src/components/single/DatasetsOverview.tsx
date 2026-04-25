import { useEffect, useMemo, useState } from 'react'
import { useReports } from '@/contexts/ReportsContext'
import { useLocale } from '@/contexts/LocaleContext'
import type { ReportData } from '@/api/types'
import { getChartUrl } from '@/api/reports'
import ChartEmbed from '@/components/charts/ChartEmbed'
import RadarChart from '@/components/charts/RadarChart'
import DataTable from '@/components/common/DataTable'
import EmptyState from '@/components/common/EmptyState'

interface Props {
  reports: ReportData[]
  reportName: string
}

export default function DatasetsOverview({ reports, reportName }: Props) {
  const { t } = useLocale()
  const { rootPath } = useReports()
  const [tableData, setTableData] = useState<{ columns: string[]; data: Record<string, unknown>[] }>({
    columns: [],
    data: [],
  })

  useEffect(() => {
    if (!reports.length) return
    const rows: Record<string, unknown>[] = []
    for (const r of reports) {
      rows.push({
        Model: r.model_name,
        Dataset: r.dataset_name,
        Score: r.score,
        Num: r.metrics[0]?.categories?.reduce((s, c) => s + c.num, 0) ?? 0,
      })
    }
    setTableData({ columns: ['Model', 'Dataset', 'Score', 'Num'], data: rows })
  }, [reports])

  // Build radar data: one point per dataset (aggregate by dataset_name)
  const radarData = useMemo(() => {
    const byDs: Record<string, number[]> = {}
    for (const r of reports) {
      if (!byDs[r.dataset_name]) byDs[r.dataset_name] = []
      byDs[r.dataset_name].push(r.score)
    }
    return Object.entries(byDs).map(([name, scores]) => {
      const avg = scores.reduce((a, b) => a + b, 0) / scores.length
      // Normalize: scores > 1 are likely percentages, convert to 0-1
      const norm = avg > 1 ? avg / 100 : avg
      return { label: name, value: norm, rawValue: avg }
    })
  }, [reports])

  if (!reports.length || !reportName) return <EmptyState />

  return (
    <div className="flex flex-col gap-6">
      {/* Radar chart - front-end rendered */}
      {radarData.length > 0 && (
        <section>
          <h4 className="text-sm font-medium mb-3 text-[var(--color-ink-muted)]">{t('single.radarChart')}</h4>
          <div className="glass-card rounded-xl p-4 flex justify-center">
            <RadarChart data={radarData} size={360} />
          </div>
        </section>
      )}

      <section>
        <h4 className="text-sm font-medium mb-2 text-[var(--color-ink-muted)]">Dataset Components</h4>
        <div className="bg-[var(--color-surface)] rounded-lg p-2 border border-[var(--color-border)]">
          <ChartEmbed src={getChartUrl(rootPath, 'sunburst', { reportName })} height={500} />
        </div>
      </section>
      <section>
        <h4 className="text-sm font-medium mb-2 text-[var(--color-ink-muted)]">Dataset Scores</h4>
        <div className="bg-[var(--color-surface)] rounded-lg p-2 border border-[var(--color-border)]">
          <ChartEmbed src={getChartUrl(rootPath, 'scores', { reportName })} height={350} />
        </div>
      </section>
      <section>
        <h4 className="text-sm font-medium mb-2 text-[var(--color-ink-muted)]">Scores Table</h4>
        <DataTable columns={tableData.columns} data={tableData.data} scoreColumns={['Score']} />
      </section>
    </div>
  )
}
