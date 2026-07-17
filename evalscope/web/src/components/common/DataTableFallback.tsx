import { Table } from 'lucide-react'
import DataTable from '@/components/common/DataTable'
import { useLocale } from '@/contexts/LocaleContext'

/**
 * Authoritative data-table representation of a chart's underlying data.
 *
 * A {@link DataTableModel} is the single source the chart and its table
 * fallback both render, so the fallback always presents the *same* data the
 * chart would have visualised.
 */
export interface DataTableModel {
  /** Ordered column headers. */
  columns: string[]
  /** Row records keyed by column name. */
  rows: Record<string, unknown>[]
  /** Columns that should be rendered with score styling/formatting. */
  scoreColumns?: string[]
}

interface DataTableFallbackProps {
  /** Chart data to present as an accessible table. */
  model: DataTableModel
  className?: string
}

/**
 * Renders the authoritative data-table fallback for a chart.
 *
 * This is shown whenever the chart itself cannot be rendered so users still
 * have access to the exact same underlying data. It reuses the shared
 * {@link DataTable} primitive and adds a localized caption/hint.
 */
export default function DataTableFallback({ model, className }: DataTableFallbackProps) {
  const { t } = useLocale()

  return (
    <div className={className}>
      <div className="mb-2 flex items-center gap-1.5 text-[var(--text-muted)]">
        <Table size={14} aria-hidden="true" />
        <span className="type-label-xs">{t('charts.fallbackTitle')}</span>
      </div>
      <p className="mb-2 text-xs text-[var(--text-muted)]">{t('charts.fallbackHint')}</p>
      <DataTable columns={model.columns} data={model.rows} scoreColumns={model.scoreColumns} />
    </div>
  )
}
