import { useTheme } from '@/contexts/ThemeContext'
import ChartFrame from '@/components/charts/ChartFrame'
import type { DataTableModel } from '@/components/common/DataTableFallback'

interface PlotlyChartProps {
  src: string
  fallbackTable: DataTableModel
  height?: number
  className?: string
  title?: string
}

/**
 * Compatibility wrapper for existing Plotly call sites.
 *
 * All charts now flow through ChartFrame, which owns theme injection, request
 * preflight, timeout/error handling, retry, and the authoritative table
 * fallback. Keeping this wrapper avoids parallel chart implementations while
 * allowing feature pages to migrate without changing their visual API.
 */
export default function PlotlyChart({ src, fallbackTable, height, className, title }: PlotlyChartProps) {
  const { theme } = useTheme()
  return (
    <ChartFrame
      baseSrc={src}
      theme={theme}
      fallbackTable={fallbackTable}
      height={height}
      className={className}
      title={title}
    />
  )
}
