import { useState, useCallback } from 'react'
import { cn } from '@/lib/utils'
import Skeleton from '@/components/ui/Skeleton'
import { AlertTriangle } from 'lucide-react'

interface PlotlyChartProps {
  src: string
  height?: number
  className?: string
  title?: string
}

export default function PlotlyChart({ src, height = 400, className, title }: PlotlyChartProps) {
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(false)

  const handleLoad = useCallback(() => {
    setLoading(false)
    setError(false)
  }, [])

  const handleError = useCallback(() => {
    setLoading(false)
    setError(true)
  }, [])

  return (
    <div
      className={cn(
        'rounded-[var(--radius)] border border-[var(--border)] bg-[var(--bg-card)] overflow-hidden',
        className,
      )}
    >
      {title && (
        <div className="px-4 py-2.5 border-b border-[var(--border)]">
          <h4 className="text-xs font-semibold uppercase tracking-wider text-[var(--text-muted)]">
            {title}
          </h4>
        </div>
      )}
      <div className="relative" style={{ height }}>
        {loading && !error && (
          <div className="absolute inset-0 flex items-center justify-center p-6">
            <Skeleton width="100%" height="100%" />
          </div>
        )}
        {error && (
          <div className="absolute inset-0 flex flex-col items-center justify-center gap-2 text-[var(--text-dim)]">
            <AlertTriangle size={24} />
            <span className="text-sm">Failed to load chart</span>
          </div>
        )}
        <iframe
          src={src}
          className={cn(
            'w-full h-full border-0 transition-opacity duration-200',
            (loading || error) && 'opacity-0',
          )}
          sandbox="allow-scripts allow-same-origin"
          onLoad={handleLoad}
          onError={handleError}
          title={title ?? 'Chart'}
        />
      </div>
    </div>
  )
}
