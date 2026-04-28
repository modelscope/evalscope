import { cn } from '@/lib/utils'

interface SkeletonProps {
  width?: string | number
  height?: string | number
  rounded?: boolean
  lines?: number
  className?: string
}

export default function Skeleton({
  width,
  height,
  rounded,
  lines,
  className,
}: SkeletonProps) {
  if (lines && lines > 1) {
    return (
      <div className={cn('flex flex-col gap-2', className)}>
        {Array.from({ length: lines }).map((_, i) => (
          <div
            key={i}
            className="skeleton-pulse rounded-[var(--radius-xs)] bg-[var(--bg-card2)]"
            style={{
              width: i === lines - 1 ? '60%' : '100%',
              height: height ?? 14,
            }}
          />
        ))}
      </div>
    )
  }

  return (
    <div
      className={cn(
        'skeleton-pulse bg-[var(--bg-card2)]',
        rounded ? 'rounded-full' : 'rounded-[var(--radius-sm)]',
        className,
      )}
      style={{
        width: width ?? '100%',
        height: height ?? 20,
      }}
    />
  )
}
