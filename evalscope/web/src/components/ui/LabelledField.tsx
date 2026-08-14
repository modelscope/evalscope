import { cn } from '@/lib/utils'

interface LabelledFieldProps {
  label: string
  value: string
  /** Numeric values line up in a column when rendered with tabular figures. */
  numeric?: boolean
  className?: string
}

/**
 * One individually-labelled read-only field, laid out label-then-value on a
 * shared baseline.
 *
 * Each fact gets its own field rather than being joined into one string, so
 * `Provider` and `Protocol` — or the three workload parameters — stay separately
 * readable instead of collapsing into a single ambiguous line. The value wraps
 * rather than truncating, because these are identifiers the user may need in full.
 */
export default function LabelledField({ label, value, numeric, className }: LabelledFieldProps) {
  return (
    <div className={cn('flex min-w-0 items-baseline gap-1.5', className)}>
      <span className="type-table-xs">{label}</span>
      <span className={cn('type-body-sm break-words text-[var(--text)]', numeric && 'tabular-nums')}>
        {value}
      </span>
    </div>
  )
}
