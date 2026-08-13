import { cn } from '@/lib/utils'

interface ScoreThresholdInputProps {
  id: string
  value: number
  onChange: (value: number) => void
  label: string
  className?: string
}

/**
 * The view-only score threshold used by the prediction surfaces.
 *
 * The threshold splits the sample list into above / below for browsing; it is
 * never a pass/fail verdict and never leaves the view, which is why it is a plain
 * control rather than part of a form's validated payload.
 */
export default function ScoreThresholdInput({
  id,
  value,
  onChange,
  label,
  className,
}: ScoreThresholdInputProps) {
  return (
    <div className={cn('flex flex-col gap-1.5', className)}>
      <label htmlFor={id} className="type-label-xs whitespace-nowrap">
        {label}
      </label>
      <input
        id={id}
        name={id}
        type="number"
        value={value}
        step={0.01}
        min={0}
        max={1}
        onChange={(event) => onChange(Number(event.target.value))}
        className="w-24 rounded-[var(--radius-sm)] border border-[var(--border)] bg-[var(--bg-deep)] px-3 py-2 text-sm text-[var(--text)] focus:border-[var(--accent)] focus:outline-none"
      />
    </div>
  )
}
