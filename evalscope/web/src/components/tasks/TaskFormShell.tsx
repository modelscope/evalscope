import type { ReactNode } from 'react'
import { ChevronDown, ChevronUp } from 'lucide-react'
import Button from '@/components/ui/Button'
import Card from '@/components/ui/Card'
import { cn } from '@/lib/utils'

interface TaskFormShellProps {
  onSubmit: (event: React.SyntheticEvent<HTMLFormElement>) => void
  /** Always-visible fields, laid out in a 1/2-column responsive grid. */
  children: ReactNode
  /** Fields behind the "more parameters" disclosure; omit when there are none. */
  moreParams?: ReactNode
  /** Column count of the more-parameters grid at `md` and up. */
  moreParamsColumns?: 2 | 3
  moreParamsLabel: string
  showMore: boolean
  onToggleMore: () => void
  submitLabel: string
  disabled?: boolean
}

/**
 * Chrome shared by the task configuration forms.
 *
 * Owns the layout and the disclosure, not the fields: the primary grid, the
 * "more parameters" toggle and the submit button are identical across the Eval
 * and Performance forms, while the fields inside them are not. Validation and
 * focus behaviour live in `useTaskForm`.
 *
 * `noValidate` is deliberate — the forms validate through the shared validators
 * so every message resolves through the locale system rather than the browser's
 * built-in, unlocalized bubbles.
 */
export default function TaskFormShell({
  onSubmit,
  children,
  moreParams,
  moreParamsColumns = 3,
  moreParamsLabel,
  showMore,
  onToggleMore,
  submitLabel,
  disabled,
}: TaskFormShellProps) {
  return (
    <form onSubmit={onSubmit} className="space-y-4" noValidate>
      <div className="grid grid-cols-1 gap-4 md:grid-cols-2">{children}</div>

      {moreParams && (
        <>
          <button
            type="button"
            onClick={onToggleMore}
            className="flex cursor-pointer items-center gap-1 text-xs text-[var(--accent)] hover:underline"
          >
            {moreParamsLabel}
            {showMore ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
          </button>

          {showMore && (
            <Card className="!p-0">
              <div
                className={cn(
                  'grid grid-cols-1 gap-4 p-4',
                  moreParamsColumns === 3 ? 'md:grid-cols-3' : 'md:grid-cols-2',
                )}
              >
                {moreParams}
              </div>
            </Card>
          )}
        </>
      )}

      <Button type="submit" variant="primary" disabled={disabled} className="btn-glow">
        {submitLabel}
      </Button>
    </form>
  )
}
