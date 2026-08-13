import { useCallback, useId, useLayoutEffect, useRef, useState, type ReactNode } from 'react'
import { createPortal } from 'react-dom'
import { cn } from '@/lib/utils'

interface TooltipProps {
  /** The hint shown on hover / focus. */
  content: ReactNode
  /** The trigger; typically an icon. */
  children: ReactNode
  /** Accessible name for the trigger when its content is icon-only. */
  label?: string
  /** Max width of the bubble in px. */
  maxWidth?: number
  className?: string
}

/** Gap between the trigger and the bubble, in px. */
const OFFSET = 6
/** Keep the bubble this far from the viewport's right edge. */
const VIEWPORT_MARGIN = 12

/**
 * A declarative, instant tooltip.
 *
 * The native `title` attribute carries an unavoidable ~500 ms delay, and the
 * hand-rolled predecessor here wrote to `document.body` imperatively with a fixed
 * element id (so two instances collided) and no keyboard path. This renders the
 * bubble through a portal so it is never clipped by an `overflow` ancestor,
 * positions it with `position: fixed` clamped to the viewport, and shows it the
 * moment the trigger is hovered or focused — dismissing on leave, blur or Escape.
 *
 * The trigger is wrapped in a focusable span linked to the bubble via
 * `aria-describedby`, so assistive technology announces the hint and keyboard
 * users reach it without a pointer.
 */
export default function Tooltip({ content, children, label, maxWidth = 220, className }: TooltipProps) {
  const tooltipId = useId()
  const triggerRef = useRef<HTMLSpanElement>(null)
  const [open, setOpen] = useState(false)
  const [coords, setCoords] = useState<{ left: number; top: number }>({ left: 0, top: 0 })

  const place = useCallback(() => {
    const rect = triggerRef.current?.getBoundingClientRect()
    if (!rect) return
    setCoords({
      left: Math.max(VIEWPORT_MARGIN, Math.min(rect.left, window.innerWidth - maxWidth - VIEWPORT_MARGIN)),
      top: rect.bottom + OFFSET,
    })
  }, [maxWidth])

  // Position before paint so the bubble never flashes at the wrong spot.
  useLayoutEffect(() => {
    if (open) place()
  }, [open, place])

  const show = () => setOpen(true)
  const hide = () => setOpen(false)

  return (
    <span
      ref={triggerRef}
      className={cn('relative inline-flex', className)}
      tabIndex={0}
      aria-label={label}
      aria-describedby={open ? tooltipId : undefined}
      onMouseEnter={show}
      onMouseLeave={hide}
      onFocus={show}
      onBlur={hide}
      onKeyDown={(event) => { if (event.key === 'Escape') hide() }}
    >
      {children}
      {open && createPortal(
        <div
          id={tooltipId}
          role="tooltip"
          className="pointer-events-none fixed z-[9999] rounded-[var(--radius-sm)] border border-[var(--border)] bg-[var(--bg-card)] px-2.5 py-1.5 type-body-xs leading-normal text-[var(--text)] shadow-[var(--shadow)]"
          style={{ left: coords.left, top: coords.top, maxWidth }}
        >
          {content}
        </div>,
        document.body,
      )}
    </span>
  )
}
