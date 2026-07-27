import { useEffect, useId, useRef } from 'react'
import { createPortal } from 'react-dom'
import { AlertTriangle } from 'lucide-react'
import Button from '@/components/ui/Button'

interface ConfirmDialogProps {
  open: boolean
  title: string
  /** Supporting message rendered under the title. */
  message?: string
  /** Optional list of affected items, rendered as a scrollable summary. */
  items?: string[]
  confirmLabel: string
  cancelLabel: string
  /** Styles the confirm button as destructive (danger palette). */
  danger?: boolean
  /** Disables both actions and shows the busy label while work is in flight. */
  busy?: boolean
  onConfirm: () => void
  onCancel: () => void
}

/**
 * In-app modal confirmation for destructive or consequential actions.
 *
 * Replaces `window.confirm` so the dialog matches the dashboard theme, can
 * enumerate the affected items, and cannot be suppressed by the browser.
 * Focus starts on the cancel button (safe default for destructive actions),
 * Tab cycles inside the dialog, and Escape / overlay click cancel.
 */
export default function ConfirmDialog({
  open,
  title,
  message,
  items,
  confirmLabel,
  cancelLabel,
  danger = false,
  busy = false,
  onConfirm,
  onCancel,
}: ConfirmDialogProps) {
  const titleId = useId()
  const messageId = useId()
  const panelRef = useRef<HTMLDivElement>(null)
  const cancelRef = useRef<HTMLButtonElement>(null)

  // Initial focus on the cancel button; restore the previous focus on close.
  useEffect(() => {
    if (!open) return
    const previous = document.activeElement as HTMLElement | null
    cancelRef.current?.focus()
    return () => previous?.focus?.()
  }, [open])

  // Escape cancels (unless busy); Tab is trapped inside the dialog.
  useEffect(() => {
    if (!open) return
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && !busy) {
        e.stopPropagation()
        onCancel()
        return
      }
      if (e.key !== 'Tab' || !panelRef.current) return
      const focusable = panelRef.current.querySelectorAll<HTMLElement>('button:not([disabled])')
      if (focusable.length === 0) return
      const first = focusable[0]
      const last = focusable[focusable.length - 1]
      if (e.shiftKey && document.activeElement === first) {
        e.preventDefault()
        last.focus()
      } else if (!e.shiftKey && document.activeElement === last) {
        e.preventDefault()
        first.focus()
      }
    }
    document.addEventListener('keydown', handleKeyDown, true)
    return () => document.removeEventListener('keydown', handleKeyDown, true)
  }, [open, busy, onCancel])

  if (!open) return null

  return createPortal(
    <div
      className="fixed inset-0 z-[100] flex items-center justify-center p-4"
      style={{ background: 'rgba(0,0,0,0.6)', backdropFilter: 'blur(4px)' }}
      onClick={() => !busy && onCancel()}
    >
      <div
        ref={panelRef}
        role="alertdialog"
        aria-modal="true"
        aria-labelledby={titleId}
        aria-describedby={message ? messageId : undefined}
        className="w-full max-w-md rounded-[var(--radius-lg)] border border-[var(--border)] bg-[var(--bg-card)] shadow-2xl"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-start gap-3 p-5 pb-3">
          {danger && (
            <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-[var(--radius-sm)] bg-[var(--danger-bg)]">
              <AlertTriangle size={20} className="text-[var(--danger)]" />
            </div>
          )}
          <div className="min-w-0 flex-1">
            <h2 id={titleId} className="text-base font-semibold text-[var(--text)]">
              {title}
            </h2>
            {message && (
              <p id={messageId} className="mt-1 text-sm text-[var(--text-muted)]">
                {message}
              </p>
            )}
          </div>
        </div>

        {items && items.length > 0 && (
          <ul className="mx-5 max-h-40 overflow-y-auto rounded-[var(--radius-sm)] border border-[var(--border)] bg-[var(--bg-deep)] px-3 py-2">
            {items.map((item) => (
              <li key={item} className="truncate py-0.5 font-mono text-xs text-[var(--text-muted)]">
                {item}
              </li>
            ))}
          </ul>
        )}

        <div className="flex items-center justify-end gap-2 p-5 pt-4">
          <Button ref={cancelRef} variant="outline" size="sm" disabled={busy} onClick={onCancel}>
            {cancelLabel}
          </Button>
          <Button
            variant="primary"
            size="sm"
            disabled={busy}
            onClick={onConfirm}
            className={
              danger
                ? 'bg-[var(--danger)] hover:bg-[var(--danger)] hover:shadow-none hover:opacity-90'
                : undefined
            }
          >
            {confirmLabel}
          </Button>
        </div>
      </div>
    </div>,
    document.body,
  )
}
