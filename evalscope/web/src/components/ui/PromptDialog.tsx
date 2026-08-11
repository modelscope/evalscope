import { useEffect, useId, useRef } from 'react'
import { createPortal } from 'react-dom'
import Button from '@/components/ui/Button'
import { inputClass } from '@/components/ui/formStyles'

interface PromptDialogProps {
  open: boolean
  title: string
  message?: string
  label: string
  value: string
  error?: string | null
  confirmLabel: string
  cancelLabel: string
  busy?: boolean
  onValueChange: (value: string) => void
  onConfirm: () => void
  onCancel: () => void
}

/** A single-input confirmation dialog (e.g. "rename this report"). */
export default function PromptDialog({
  open,
  title,
  message,
  label,
  value,
  error,
  confirmLabel,
  cancelLabel,
  busy = false,
  onValueChange,
  onConfirm,
  onCancel,
}: PromptDialogProps) {
  const titleId = useId()
  const messageId = useId()
  const inputId = useId()
  const errorId = useId()
  const panelRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLInputElement>(null)

  useEffect(() => {
    if (!open) return
    const previous = document.activeElement as HTMLElement | null
    inputRef.current?.focus()
    inputRef.current?.select()
    return () => previous?.focus?.()
  }, [open])

  useEffect(() => {
    if (!open) return
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && !busy) {
        e.stopPropagation()
        onCancel()
        return
      }
      if (e.key === 'Enter' && !busy) {
        e.preventDefault()
        onConfirm()
        return
      }
      if (e.key !== 'Tab' || !panelRef.current) return
      const focusable = panelRef.current.querySelectorAll<HTMLElement>('input:not([disabled]), button:not([disabled])')
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
  }, [open, busy, onCancel, onConfirm])

  if (!open) return null

  return createPortal(
    <div
      className="fixed inset-0 z-[100] flex items-center justify-center p-4"
      style={{ background: 'rgba(0,0,0,0.6)', backdropFilter: 'blur(4px)' }}
      onClick={() => !busy && onCancel()}
    >
      <div
        ref={panelRef}
        role="dialog"
        aria-modal="true"
        aria-labelledby={titleId}
        aria-describedby={message ? messageId : undefined}
        className="w-full max-w-md rounded-[var(--radius-lg)] border border-[var(--border)] bg-[var(--bg-card)] shadow-2xl"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="p-5 pb-3">
          <h2 id={titleId} className="text-base font-semibold text-[var(--text)]">
            {title}
          </h2>
          {message && (
            <p id={messageId} className="mt-1 text-sm text-[var(--text-muted)]">
              {message}
            </p>
          )}
        </div>

        <div className="px-5 pb-2">
          <label htmlFor={inputId} className="mb-1 block text-xs font-medium text-[var(--text-muted)]">
            {label}
          </label>
          <input
            ref={inputRef}
            id={inputId}
            type="text"
            value={value}
            disabled={busy}
            aria-invalid={Boolean(error)}
            aria-describedby={error ? errorId : undefined}
            onChange={(e) => onValueChange(e.target.value)}
            className={inputClass(error)}
          />
          {error && (
            <p id={errorId} className="mt-1 text-xs text-[var(--danger)]">
              {error}
            </p>
          )}
        </div>

        <div className="flex items-center justify-end gap-2 p-5 pt-4">
          <Button variant="outline" size="sm" disabled={busy} onClick={onCancel}>
            {cancelLabel}
          </Button>
          <Button variant="primary" size="sm" disabled={busy} onClick={onConfirm}>
            {confirmLabel}
          </Button>
        </div>
      </div>
    </div>,
    document.body,
  )
}
