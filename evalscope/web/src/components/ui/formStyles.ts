import { cn } from '@/lib/utils'

export const FORM_INPUT_CLASS =
  'w-full px-3 py-2 text-sm rounded-[var(--radius-sm)] bg-[var(--bg-deep)] border border-[var(--border)] text-[var(--text)] placeholder:text-[var(--text-dim)] focus:outline-none focus:border-[var(--accent)] focus:ring-1 focus:ring-[var(--accent-dim)] transition-all'

export const FORM_LABEL_CLASS = 'block text-xs font-medium text-[var(--text-muted)] mb-1'

export const FORM_ERROR_INPUT_CLASS =
  'border-[var(--danger)] focus:border-[var(--danger)] focus:ring-[var(--danger-bg)]'

/** Compose the input class with optional error state. */
export function inputClass(error?: string | null, extra?: string) {
  return cn(FORM_INPUT_CLASS, error && FORM_ERROR_INPUT_CLASS, extra)
}
