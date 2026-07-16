import type { ReactNode } from 'react'
import { useLocale } from '@/contexts/LocaleContext'
import { FORM_LABEL_CLASS } from './formStyles'

/**
 * ARIA props produced by {@link Field} and applied by the caller to the actual
 * form control (input / select / textarea). Passing these through a render prop
 * keeps the primitive control-agnostic while guaranteeing a programmatic,
 * non-empty accessible name and correct error association.
 */
export interface FieldAriaProps {
  /** Stable control id, matches the label's `htmlFor` (Req 10.2). */
  id: string
  /** Stable control name for form submission / autofill (Req 10.8). */
  name: string
  /** References the visible label so the control has a non-empty name (Req 10.2). */
  'aria-labelledby': string
  /** True while the field has a validation error (Req 10.3). */
  'aria-invalid': boolean
  /** References the error element when an error is present (Req 10.3). */
  'aria-describedby'?: string
  /** Autocomplete hint, including API key scenarios (Req 10.8, 10.12). */
  autoComplete?: string
}

export interface FieldProps {
  /** Stable, unique field id used to derive label / error element ids. */
  id: string
  /** Locale key for the visible label; rendered through the locale system (Req 10.11). */
  labelKey: string
  /** Marks the field as required and renders a visual indicator. */
  required?: boolean
  /**
   * Already-localized error message. When present the control is marked
   * `aria-invalid` and associated with the error element via `aria-describedby`
   * (Req 10.3). Error text must be resolved through the locale system by the
   * caller (Req 10.10).
   */
  error?: string
  /** Stable control name (Req 10.8). */
  name: string
  /** Autocomplete hint forwarded to the control (Req 10.8, 10.12). */
  autoComplete?: string
  /** Extra class names for the field wrapper. */
  className?: string
  /** Render prop receiving the ARIA props to spread onto the control. */
  children: (ariaProps: FieldAriaProps) => ReactNode
}

/**
 * Semantic form field primitive. It renders a localized `<label>` that is
 * programmatically linked to the control it wraps (via `htmlFor`/`id` plus
 * `aria-labelledby`), so every field exposes a non-empty accessible name
 * (Req 10.2). When an error is provided the control is marked `aria-invalid`
 * and associated with a live error region through `aria-describedby` (Req 10.3).
 *
 * The control itself is supplied by the caller through a render prop, allowing
 * inputs, selects, textareas or the Dataset_Combobox to share the same
 * accessibility contract. Actual error announcement orchestration lives in the
 * form layer; this primitive only reserves the live region container and the
 * programmatic association.
 */
export default function Field({
  id,
  labelKey,
  required,
  error,
  name,
  autoComplete,
  className,
  children,
}: FieldProps) {
  const { t } = useLocale()

  const labelId = `${id}-label`
  const errorId = `${id}-error`

  const ariaProps: FieldAriaProps = {
    id,
    name,
    'aria-labelledby': labelId,
    'aria-invalid': Boolean(error),
    // Only associate the error element when an error is actually present.
    ...(error ? { 'aria-describedby': errorId } : {}),
    ...(autoComplete ? { autoComplete } : {}),
  }

  return (
    <div className={className}>
      <label id={labelId} htmlFor={id} className={FORM_LABEL_CLASS}>
        {t(labelKey)}
        {required && <span className="text-[var(--danger)]"> *</span>}
      </label>
      {children(ariaProps)}
      {/*
        Error container doubles as a live region so the form layer can announce
        validation errors within 1s (Req 10.3). It is always rendered (empty
        when there is no error) to keep the live region stable for assistive
        technology; announcement orchestration is wired in the form layer.
      */}
      <p
        id={errorId}
        role="alert"
        aria-live="polite"
        className="text-xs text-[var(--danger)] mt-1 empty:hidden"
      >
        {error}
      </p>
    </div>
  )
}
