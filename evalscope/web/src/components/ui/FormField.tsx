import type { ReactNode } from 'react'
import { FORM_LABEL_CLASS } from './formStyles'

interface FormFieldProps {
  label: string
  required?: boolean
  error?: string
  children: ReactNode
  className?: string
}

/** label + (children: control) + error message wrapper. */
export default function FormField({ label, required, error, children, className }: FormFieldProps) {
  return (
    <div className={className}>
      <label className={FORM_LABEL_CLASS}>
        {label}
        {required && <span className="text-[var(--danger)]"> *</span>}
      </label>
      {children}
      {error && <p className="text-xs text-[var(--danger)] mt-1">{error}</p>}
    </div>
  )
}
