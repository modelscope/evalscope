import { useCallback, useState } from 'react'
import { useLocale } from '@/contexts/LocaleContext'

export function useFormErrors() {
  const { t } = useLocale()
  const [errors, setErrors] = useState<Record<string, string>>({})

  const errorFor = useCallback(
    (id: string): string | undefined => (errors[id] ? t(errors[id]) : undefined),
    [errors, t],
  )

  const clearError = useCallback((id: string) => {
    setErrors((previous) => {
      if (!previous[id]) return previous
      const next = { ...previous }
      delete next[id]
      return next
    })
  }, [])

  return { setErrors, errorFor, clearError }
}
