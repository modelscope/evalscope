import { useSearchParams } from 'react-router-dom'
import { useCallback } from 'react'

export function useQueryParams() {
  const [searchParams, setSearchParams] = useSearchParams()

  const get = useCallback((key: string) => searchParams.get(key) ?? undefined, [searchParams])

  const set = useCallback(
    (key: string, value: string | undefined) => {
      setSearchParams((prev) => {
        const next = new URLSearchParams(prev)
        if (value === undefined) {
          next.delete(key)
        } else {
          next.set(key, value)
        }
        return next
      })
    },
    [setSearchParams],
  )

  const getAll = useCallback(() => Object.fromEntries(searchParams.entries()), [searchParams])

  return { get, set, getAll }
}
