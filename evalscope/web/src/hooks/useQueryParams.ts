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

  /** Read every value of a repeated query parameter (e.g. `?report=a&report=b`). */
  const getList = useCallback((key: string) => searchParams.getAll(key), [searchParams])

  /** Replace every value of a repeated query parameter with the given list. */
  const setList = useCallback(
    (key: string, values: string[]) => {
      setSearchParams((prev) => {
        const next = new URLSearchParams(prev)
        next.delete(key)
        for (const value of values) next.append(key, value)
        return next
      })
    },
    [setSearchParams],
  )

  return { get, set, getList, setList }
}
