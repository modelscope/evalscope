import { useCallback, useEffect, useRef, useState } from 'react'

interface UsePollingOptions<T> {
  fn: () => Promise<T>
  interval?: number
  enabled?: boolean
  onData?: (data: T) => void
}

export function usePolling<T>({ fn, interval = 5000, enabled = false, onData }: UsePollingOptions<T>) {
  const [data, setData] = useState<T | null>(null)
  const [error, setError] = useState<string | null>(null)
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const mountedRef = useRef(true)

  // Use refs so that changes to fn / onData do NOT restart the polling cycle.
  // Written in an effect (not during render) to satisfy react-hooks/refs.
  const fnRef = useRef(fn)
  const onDataRef = useRef(onData)
  useEffect(() => {
    fnRef.current = fn
    onDataRef.current = onData
  })

  const poll = useCallback(async () => {
    try {
      const result = await fnRef.current()
      if (!mountedRef.current) return
      setData(result)
      setError(null)
      onDataRef.current?.(result)
    } catch (e) {
      if (!mountedRef.current) return
      setError(e instanceof Error ? e.message : String(e))
    }
  }, [])

  useEffect(() => {
    mountedRef.current = true
    if (!enabled) {
      if (timerRef.current) clearTimeout(timerRef.current)
      return
    }
    let cancelled = false
    const tick = async () => {
      await poll()
      if (!cancelled && mountedRef.current) {
        timerRef.current = setTimeout(tick, interval)
      }
    }
    tick()
    return () => {
      cancelled = true
      mountedRef.current = false
      if (timerRef.current) clearTimeout(timerRef.current)
    }
  }, [enabled, interval, poll])

  return { data, error }
}
