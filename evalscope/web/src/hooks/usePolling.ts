import { useCallback, useEffect, useRef, useState } from 'react'

interface UsePollingOptions<T> {
  fn: () => Promise<T>
  interval?: number
  enabled?: boolean
  onData?: (data: T) => void
  onError?: (message: string) => void
}

export function usePolling<T>({ fn, interval = 5000, enabled = false, onData, onError }: UsePollingOptions<T>) {
  const [data, setData] = useState<T | null>(null)
  const [failure, setFailure] = useState<Error | null>(null)
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const mountedRef = useRef(true)
  const requestRef = useRef(0)

  // Use refs so that changes to fn / onData do NOT restart the polling cycle.
  // Written in an effect (not during render) to satisfy react-hooks/refs.
  const fnRef = useRef(fn)
  const onDataRef = useRef(onData)
  const onErrorRef = useRef(onError)
  useEffect(() => {
    fnRef.current = fn
    onDataRef.current = onData
    onErrorRef.current = onError
  })

  const poll = useCallback(async () => {
    const requestId = ++requestRef.current
    try {
      const result = await fnRef.current()
      if (!mountedRef.current || requestId !== requestRef.current) return
      setData(result)
      setFailure(null)
      onDataRef.current?.(result)
    } catch (e) {
      if (!mountedRef.current || requestId !== requestRef.current) return
      const message = e instanceof Error ? e.message : String(e)
      setFailure(e instanceof Error ? e : new Error(message))
      onErrorRef.current?.(message)
    }
  }, [])

  useEffect(() => {
    requestRef.current += 1
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

  return { data, error: failure?.message ?? null, failure, refresh: poll }
}
