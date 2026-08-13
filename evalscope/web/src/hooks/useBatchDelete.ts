import { useCallback, useState } from 'react'

export interface BatchDelete {
  deleting: boolean
  confirmOpen: boolean
  error: string | null
  request: () => void
  cancel: () => void
  confirm: () => Promise<void>
}

interface BatchDeleteOptions<T> {
  items: readonly T[]
  deleteItem: (item: T) => Promise<unknown>
  onSettled: (remaining: T[]) => void
  reload: () => void
  formatError: (message: string) => string
}

/** Confirmed sequential deletion that preserves unprocessed items after a partial failure. */
export function useBatchDelete<T>({
  items,
  deleteItem,
  onSettled,
  reload,
  formatError,
}: BatchDeleteOptions<T>): BatchDelete {
  const [deleting, setDeleting] = useState(false)
  const [confirmOpen, setConfirmOpen] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const request = useCallback(() => {
    if (items.length === 0 || deleting) return
    setConfirmOpen(true)
  }, [items.length, deleting])

  const cancel = useCallback(() => setConfirmOpen(false), [])

  const confirm = useCallback(async () => {
    if (deleting || items.length === 0) return
    setDeleting(true)
    setError(null)
    const deleted = new Set<T>()
    try {
      for (const item of items) {
        await deleteItem(item)
        deleted.add(item)
      }
      onSettled([])
    } catch (err) {
      onSettled(items.filter((item) => !deleted.has(item)))
      setError(formatError(err instanceof Error ? err.message : String(err)))
    } finally {
      setDeleting(false)
      setConfirmOpen(false)
      reload()
    }
  }, [deleting, items, deleteItem, onSettled, reload, formatError])

  return { deleting, confirmOpen, error, request, cancel, confirm }
}
