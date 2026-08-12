import { useCallback, useState } from 'react'

/** Everything a view needs to drive a confirmed batch deletion. */
export interface BatchDelete {
  /** True while the deletion loop is running. */
  deleting: boolean
  /** True while the confirmation dialog should be open. */
  confirmOpen: boolean
  /** Failure message from the last attempt, or `null`. */
  error: string | null
  /** Ask for confirmation. Ignored when nothing is selected or a run is active. */
  request: () => void
  /** Dismiss the confirmation without deleting. */
  cancel: () => void
  /** Run the deletion for the current selection. */
  confirm: () => Promise<void>
}

interface BatchDeleteOptions<T> {
  /** Items to delete, in the order they should be attempted. */
  items: readonly T[]
  /** Deletes one item. Rejecting stops the loop and keeps the remaining items. */
  deleteItem: (item: T) => Promise<unknown>
  /**
   * Replaces the selection with whatever is left.
   *
   * Called with the items that were *not* deleted, so a partial failure leaves
   * the user holding exactly the ones still on the server rather than an empty
   * tray that hides the failure.
   */
  onSettled: (remaining: T[]) => void
  /** Re-reads the list once the attempt finishes, successfully or not. */
  reload: () => void
  /** Renders the failure message for a rejection. */
  formatError: (message: string) => string
}

/**
 * Confirmed batch deletion with partial-failure recovery.
 *
 * Deletion is sequential rather than concurrent: the server is asked to remove
 * one run at a time so a mid-way rejection leaves a known state — everything
 * before the failure is gone, everything from it onwards is untouched — instead
 * of a set of independent outcomes the view cannot describe to the user.
 *
 * The selection is reconciled from what actually succeeded, so the tray keeps the
 * runs that are still there and the failure stays visible next to them. The list
 * is always re-read, because a partial deletion changed the server even though
 * the attempt failed.
 */
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
