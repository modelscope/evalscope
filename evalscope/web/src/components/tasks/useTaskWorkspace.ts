import { useEffect, useRef, useState } from 'react'
import { emptyWorkspace, readWorkspace, writeWorkspace, type Workspace } from './configStorage'

/** Owns browser persistence, not form validation or execution state. */
export function useTaskWorkspace(initialModel?: string, initialDataset?: string) {
  const [initial] = useState(() => {
    const result = readWorkspace()
    const workspace = result.ok ? result.value : emptyWorkspace()
    return {
      workspace: {
        ...workspace,
        connection: { ...workspace.connection, model: initialModel || workspace.connection.model },
        eval: { ...workspace.eval, datasets: initialDataset || workspace.eval.datasets },
      },
      error: result.ok ? '' : result.error,
      blocked: !result.ok && result.error === 'invalidStorage',
      restored: result.ok && Boolean(result.value.selectedConnection || result.value.eval.datasets),
    }
  })
  const [workspace, setWorkspace] = useState(initial.workspace)
  const [message, setMessage] = useState(initial.error || (initial.restored ? 'restored' : ''))
  const blocked = useRef(initial.blocked)
  const lastWritten = useRef<Workspace>(initial.workspace)
  const latest = useRef<Workspace>(initial.workspace)

  useEffect(() => {
    latest.current = workspace
  }, [workspace])
  useEffect(() => {
    const flush = () => {
      if (!blocked.current && latest.current !== lastWritten.current) writeWorkspace(latest.current)
    }
    window.addEventListener('pagehide', flush)
    return () => {
      window.removeEventListener('pagehide', flush)
      flush()
    }
  }, [])

  useEffect(() => {
    if (blocked.current || workspace === lastWritten.current) return
    const timer = window.setTimeout(() => {
      const result = writeWorkspace(workspace)
      if (result.ok) lastWritten.current = workspace
      else setMessage(result.error)
    }, 350)
    return () => window.clearTimeout(timer)
  }, [workspace])

  function commit(next: Workspace): boolean {
    if (blocked.current) {
      setMessage('invalidStorage')
      return false
    }
    const result = writeWorkspace(next)
    if (!result.ok) {
      setMessage(result.error)
      return false
    }
    lastWritten.current = next
    setWorkspace(next)
    setMessage('saved')
    return true
  }

  function clear(): boolean {
    const next = emptyWorkspace()
    const result = writeWorkspace(next)
    if (!result.ok) {
      setMessage(result.error)
      return false
    }
    blocked.current = false
    lastWritten.current = next
    setWorkspace(next)
    setMessage('cleared')
    return true
  }

  return { workspace, setWorkspace, message, setMessage, commit, clear }
}
