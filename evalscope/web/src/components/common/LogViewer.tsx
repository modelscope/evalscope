import { useEffect, useRef } from 'react'

interface Props {
  content: string
  maxHeight?: string
}

export default function LogViewer({ content, maxHeight = '500px' }: Props) {
  const ref = useRef<HTMLPreElement>(null)

  useEffect(() => {
    if (ref.current) {
      ref.current.scrollTop = ref.current.scrollHeight
    }
  }, [content])

  return (
    <pre
      ref={ref}
      className="bg-[#0d1117] text-[#c9d1d9] text-xs font-mono p-3 rounded-lg overflow-auto border border-[var(--color-border)]"
      style={{ maxHeight }}
    >
      {content || 'Waiting for output...'}
    </pre>
  )
}
