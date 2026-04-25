import { useEffect, useRef } from 'react'
import { useLocale } from '@/contexts/LocaleContext'

interface Props {
  content: string
  maxHeight?: string
}

export default function LogViewer({ content, maxHeight = '500px' }: Props) {
  const { t } = useLocale()
  const ref = useRef<HTMLPreElement>(null)

  useEffect(() => {
    if (ref.current) {
      ref.current.scrollTop = ref.current.scrollHeight
    }
  }, [content])

  return (
    <pre
      ref={ref}
      className="text-xs p-4 rounded-[var(--radius-sm)] overflow-auto border border-[var(--border)]"
      style={{
        maxHeight,
        background: 'var(--bg-deep)',
        color: 'var(--text-muted)',
        fontFamily: 'var(--font-mono)',
      }}
    >
      {content || t('common.loading')}
    </pre>
  )
}
