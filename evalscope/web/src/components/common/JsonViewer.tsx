import { Light as SyntaxHighlighter } from 'react-syntax-highlighter'
import json from 'react-syntax-highlighter/dist/esm/languages/hljs/json'
import { atomOneDark } from 'react-syntax-highlighter/dist/esm/styles/hljs'
import { atomOneLight } from 'react-syntax-highlighter/dist/esm/styles/hljs'
import { useTheme } from '@/contexts/ThemeContext'

SyntaxHighlighter.registerLanguage('json', json)

interface Props {
  value: unknown
  maxHeight?: string | number
  className?: string
}

export default function JsonViewer({ value, maxHeight = 400, className = '' }: Props) {
  const { theme } = useTheme()
  let code: string
  try {
    code = typeof value === 'string' ? value : JSON.stringify(value, null, 2)
  } catch {
    code = String(value)
  }

  const maxH = typeof maxHeight === 'number' ? `${maxHeight}px` : maxHeight

  return (
    <div className={`text-xs overflow-auto rounded-lg ${className}`} style={{ maxHeight: maxH }}>
      <SyntaxHighlighter
        language="json"
        style={theme === 'dark' ? atomOneDark : atomOneLight}
        customStyle={{
          background: 'transparent',
          padding: 0,
          margin: 0,
          fontSize: '0.75rem',
          lineHeight: '1.6',
        }}
        wrapLongLines
      >
        {code}
      </SyntaxHighlighter>
    </div>
  )
}
