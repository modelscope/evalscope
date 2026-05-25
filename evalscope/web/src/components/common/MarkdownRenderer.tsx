import { useMemo } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import remarkMath from 'remark-math'
import rehypeKatex from 'rehype-katex'
import type { Components } from 'react-markdown'
import { useTheme } from '@/contexts/ThemeContext'
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter'
import { vscDarkPlus, oneLight } from 'react-syntax-highlighter/dist/esm/styles/prism'
import ImageLightbox from './ImageLightbox'

interface Props {
  content: string
}

const INLINE_IMG_STYLE = { maxHeight: 200, maxWidth: 320, display: 'inline-block', verticalAlign: 'top' as const }

export default function MarkdownRenderer({ content }: Props) {
  const { theme } = useTheme()

  const markdownComponents = useMemo<Components>(() => ({
    img: ({ src, alt }) => <ImageLightbox src={src ?? ''} alt={alt} style={INLINE_IMG_STYLE} />,
    code: ({ className, children }) => {
      const match = /language-(\w+)/.exec(className || '')
      if (match) {
        return (
          <SyntaxHighlighter
            language={match[1]}
            style={(theme === 'dark' ? vscDarkPlus : oneLight) as any}
            PreTag="div"
            customStyle={{
              margin: 0,
              borderRadius: '0.5rem',
              padding: '1rem',
              fontSize: '0.8125rem',
              lineHeight: 1.6,
            }}
            codeTagProps={{ style: { fontFamily: 'var(--font-mono)' } }}
          >
            {String(children).replace(/\n$/, '')}
          </SyntaxHighlighter>
        )
      }
      return (
        <code className="bg-[var(--bg-card)] px-1.5 py-0.5 rounded text-[0.85em] font-mono">
          {children}
        </code>
      )
    },
    pre: ({ children }) => <div className="not-prose my-3">{children}</div>,
    table: ({ children }) => (
      <div className="overflow-x-auto my-4">
        <table className="w-full">{children}</table>
      </div>
    ),
  }), [theme])

  if (!content) return null

  return (
    <div className={`prose prose-sm max-w-none break-words${theme === 'dark' ? ' prose-invert' : ''}`}>
      <ReactMarkdown
        remarkPlugins={[remarkGfm, remarkMath]}
        rehypePlugins={[rehypeKatex]}
        urlTransform={(url) => url}
        components={markdownComponents}
      >
        {content}
      </ReactMarkdown>
    </div>
  )
}
