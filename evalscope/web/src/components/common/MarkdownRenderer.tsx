import { useState } from 'react'
import { createPortal } from 'react-dom'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import remarkMath from 'remark-math'
import rehypeKatex from 'rehype-katex'
import type { Components } from 'react-markdown'
import { X } from 'lucide-react'
import { useTheme } from '@/contexts/ThemeContext'
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter'
import { vscDarkPlus } from 'react-syntax-highlighter/dist/esm/styles/prism'
import { oneLight } from 'react-syntax-highlighter/dist/esm/styles/prism'

interface Props {
  content: string
}

function ImageWithLightbox({ src, alt }: { src?: string; alt?: string }) {
  const [open, setOpen] = useState(false)
  if (!src) return null
  return (
    <>
      <img
        src={src}
        alt={alt ?? ''}
        onClick={() => setOpen(true)}
        className="rounded-lg cursor-zoom-in border border-[var(--color-border)] hover:border-[var(--color-border-strong)] transition-all hover:scale-[1.02]"
        style={{ maxHeight: 200, maxWidth: 320, display: 'inline-block', verticalAlign: 'top' }}
      />
      {open && createPortal(
        <div
          className="fixed inset-0 z-[9999] flex items-center justify-center"
          style={{ background: 'var(--overlay-bg)', backdropFilter: 'blur(6px)' }}
          onClick={() => setOpen(false)}
        >
          <div className="relative max-w-[90vw] max-h-[90vh]" onClick={(e) => e.stopPropagation()}>
            <button
              onClick={() => setOpen(false)}
              className="absolute -top-3 -right-3 z-10 rounded-full p-1 bg-[var(--color-surface)] border border-[var(--color-border)] hover:bg-[var(--color-surface-hover)] transition-colors"
            >
              <X size={16} />
            </button>
            <img
              src={src}
              alt={alt ?? ''}
              className="max-w-full max-h-[85vh] rounded-xl object-contain shadow-2xl"
            />
          </div>
        </div>,
        document.body
      )}
    </>
  )
}

export default function MarkdownRenderer({ content }: Props) {
  const { theme } = useTheme()

  const markdownComponents: Components = {
    img: ({ src, alt }) => <ImageWithLightbox src={src} alt={alt} />,
    code: ({ className, children }) => {
      const match = /language-(\w+)/.exec(className || '')
      const language = match ? match[1] : ''

      if (match) {
        return (
          <SyntaxHighlighter
            language={language}
            style={(theme === 'dark' ? vscDarkPlus : oneLight) as any}
            PreTag="div"
            customStyle={{
              margin: 0,
              borderRadius: '0.5rem',
              padding: '1rem',
              fontSize: '0.8125rem',
              lineHeight: 1.6,
            }}
            codeTagProps={{
              style: {
                fontFamily: 'var(--font-mono)',
              },
            }}
          >
            {String(children).replace(/\n$/, '')}
          </SyntaxHighlighter>
        )
      }

      return (
        <code className="bg-[var(--color-surface)] px-1.5 py-0.5 rounded text-[0.85em] font-mono">
          {children}
        </code>
      )
    },
    pre: ({ children }) => (
      <div className="not-prose my-3">{children}</div>
    ),
    table: ({ children }) => (
      <div className="overflow-x-auto my-4">
        <table className="w-full">{children}</table>
      </div>
    ),
  }

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
