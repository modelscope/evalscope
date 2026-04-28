import { useState } from 'react'
import { createPortal } from 'react-dom'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import remarkMath from 'remark-math'
import rehypeKatex from 'rehype-katex'
import type { Components } from 'react-markdown'
import { X } from 'lucide-react'
import { useTheme } from '@/contexts/ThemeContext'

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

const markdownComponents: Components = {
  img: ({ src, alt }) => <ImageWithLightbox src={src} alt={alt} />,
}

export default function MarkdownRenderer({ content }: Props) {
  const { theme } = useTheme()
  if (!content) return null
  return (
    <div className={`prose prose-sm max-w-none break-words [&_table]:text-xs [&_pre]:bg-[var(--color-surface)] [&_code]:bg-[var(--color-surface)] [&_code]:px-1 [&_code]:rounded [&_img]:max-w-full [&_img]:rounded${theme === 'dark' ? ' prose-invert' : ''}`}>
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
