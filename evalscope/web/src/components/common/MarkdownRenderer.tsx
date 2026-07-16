import { useEffect, useMemo, useState } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import type { Components, Options } from 'react-markdown'
import { useTheme } from '@/contexts/ThemeContext'
import { useLocale } from '@/contexts/LocaleContext'
import { containsMath, isLargeTable, shouldRenderHeavy } from '@/domain/markdown/heavyContent'
import ImageLightbox from './ImageLightbox'
import LazyCodeBlock from './LazyCodeBlock'

interface Props {
  content: string
  /**
   * Whether heavy content (math, code blocks, large tables) is collapsed.
   * Defaults to `false` so the rendered output is identical to before; when a
   * caller collapses the region the heavy content is not rendered (Req 16.3)
   * and its optional modules are not loaded (Req 16.1).
   */
  collapsed?: boolean
}

const INLINE_IMG_STYLE = { maxHeight: 200, maxWidth: 320, display: 'inline-block', verticalAlign: 'top' as const }

type MathPlugins = {
  remark: NonNullable<Options['remarkPlugins']>
  rehype: NonNullable<Options['rehypePlugins']>
}

type MathState = 'idle' | 'ready' | 'error'

interface MathLoadResult {
  key: string
  plugins: MathPlugins | null
  state: MathState
}

/** Extract the number of Markdown source lines a hast node spans, or 0 if unknown. */
function nodeLineSpan(node: unknown): number {
  const position = (node as { position?: { start?: { line?: number }; end?: { line?: number } } })?.position
  const start = position?.start?.line
  const end = position?.end?.line
  if (typeof start !== 'number' || typeof end !== 'number') {
    return 0
  }
  return end - start + 1
}

/**
 * Lazily load the math pipeline (`remark-math` + `rehype-katex` + KaTeX CSS).
 *
 * The modules are imported only when the content actually contains math and the
 * math region is not collapsed (Req 16.1, 16.3). On load failure the hook
 * reports `'error'` so the caller can show a placeholder while still rendering
 * the rest of the document (Req 16.5).
 */
function useMathPlugins(content: string, collapsed: boolean): { plugins: MathPlugins | null; state: MathState } {
  const needsMath = useMemo(
    () => containsMath(content) && shouldRenderHeavy(collapsed, 'math'),
    [content, collapsed],
  )
  const requestKey = needsMath ? content : ''
  const [result, setResult] = useState<MathLoadResult>({ key: '', plugins: null, state: 'idle' })

  useEffect(() => {
    if (!needsMath) return
    let cancelled = false
    Promise.all([
      import('remark-math'),
      import('rehype-katex'),
      import('katex/dist/katex.min.css'),
    ])
      .then(([remarkMath, rehypeKatex]) => {
        if (cancelled) return
        setResult({
          key: requestKey,
          plugins: { remark: [remarkMath.default], rehype: [rehypeKatex.default] },
          state: 'ready',
        })
      })
      .catch(() => {
        if (cancelled) return
        setResult({ key: requestKey, plugins: null, state: 'error' })
      })
    return () => {
      cancelled = true
    }
  }, [needsMath, requestKey])

  if (!needsMath || result.key !== requestKey) return { plugins: null, state: 'idle' }
  return { plugins: result.plugins, state: result.state }
}

export default function MarkdownRenderer({ content, collapsed = false }: Props) {
  const { theme } = useTheme()
  const { t } = useLocale()
  const { plugins: mathPlugins, state: mathState } = useMathPlugins(content, collapsed)

  const markdownComponents = useMemo<Components>(() => ({
    img: ({ src, alt }) => <ImageLightbox src={src ?? ''} alt={alt} style={INLINE_IMG_STYLE} />,
    code: ({ className, children }) => {
      const match = /language-(\w+)/.exec(className || '')
      if (match) {
        // Fenced code block: heavy content, gated by collapse (Req 16.3) and
        // rendered by the on-demand highlighter (Req 16.1, 16.2).
        if (!shouldRenderHeavy(collapsed, 'code')) {
          return (
            <div className="not-prose my-3 text-[0.75rem] text-[var(--text-dim)] italic" role="status">
              {t('markdown.collapsedContent')}
            </div>
          )
        }
        return <LazyCodeBlock language={match[1]} value={String(children)} />
      }
      return (
        <code className="bg-[var(--bg-card)] px-1.5 py-0.5 rounded text-[0.85em] font-mono">
          {children}
        </code>
      )
    },
    pre: ({ children }) => <div className="not-prose my-3">{children}</div>,
    table: ({ children, node }) => {
      // A table longer than the threshold is heavy and gated by collapse
      // (Req 16.3); smaller tables always render.
      const large = isLargeTable(nodeLineSpan(node))
      if (large && !shouldRenderHeavy(collapsed, 'large-table')) {
        return (
          <div className="not-prose my-3 text-[0.75rem] text-[var(--text-dim)] italic" role="status">
            {t('markdown.collapsedContent')}
          </div>
        )
      }
      return (
        <div className="overflow-x-auto my-4">
          <table className="w-full">{children}</table>
        </div>
      )
    },
  }), [collapsed, t])

  if (!content) return null

  const remarkPlugins: NonNullable<Options['remarkPlugins']> = mathPlugins
    ? [remarkGfm, ...mathPlugins.remark]
    : [remarkGfm]
  const rehypePlugins: NonNullable<Options['rehypePlugins']> = mathPlugins ? mathPlugins.rehype : []

  return (
    <div className={`prose prose-sm max-w-none break-words${theme === 'dark' ? ' prose-invert' : ''}`}>
      {mathState === 'error' && (
        <div className="text-[0.75rem] text-[var(--text-dim)] mb-1" role="status">
          {t('markdown.mathLoadError')}
        </div>
      )}
      <ReactMarkdown
        remarkPlugins={remarkPlugins}
        rehypePlugins={rehypePlugins}
        urlTransform={(url) => url}
        components={markdownComponents}
      >
        {content}
      </ReactMarkdown>
    </div>
  )
}
