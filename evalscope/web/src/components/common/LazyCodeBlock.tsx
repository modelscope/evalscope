import { useEffect, useState, type ComponentType, type CSSProperties } from 'react'
import { useTheme } from '@/contexts/ThemeContext'
import { useLocale } from '@/contexts/LocaleContext'

interface Props {
  language: string
  value: string
}

type PrismStyle = { [key: string]: CSSProperties }

interface HighlighterModule {
  SyntaxHighlighter: ComponentType<Record<string, unknown>>
  vscDarkPlus: PrismStyle
  oneLight: PrismStyle
}

type LoadState = 'loading' | 'ready' | 'error'

interface LoadResult {
  language: string
  state: LoadState
  module: HighlighterModule | null
}

const CODE_CUSTOM_STYLE: CSSProperties = {
  margin: 0,
  borderRadius: '0.5rem',
  padding: '1rem',
  fontSize: '0.8125rem',
  lineHeight: 1.6,
}

const RAW_CODE_STYLE: CSSProperties = {
  ...CODE_CUSTOM_STYLE,
  overflowX: 'auto',
  fontFamily: 'var(--font-mono)',
  background: 'var(--bg-card)',
  whiteSpace: 'pre',
}

/**
 * Dynamically-loaded syntax highlighter.
 *
 * The heavy `react-syntax-highlighter` core is imported via `import()` only
 * when a code block is actually rendered, and only the specific Prism language
 * module for this block is registered — never the full language set.
 * The core module and styles are cached at module scope, and each language is
 * registered at most once.
 *
 * Until the highlighter is ready the raw code is shown as plain preformatted
 * text, so content is never blank. If the on-demand modules fail to load, a
 * localized placeholder is shown alongside the raw code and the rest of the
 * document is unaffected.
 */

// Module-scope caches so the highlighter core + styles load at most once, and
// each Prism language is registered at most once across all code blocks.
let highlighterPromise: Promise<HighlighterModule> | null = null
const registeredLanguages = new Set<string>()
const languagePromises = new Map<string, Promise<void>>()

async function loadHighlighter(): Promise<HighlighterModule> {
  if (!highlighterPromise) {
    highlighterPromise = (async () => {
      const [core, styles] = await Promise.all([
        import('react-syntax-highlighter/dist/esm/prism-async-light'),
        import('react-syntax-highlighter/dist/esm/styles/prism'),
      ])
      return {
        SyntaxHighlighter: core.default as unknown as ComponentType<Record<string, unknown>>,
        vscDarkPlus: styles.vscDarkPlus as PrismStyle,
        oneLight: styles.oneLight as PrismStyle,
      }
    })().catch((err) => {
      // Reset so a later render can retry the load.
      highlighterPromise = null
      throw err
    })
  }
  return highlighterPromise
}

/** Normalize a fenced language token to a safe Prism language id, or '' if none. */
function normalizeLanguage(language: string): string {
  return language.toLowerCase().replace(/[^a-z0-9#+-]/g, '')
}

async function registerLanguage(
  SyntaxHighlighter: ComponentType<Record<string, unknown>>,
  language: string,
): Promise<void> {
  if (!language || registeredLanguages.has(language)) {
    return
  }
  let promise = languagePromises.get(language)
  if (!promise) {
    promise = (async () => {
      // Only the language actually used by this block is imported.
      const mod = await import(
        /* @vite-ignore */ `react-syntax-highlighter/dist/esm/languages/prism/${language}`
      )
      const register = (SyntaxHighlighter as unknown as {
        registerLanguage?: (name: string, syntax: unknown) => void
      }).registerLanguage
      register?.(language, mod.default)
      registeredLanguages.add(language)
    })().catch((err) => {
      // A missing/unknown language is non-fatal: drop the cache entry so we do
      // not permanently mark it registered, then rethrow for the caller.
      languagePromises.delete(language)
      throw err
    })
    languagePromises.set(language, promise)
  }
  return promise
}

export default function LazyCodeBlock({ language, value }: Props) {
  const { theme } = useTheme()
  const { t } = useLocale()
  const normalized = normalizeLanguage(language)
  const [result, setResult] = useState<LoadResult>({ language: normalized, state: 'loading', module: null })

  useEffect(() => {
    let cancelled = false
    loadHighlighter()
      .then(async (mod) => {
        // A failure to register a specific language must not fail the whole
        // block; fall back to the highlighter without that grammar.
        try {
          await registerLanguage(mod.SyntaxHighlighter, normalized)
        } catch {
          /* unknown language — render without its grammar */
        }
        if (cancelled) return
        setResult({ language: normalized, state: 'ready', module: mod })
      })
      .catch(() => {
        if (cancelled) return
        setResult({ language: normalized, state: 'error', module: null })
      })
    return () => {
      cancelled = true
    }
  }, [normalized])

  const current = result.language === normalized ? result : { language: normalized, state: 'loading' as const, module: null }

  if (current.state === 'ready' && current.module) {
    const { SyntaxHighlighter, vscDarkPlus, oneLight } = current.module
    return (
      <SyntaxHighlighter
        language={normalized || 'text'}
        style={theme === 'dark' ? vscDarkPlus : oneLight}
        PreTag="div"
        customStyle={CODE_CUSTOM_STYLE}
        codeTagProps={{ style: { fontFamily: 'var(--font-mono)' } }}
      >
        {value.replace(/\n$/, '')}
      </SyntaxHighlighter>
    )
  }

  // Loading and error states both show the raw code so content is never blank.
  // The error state adds a localized placeholder without hiding the code.
  return (
    <div>
      {current.state === 'error' && (
        <div className="text-[0.75rem] text-[var(--text-dim)] mb-1" role="status">
          {t('markdown.codeLoadError')}
        </div>
      )}
      <pre style={RAW_CODE_STYLE} className="not-prose">
        <code style={{ fontFamily: 'var(--font-mono)' }}>{value.replace(/\n$/, '')}</code>
      </pre>
    </div>
  )
}
