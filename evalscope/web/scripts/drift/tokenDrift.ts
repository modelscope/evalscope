import { readFileSync, writeFileSync } from 'node:fs'
import { dirname, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

type TokenScope = 'shared' | 'dark' | 'light'

interface TokenLink {
  css: string
  markdown: string
  scope: TokenScope
}

const TOKEN_LINKS: readonly TokenLink[] = [
  ...[
    ['--radius-xs', 'rounded.xs'],
    ['--radius-sm', 'rounded.sm'],
    ['--radius', 'rounded.md'],
    ['--radius-lg', 'rounded.lg'],
    ['--radius-xl', 'rounded.xl'],
    ['--transition-fast', 'transition.fast'],
    ['--transition-base', 'transition.base'],
    ['--transition-slow', 'transition.slow'],
  ].map(([css, markdown]) => ({ css, markdown, scope: 'shared' as const })),
  ...[
    ['--accent', 'colors.accent', 'colors.accent-light'],
    ['--accent-dim', 'colors.accent-dim', 'colors.accent-dim-light'],
    ['--bg', 'colors.bg', 'colors.bg-light'],
    ['--bg-deep', 'colors.bg-deep', 'colors.bg-deep-light'],
    ['--bg-card', 'colors.bg-card', 'colors.bg-card-light'],
    ['--bg-card2', 'colors.bg-card2', 'colors.bg-card2-light'],
    ['--surface-glass', 'colors.surface-glass', 'colors.surface-glass-light'],
    ['--text', 'colors.text', 'colors.text-light'],
    ['--text-muted', 'colors.text-muted', 'colors.text-muted-light'],
    ['--text-dim', 'colors.text-dim', 'colors.text-dim-light'],
    ['--border', 'colors.border', 'colors.border-light'],
    ['--border-md', 'colors.border-md', 'colors.border-md-light'],
    ['--border-strong', 'colors.border-strong', 'colors.border-strong-light'],
    ['--shadow-sm', 'shadows.sm', 'shadows.sm-light'],
    ['--shadow', 'shadows.md', 'shadows.md-light'],
    ['--shadow-lg', 'shadows.lg', 'shadows.lg-light'],
    ['--shadow-glow', 'shadows.glow', 'shadows.glow-light'],
    ['--shadow-glow-soft', 'shadows.glow-soft', 'shadows.glow-soft-light'],
  ].flatMap(([css, dark, light]) => [
    { css, markdown: dark, scope: 'dark' as const },
    { css, markdown: light, scope: 'light' as const },
  ]),
]

export interface TokenDriftRunResult {
  ok: boolean
  changed: string[]
}

function declarations(css: string, selector: RegExp): Map<string, string> {
  const body = css.replace(/\/\*[\s\S]*?\*\//g, '').match(selector)?.[1] ?? ''
  return new Map([...body.matchAll(/(--[\w-]+)\s*:\s*([^;{}]+);/g)].map((match) => [match[1], match[2].trim()]))
}

function cssValues(css: string): Record<TokenScope, Map<string, string>> {
  return {
    shared: declarations(css, /:root\s*\{([^}]*)\}/),
    dark: declarations(css, /:root\s*,\s*\[data-theme="dark"\]\s*\{([^}]*)\}/),
    light: declarations(css, /\[data-theme="light"\]\s*\{([^}]*)\}/),
  }
}

function frontmatterLines(markdown: string): Map<string, number> {
  const lines = markdown.split(/\r?\n/)
  const result = new Map<string, number>()
  const stack: { indent: number; key: string }[] = []
  let fenceCount = 0

  lines.forEach((line, index) => {
    if (line.trim() === '---') {
      fenceCount++
      return
    }
    if (fenceCount !== 1 || line.trim() === '' || line.trimStart().startsWith('#')) return
    const match = line.match(/^(\s*)([^:]+):\s*(.*)$/)
    if (!match) return
    const indent = match[1].length
    const key = match[2].trim()
    const value = match[3].trim()
    while (stack.length > 0 && stack[stack.length - 1].indent >= indent) stack.pop()
    const path = [...stack.map((entry) => entry.key), key].join('.')
    if (value === '') stack.push({ indent, key })
    else result.set(path, index)
  })
  return result
}

function inlineCommentIndex(value: string): number {
  let single = false
  let double = false
  for (let index = 0; index < value.length; index++) {
    const char = value[index]
    if (char === "'" && !double) single = !single
    else if (char === '"' && !single) double = !double
    else if (char === '#' && !single && !double && (index === 0 || /\s/.test(value[index - 1]))) return index
  }
  return -1
}

export function syncDesignTokens(css: string, markdown: string): { markdown: string; changed: string[] } {
  const values = cssValues(css)
  const lines = markdown.split(/\r?\n/)
  const entries = frontmatterLines(markdown)
  const changed: string[] = []

  for (const link of TOKEN_LINKS) {
    const value = values[link.scope].get(link.css)
    const lineIndex = entries.get(link.markdown)
    if (value === undefined || lineIndex === undefined) {
      changed.push(`${link.markdown} (missing)`)
      continue
    }
    const line = lines[lineIndex]
    const colon = line.indexOf(':')
    const tail = line.slice(colon + 1)
    const commentAt = inlineCommentIndex(tail)
    const comment = commentAt >= 0 ? ` ${tail.slice(commentAt).trimStart()}` : ''
    const next = `${line.slice(0, colon + 1)} ${JSON.stringify(value)}${comment}`
    if (line !== next) {
      lines[lineIndex] = next
      changed.push(link.markdown)
    }
  }
  return { markdown: lines.join('\n'), changed }
}

function resolvePaths(): { cssPath: string; markdownPath: string } {
  const webRoot = resolve(dirname(fileURLToPath(import.meta.url)), '..', '..')
  return { cssPath: resolve(webRoot, 'src/index.css'), markdownPath: resolve(webRoot, '..', '..', 'DESIGN.md') }
}

export function runTokenDriftCheck(write = false): TokenDriftRunResult {
  const { cssPath, markdownPath } = resolvePaths()
  const current = readFileSync(markdownPath, 'utf8')
  const result = syncDesignTokens(readFileSync(cssPath, 'utf8'), current)
  if (write && result.changed.length > 0) writeFileSync(markdownPath, result.markdown)
  return { ok: result.changed.length === 0, changed: result.changed }
}

export function formatDriftReport(result: TokenDriftRunResult): string {
  if (result.ok) return 'Token drift check passed: CSS is the source and DESIGN.md is synchronized.'
  return `Token drift check failed: run npm run design:tokens to sync ${result.changed.join(', ')}.`
}

if (import.meta.url === `file://${process.argv[1]}`) {
  const write = process.argv.includes('--write')
  const result = runTokenDriftCheck(write)
  process.stdout.write(`${write ? 'Design tokens synchronized from CSS.' : formatDriftReport(result)}\n`)
  if (!write && !result.ok) process.exitCode = 1
}
