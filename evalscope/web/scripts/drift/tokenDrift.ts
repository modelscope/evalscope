/**
 * Design token drift check (Req 17.3, 17.4).
 *
 * `checkTokenDrift` is a PURE function: given the design token single source of
 * truth (SSOT), the raw CSS text (`src/index.css`) and the raw Markdown text
 * (`DESIGN.md`), it compares every token value in all three places and returns
 * one {@link DriftEntry} for each token whose three values are not byte-for-byte
 * identical. It never mutates the SSOT and never writes files (Req 17.4 — the
 * caller decides how to fail).
 *
 * The comparison is intentionally byte-for-byte (no whitespace normalization),
 * because Req 17.2 requires the same token to read identically in both surfaces;
 * a `rgba(1,2,3,0.1)` in Markdown vs `rgba(1, 2, 3, 0.1)` in CSS IS drift and
 * must be surfaced rather than silently reconciled.
 *
 * A small file-reading runner ({@link runTokenDriftCheck}) is exported for
 * CLI/CI use, but the core comparison stays a pure function so it is trivial to
 * property-test (task 17.4, Property 29).
 */

import { readFileSync } from 'node:fs'
import { dirname, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

import { flattenTokens, TOKEN_SOURCE } from '../../design/tokens.source.ts'
import type { ThemeName, TokenSource } from '../../design/tokens.source.ts'

/**
 * A single token whose value is inconsistent across the SSOT, the CSS source and
 * the Markdown source. `token` is the reporting identifier (theme-qualified for
 * themed tokens so dark/light drift is distinguishable), and the two value fields
 * carry exactly what was found in each surface (or {@link MISSING_VALUE} when the
 * token could not be located there).
 */
export interface DriftEntry {
  /** Reporting identifier, e.g. `radius-sm` or `accent (light)`. */
  token: string
  /** Value found in the Markdown source, or {@link MISSING_VALUE} if absent. */
  markdownValue: string
  /** Value found in the CSS source, or {@link MISSING_VALUE} if absent. */
  cssValue: string
}

/** Sentinel used in a {@link DriftEntry} when a token has no value in a surface. */
export const MISSING_VALUE = '(not found)'

// ─────────────────────────────────────────────────────────────────────────────
// CSS parsing
// ─────────────────────────────────────────────────────────────────────────────

/** Per-theme resolved CSS custom-property maps (theme block merged over :root). */
type CssThemeMaps = Record<ThemeName, Map<string, string>>

/** Strip `/* ... *\/` block comments so they cannot mask selectors or values. */
function stripCssComments(css: string): string {
  return css.replace(/\/\*[\s\S]*?\*\//g, '')
}

/**
 * Extract and concatenate the bodies of every block whose selector matches the
 * given pattern. The pattern must end at the opening `{`. Bodies are captured
 * with brace counting so nested braces (e.g. stray at-rules) cannot desync the
 * scan. Multiple matching blocks are merged in document order.
 *
 * @param css CSS text (comments already stripped).
 * @param selectorPattern Global regex matching `<selector> {` for the target scope.
 * @returns The concatenated declaration bodies of all matching blocks.
 */
function extractBlockBodies(css: string, selectorPattern: RegExp): string {
  const re = new RegExp(selectorPattern.source, 'g')
  let body = ''
  let match: RegExpExecArray | null
  while ((match = re.exec(css)) !== null) {
    const start = match.index + match[0].length
    let depth = 1
    let i = start
    for (; i < css.length && depth > 0; i++) {
      const ch = css[i]
      if (ch === '{') {
        depth++
      } else if (ch === '}') {
        depth--
      }
    }
    // Slice up to (but excluding) the closing brace.
    body += css.slice(start, depth === 0 ? i - 1 : i) + '\n'
  }
  return body
}

/**
 * Parse `--name: value;` declarations out of a block body into a map. Later
 * declarations overwrite earlier ones, mirroring the CSS cascade within a scope.
 */
function parseDeclarations(body: string): Map<string, string> {
  const map = new Map<string, string>()
  const declRe = /(--[\w-]+)\s*:\s*([^;{}]+);/g
  let match: RegExpExecArray | null
  while ((match = declRe.exec(body)) !== null) {
    map.set(match[1], match[2].trim())
  }
  return map
}

/** Merge `override` on top of `base`, returning a new map (override wins). */
function mergeMaps(base: Map<string, string>, override: Map<string, string>): Map<string, string> {
  const merged = new Map(base)
  for (const [key, value] of override) {
    merged.set(key, value)
  }
  return merged
}

/**
 * Parse the CSS source into per-theme resolved custom-property maps.
 *
 * Shared tokens live in `:root`; themed tokens are redeclared in
 * `[data-theme="dark"]` / `[data-theme="light"]`. Each theme map is the theme
 * block merged over `:root`, so a shared token resolves via the `:root` fallback
 * and a themed token via its theme block.
 */
function parseCssTokens(css: string): CssThemeMaps {
  const clean = stripCssComments(css)
  const root = parseDeclarations(extractBlockBodies(clean, /:root\s*\{/))
  const dark = parseDeclarations(extractBlockBodies(clean, /\[data-theme\s*=\s*["']dark["']\]\s*\{/))
  const light = parseDeclarations(extractBlockBodies(clean, /\[data-theme\s*=\s*["']light["']\]\s*\{/))
  return {
    dark: mergeMaps(root, dark),
    light: mergeMaps(root, light),
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Markdown (DESIGN.md YAML frontmatter) parsing
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Extract the YAML frontmatter body delimited by the first two `---` fences.
 * Returns an empty string when no frontmatter block is present.
 */
function extractFrontmatter(md: string): string {
  const lines = md.split(/\r?\n/)
  let start = -1
  let end = -1
  for (let i = 0; i < lines.length; i++) {
    if (lines[i].trim() === '---') {
      if (start === -1) {
        start = i
      } else {
        end = i
        break
      }
    }
  }
  if (start === -1 || end === -1) {
    return ''
  }
  return lines.slice(start + 1, end).join('\n')
}

/**
 * Strip a trailing YAML inline comment (`# ...`) that lies outside quotes.
 * A `#` is only treated as a comment when it starts the content or is preceded
 * by whitespace, so hex colors such as `#816DF8` inside quotes are preserved.
 */
function stripInlineComment(text: string): string {
  let inSingle = false
  let inDouble = false
  for (let i = 0; i < text.length; i++) {
    const ch = text[i]
    if (ch === "'" && !inDouble) {
      inSingle = !inSingle
    } else if (ch === '"' && !inSingle) {
      inDouble = !inDouble
    } else if (ch === '#' && !inSingle && !inDouble && (i === 0 || /\s/.test(text[i - 1]))) {
      return text.slice(0, i)
    }
  }
  return text
}

/** Remove a single pair of matching surrounding quotes, preserving inner bytes. */
function unquote(value: string): string {
  if (value.length >= 2) {
    const first = value[0]
    const last = value[value.length - 1]
    if ((first === '"' && last === '"') || (first === "'" && last === "'")) {
      return value.slice(1, -1)
    }
  }
  return value
}

/**
 * Flatten a (shallow-typed) YAML frontmatter block into dotted key paths.
 *
 * Only the minimal subset used by the token frontmatter is supported: nested
 * maps expressed through indentation and `key: value` / `key:` (parent) lines.
 * Sequences and multi-line scalars are not used by the token SSOT and are
 * ignored. Values are byte-preserved apart from surrounding quotes, trailing
 * inline comments and outer whitespace.
 *
 * @param yaml Raw frontmatter text.
 * @returns Map from dotted path (e.g. `colors.accent`, `rounded.xs`) to value.
 */
function parseYamlFlat(yaml: string): Map<string, string> {
  const result = new Map<string, string>()
  const stack: { indent: number; key: string }[] = []
  for (const rawLine of yaml.split(/\r?\n/)) {
    if (rawLine.trim() === '') {
      continue
    }
    const leading = rawLine.match(/^(\s*)(.*)$/)
    if (leading === null) {
      continue
    }
    const indent = leading[1].length
    let content = leading[2]
    if (content.startsWith('#')) {
      continue
    }
    content = stripInlineComment(content).replace(/\s+$/, '')
    if (content === '') {
      continue
    }
    const kv = content.match(/^([^:]+):\s*(.*)$/)
    if (kv === null) {
      continue
    }
    const key = kv[1].trim()
    const rawValue = kv[2]
    // Close any scopes at or deeper than the current indentation.
    while (stack.length > 0 && stack[stack.length - 1].indent >= indent) {
      stack.pop()
    }
    if (rawValue === '') {
      // Parent key: descend into a new nested scope.
      stack.push({ indent, key })
    } else {
      const path = [...stack.map((entry) => entry.key), key].join('.')
      result.set(path, unquote(rawValue.trim()))
    }
  }
  return result
}

/**
 * Parse the Markdown source into a token-value lookup.
 *
 * The lookup contains every flattened dotted path (e.g. `rounded.xs`,
 * `shadows.sm`) plus a group-stripped alias for each nested leaf (e.g.
 * `colors.accent` → `accent`). This lets the SSOT reference color tokens by a
 * bare key while grouped scales (rounded/transition/shadows) keep their prefix.
 * On alias collision the first occurrence in document order wins.
 */
function parseMarkdownTokens(md: string): Map<string, string> {
  const flat = parseYamlFlat(extractFrontmatter(md))
  const enriched = new Map(flat)
  for (const [path, value] of flat) {
    const dot = path.indexOf('.')
    if (dot >= 0) {
      const stripped = path.slice(dot + 1)
      if (!enriched.has(stripped)) {
        enriched.set(stripped, value)
      }
    }
  }
  return enriched
}

// ─────────────────────────────────────────────────────────────────────────────
// Core drift check
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Compare every token's value across the SSOT, the CSS source and the Markdown
 * source, returning one {@link DriftEntry} per token whose three values are not
 * all identical.
 *
 * The returned set is exact — no false negatives and no false positives
 * (Property 29): a token is reported if and only if either surface disagrees
 * with the SSOT canonical value for that (token, theme) pair. This function is
 * pure and performs no I/O or mutation (Req 17.4).
 *
 * @param source Token single source of truth.
 * @param css Raw CSS text (typically `src/index.css`).
 * @param md Raw Markdown text (typically `DESIGN.md`).
 * @returns Drift entries, in SSOT declaration order.
 */
export function checkTokenDrift(source: TokenSource, css: string, md: string): DriftEntry[] {
  const expectations = flattenTokens(source)
  const cssByTheme = parseCssTokens(css)
  const markdownValues = parseMarkdownTokens(md)

  // Token names appearing under more than one theme are themed; qualify their
  // reporting id with the theme so dark/light drift stays distinguishable.
  const nameCounts = new Map<string, number>()
  for (const expectation of expectations) {
    nameCounts.set(expectation.name, (nameCounts.get(expectation.name) ?? 0) + 1)
  }

  const entries: DriftEntry[] = []
  for (const expectation of expectations) {
    const canonical = expectation.value
    const cssValue = cssByTheme[expectation.theme].get(expectation.cssVar)
    const markdownValue = markdownValues.get(expectation.markdownKey)
    if (cssValue === canonical && markdownValue === canonical) {
      continue
    }
    const isThemed = (nameCounts.get(expectation.name) ?? 0) > 1
    const token = isThemed ? `${expectation.name} (${expectation.theme})` : expectation.name
    entries.push({
      token,
      markdownValue: markdownValue ?? MISSING_VALUE,
      cssValue: cssValue ?? MISSING_VALUE,
    })
  }
  return entries
}

// ─────────────────────────────────────────────────────────────────────────────
// CLI / CI runner (impure convenience wrapper around the pure check above)
// ─────────────────────────────────────────────────────────────────────────────

/** Options for {@link runTokenDriftCheck}. All paths default to repo locations. */
export interface TokenDriftRunOptions {
  /** Path to the CSS source. Defaults to `src/index.css` under the web root. */
  cssPath?: string
  /** Path to the Markdown source. Defaults to the repo-root `DESIGN.md`. */
  mdPath?: string
  /** Token source to check against. Defaults to {@link TOKEN_SOURCE}. */
  source?: TokenSource
}

/** Result of a runner invocation: the resolved paths, entries and pass flag. */
export interface TokenDriftRunResult {
  cssPath: string
  mdPath: string
  entries: DriftEntry[]
  /** `true` when no drift was detected. */
  ok: boolean
}

/** Resolve the web project root from this module's location (`scripts/drift`). */
function resolveWebRoot(): string {
  return resolve(dirname(fileURLToPath(import.meta.url)), '..', '..')
}

/**
 * Read the actual CSS and Markdown sources from disk and run the pure
 * {@link checkTokenDrift}. Performs no mutation of the SSOT (Req 17.4); the
 * caller inspects `ok` / `entries` to decide how to fail.
 */
export function runTokenDriftCheck(options: TokenDriftRunOptions = {}): TokenDriftRunResult {
  const webRoot = resolveWebRoot()
  const cssPath = options.cssPath ?? resolve(webRoot, 'src', 'index.css')
  const mdPath = options.mdPath ?? resolve(webRoot, '..', '..', 'DESIGN.md')
  const source = options.source ?? TOKEN_SOURCE

  const css = readFileSync(cssPath, 'utf8')
  const md = readFileSync(mdPath, 'utf8')
  const entries = checkTokenDrift(source, css, md)
  return { cssPath, mdPath, entries, ok: entries.length === 0 }
}

/** Render a human-readable drift report for CLI/CI logs. */
export function formatDriftReport(result: TokenDriftRunResult): string {
  if (result.ok) {
    return 'Token drift check passed: Markdown, CSS and SSOT agree on every token.'
  }
  const lines = [`Token drift check FAILED: ${result.entries.length} inconsistent token(s).`, '']
  for (const entry of result.entries) {
    lines.push(`  • ${entry.token}`)
    lines.push(`      markdown: ${entry.markdownValue}`)
    lines.push(`      css:      ${entry.cssValue}`)
  }
  return lines.join('\n')
}

/** Entry point used when this module is executed directly (`node tokenDrift.ts`). */
function main(): void {
  const result = runTokenDriftCheck()
  process.stdout.write(formatDriftReport(result) + '\n')
  if (!result.ok) {
    // Non-zero exit so CI fails; the SSOT is left untouched (Req 17.4).
    process.exitCode = 1
  }
}

// Only run the CLI when invoked directly, not when imported by tests.
if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  main()
}
