import { readFileSync, readdirSync, statSync } from 'node:fs'
import { dirname, join, relative, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

/**
 * Structural drift checks over `src/`.
 *
 * These guard the three failure modes that reviews kept catching by hand:
 *
 * 1. **Phantom typography classes.** `type-*` utilities are declared in
 *    `index.css`; a typo such as `type-title-sm` silently renders as no class at
 *    all, so the element quietly loses its size token.
 * 2. **Dead design-system components.** A component under `ui/` or `common/`
 *    with no importer is either superseded or was never wired up — both mean the
 *    documented component layer no longer describes the app.
 * 3. **Misplaced pages.** `pages/` is for route targets. A tab or panel parked
 *    there reads as a route that does not exist, and is invisible to anyone
 *    tracing the router.
 */

export interface SourceCheckResult {
  ok: boolean
  /** `type-*` classes used in source but never declared in `index.css`. */
  undefinedTypeClasses: string[]
  /** Components under `ui/` or `common/` that nothing imports. */
  unusedComponents: string[]
  /** Modules under `pages/` that the router never renders. */
  unroutedPages: string[]
}

/** Files that legitimately live in `pages/` without being a route target. */
const PAGE_ALLOWLIST = new Set<string>([])

function walk(dir: string, out: string[] = []): string[] {
  for (const entry of readdirSync(dir)) {
    const full = join(dir, entry)
    if (statSync(full).isDirectory()) walk(full, out)
    else if (/\.tsx?$/.test(entry)) out.push(full)
  }
  return out
}

const isTest = (path: string) => /\.(test|spec)\.tsx?$/.test(path)

/**
 * Run every structural check against a source tree.
 *
 * @param srcRoot - Absolute path of the `src` directory.
 * @returns The findings; `ok` is true only when all three lists are empty.
 */
export function checkSource(srcRoot: string): SourceCheckResult {
  const files = walk(srcRoot)
  const sources = files.filter((f) => !isTest(f))
  const contents = new Map(files.map((f) => [f, readFileSync(f, 'utf8')]))

  // ── 1. Typography classes ──────────────────────────────────────────────
  const css = readFileSync(join(srcRoot, 'index.css'), 'utf8')
  const declared = new Set(Array.from(css.matchAll(/\.(type-[\w-]+)\s*\{/g), (m) => m[1]))
  const used = new Set<string>()
  for (const [file, text] of contents) {
    if (file.endsWith('.css')) continue
    for (const match of text.matchAll(/\b(type-[a-z0-9-]+)\b/g)) used.add(match[1])
  }
  const undefinedTypeClasses = [...used].filter((name) => !declared.has(name)).sort()

  // ── 2. Dead design-system components ───────────────────────────────────
  const unusedComponents: string[] = []
  for (const file of sources) {
    const rel = relative(srcRoot, file)
    if (!/^components\/(ui|common)\//.test(rel)) continue
    const moduleName = rel.replace(/\.tsx?$/, '')
    const importPath = `@/${moduleName}`
    const baseName = moduleName.split('/').pop()!
    const referenced = sources.some((other) => {
      const text = contents.get(other) ?? ''
      return other !== file
        && (text.includes(`'${importPath}'`) || text.includes(`'./${baseName}'`))
    })
    if (!referenced) unusedComponents.push(rel)
  }

  // ── 3. Page modules the router never renders ───────────────────────────
  const appSource = contents.get(join(srcRoot, 'App.tsx')) ?? ''
  const unroutedPages: string[] = []
  const pagesDir = join(srcRoot, 'pages')
  for (const file of sources) {
    if (!file.startsWith(pagesDir)) continue
    const rel = relative(srcRoot, file)
    if (PAGE_ALLOWLIST.has(rel)) continue
    const moduleName = rel.replace(/\.tsx?$/, '')
    if (!appSource.includes(`@/${moduleName}'`)) unroutedPages.push(rel)
  }

  return {
    ok: undefinedTypeClasses.length === 0
      && unusedComponents.length === 0
      && unroutedPages.length === 0,
    undefinedTypeClasses,
    unusedComponents,
    unroutedPages: unroutedPages.sort(),
  }
}

export function runSourceCheck(): SourceCheckResult {
  const webRoot = resolve(dirname(fileURLToPath(import.meta.url)), '..', '..')
  return checkSource(join(webRoot, 'src'))
}

export function formatSourceReport(result: SourceCheckResult): string {
  if (result.ok) return 'Source structure check passed.'
  const lines = ['Source structure check failed:']
  if (result.undefinedTypeClasses.length > 0) {
    lines.push(`  undefined type-* classes: ${result.undefinedTypeClasses.join(', ')}`)
  }
  if (result.unusedComponents.length > 0) {
    lines.push(`  components with no importer: ${result.unusedComponents.join(', ')}`)
  }
  if (result.unroutedPages.length > 0) {
    lines.push(`  pages/ modules not reachable from the router: ${result.unroutedPages.join(', ')}`)
  }
  return lines.join('\n')
}

if (import.meta.url === `file://${process.argv[1]}`) {
  const result = runSourceCheck()
  process.stdout.write(`${formatSourceReport(result)}\n`)
  if (!result.ok) process.exitCode = 1
}
