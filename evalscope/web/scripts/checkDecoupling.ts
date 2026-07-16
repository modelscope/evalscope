/**
 * Data-model / rendering decoupling guard (Requirements 15.1, 15.2).
 *
 * Task 16.3 confirms that the Compare and Predictions data models stay separate
 * from their rendering layers:
 *   - Req 15.1: the Compare data model module holds no rendering logic (the
 *     render module holds no data-model definition).
 *   - Req 15.2: the Predictions/Trace presentation selector does not reference
 *     its rendering internals.
 *
 * This script makes that contract enforceable. Its core (`findRenderingImports`)
 * is PURE: given the source text of a data-model module it returns every import
 * specifier that reaches into the rendering layer (React itself, the component
 * tree, or the pages/routing tree). A data-model module MUST have none. A thin
 * file-reading runner ({@link runDecouplingCheck}) reads the guarded modules and
 * is wired to a non-zero exit code for CLI/CI use.
 *
 * The check is intentionally import-based rather than a full parser: a pure data
 * model cannot render without importing React or a component, so a zero-render
 * import surface is a sound, low-false-positive proxy for "no rendering logic".
 */

import { readFileSync } from 'node:fs'
import { dirname, relative, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

/** A rendering-layer import found inside a data-model module. */
export interface RenderingImport {
  /** The imported module specifier, e.g. `react` or `@/components/ui/Card`. */
  specifier: string
  /** 1-based line number of the import statement. */
  line: number
}

/**
 * Decide whether an import specifier reaches into the rendering layer.
 *
 * Rendering imports are:
 *   - `react` / `react-dom` (and their subpaths) — the render runtime itself;
 *   - the component tree: `@/components/...` or any relative path containing a
 *     `components/` segment;
 *   - the pages / routing tree: `@/pages/...` or a relative `pages/` segment;
 *   - `react-router` / `react-router-dom` — routing is a render concern.
 *
 * Type-only imports still count: a data model that needs a rendering type has a
 * design smell and the render module should own that shape instead.
 *
 * @param specifier The raw module specifier from an import statement.
 * @returns `true` when the specifier belongs to the rendering layer.
 */
export function isRenderingSpecifier(specifier: string): boolean {
  if (specifier === 'react' || specifier.startsWith('react/')) return true
  if (specifier === 'react-dom' || specifier.startsWith('react-dom/')) return true
  if (specifier === 'react-router' || specifier === 'react-router-dom') return true

  // Alias-rooted rendering trees.
  if (specifier.startsWith('@/components/') || specifier.startsWith('@/pages/')) return true

  // Relative paths that reach into a components/ or pages/ tree.
  if (/(^|\/)components\//.test(specifier) || /(^|\/)pages\//.test(specifier)) return true

  return false
}

/**
 * Find every rendering-layer import in a module's source text.
 *
 * Pure and side-effect free: the same input always yields the same result. Both
 * `import ... from '...'` and side-effect `import '...'` forms are recognised,
 * including `import type`.
 *
 * @param source Raw source text of a data-model module.
 * @returns The rendering-layer imports in source order.
 */
export function findRenderingImports(source: string): RenderingImport[] {
  const found: RenderingImport[] = []
  // Matches `... from 'x'` and bare `import 'x'`, single or double quoted.
  const re = /\bfrom\s*['"]([^'"]+)['"]|\bimport\s*['"]([^'"]+)['"]/g
  let match: RegExpExecArray | null
  while ((match = re.exec(source)) !== null) {
    const specifier = match[1] ?? match[2]
    if (!specifier || !isRenderingSpecifier(specifier)) continue
    const line = (source.slice(0, match.index).match(/\n/g)?.length ?? 0) + 1
    found.push({ specifier, line })
  }
  return found
}

// ─────────────────────────────────────────────────────────────────────────────
// CLI / CI runner (impure convenience wrapper around the pure core above)
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Data-model modules that MUST NOT import the rendering layer (Req 15.1, 15.2),
 * relative to the web project root.
 */
export const DATA_MODEL_MODULES: readonly string[] = [
  'src/domain/compare/compareModel.ts',
  'src/domain/perf/compareModel.ts',
  'src/domain/predictions/chatPresentation.ts',
]

/** Resolve the web project root from this module's location (`scripts/`). */
function resolveWebRoot(): string {
  return resolve(dirname(fileURLToPath(import.meta.url)), '..')
}

/** Per-module scan result. */
export interface ModuleScanResult {
  /** Module path, web-root-relative. */
  file: string
  /** Rendering imports found in the module (empty when properly decoupled). */
  imports: RenderingImport[]
}

/** Options for {@link runDecouplingCheck}. */
export interface DecouplingCheckRunOptions {
  /** Web project root. Defaults to the directory above `scripts/`. */
  webRoot?: string
  /** Modules to check. Defaults to {@link DATA_MODEL_MODULES}. */
  modules?: readonly string[]
}

/** Outcome of a runner invocation. */
export interface DecouplingCheckResult {
  /** `true` when no data-model module imports the rendering layer. */
  ok: boolean
  /** Only the modules that contain at least one rendering import. */
  violations: ModuleScanResult[]
  /** Number of modules scanned. */
  scannedModules: number
}

/**
 * Read each guarded data-model module and run the pure {@link findRenderingImports}
 * over it. Performs no mutation; the caller inspects `ok` to decide how to fail
 * (Req 15.1, 15.2).
 */
export function runDecouplingCheck(options: DecouplingCheckRunOptions = {}): DecouplingCheckResult {
  const webRoot = options.webRoot ?? resolveWebRoot()
  const modules = options.modules ?? DATA_MODEL_MODULES

  const violations: ModuleScanResult[] = []
  let scannedModules = 0

  for (const module of modules) {
    const absFile = resolve(webRoot, module)
    const source = readFileSync(absFile, 'utf8')
    scannedModules++
    const imports = findRenderingImports(source)
    if (imports.length > 0) {
      violations.push({ file: relative(webRoot, absFile), imports })
    }
  }

  return { ok: violations.length === 0, violations, scannedModules }
}

/** Render a human-readable report for CLI/CI logs. */
export function formatDecouplingReport(result: DecouplingCheckResult): string {
  if (result.ok) {
    return `Decoupling check passed: ${result.scannedModules} data-model module(s) import no rendering layer.`
  }
  const lines = [
    `Decoupling check FAILED: ${result.violations.length} data-model module(s) import the rendering layer.`,
    '',
    'Data models must not depend on React / components / pages (Req 15.1, 15.2).',
    '',
  ]
  for (const violation of result.violations) {
    for (const imp of violation.imports) {
      lines.push(`  • ${violation.file}:${imp.line} imports '${imp.specifier}'`)
    }
  }
  return lines.join('\n')
}

/** Entry point used when this module is executed directly. */
function main(): void {
  const result = runDecouplingCheck()
  process.stdout.write(formatDecouplingReport(result) + '\n')
  if (!result.ok) {
    // Non-zero exit so CI fails when a data model reaches into rendering.
    process.exitCode = 1
  }
}

// Only run the CLI when invoked directly, not when imported by tests.
if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  main()
}
