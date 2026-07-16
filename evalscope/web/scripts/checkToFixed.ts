/**
 * Direct-domain `toFixed` guard (Requirement 15.5).
 *
 * Task 4.2 / 4.3 / 12.5 migrated the list, header, overview, detail, compare and
 * performance surfaces to format every metric through the centralized
 * `formatMetric` entry point. This script asserts that the migration stays
 * intact: the number of direct domain `.toFixed(` calls inside the migrated
 * components/pages MUST be 0 (Req 15.5). A regression (someone re-introducing a
 * raw `score.toFixed(2)`) fails CI.
 *
 * The core (`countDirectToFixed`, `scanSource`) is PURE: given source text it
 * returns the `.toFixed(` call sites, ignoring occurrences inside comments and
 * string/template literals so mentions in doc comments (e.g. the JSDoc of
 * `PerfMetricsPanel`) are not counted. A thin file-reading runner
 * ({@link runToFixedCheck}) walks the migrated targets and is wired to a
 * non-zero exit code for CLI/CI use.
 */

import { readdirSync, readFileSync, statSync } from 'node:fs'
import { dirname, relative, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

/** A single direct `.toFixed(` call site found in a source file. */
export interface ToFixedHit {
  /** 1-based line number of the call. */
  line: number
  /** 1-based column of the `.toFixed` token. */
  column: number
}

/** Per-file scan result. */
export interface FileScanResult {
  /** File path as supplied to the scanner (typically web-root-relative). */
  file: string
  /** Every direct `.toFixed(` call site in the file, in source order. */
  hits: ToFixedHit[]
}

/**
 * Strip comments and string/template literals from TypeScript/TSX source,
 * replacing their contents with spaces so line/column offsets are preserved.
 *
 * This is a deliberately small lexical scrubber — not a full parser — but it
 * correctly handles the constructs that would otherwise cause false positives:
 * line comments (`// ...`), block comments (`/* ... *\/`), single/double quoted
 * strings and template literals (including escapes). Regex literals are left
 * intact; a `.toFixed(` never appears inside one so this does not affect the
 * count.
 *
 * @param source Raw source text.
 * @returns Source with comment/string contents blanked out (length preserved).
 */
export function stripCommentsAndStrings(source: string): string {
  const out: string[] = []
  const n = source.length
  let i = 0

  // Blank a character while preserving newlines so line numbers stay aligned.
  const blank = (ch: string): string => (ch === '\n' ? '\n' : ' ')

  while (i < n) {
    const ch = source[i]
    const next = i + 1 < n ? source[i + 1] : ''

    // Line comment: blank to end of line.
    if (ch === '/' && next === '/') {
      while (i < n && source[i] !== '\n') {
        out.push(blank(source[i]))
        i++
      }
      continue
    }

    // Block comment: blank until the closing */.
    if (ch === '/' && next === '*') {
      out.push(' ', ' ')
      i += 2
      while (i < n && !(source[i] === '*' && i + 1 < n && source[i + 1] === '/')) {
        out.push(blank(source[i]))
        i++
      }
      if (i < n) {
        out.push(' ', ' ')
        i += 2
      }
      continue
    }

    // String or template literal: blank contents, keep the delimiters as spaces.
    if (ch === '"' || ch === "'" || ch === '`') {
      const quote = ch
      out.push(' ')
      i++
      while (i < n) {
        const c = source[i]
        if (c === '\\') {
          // Escaped char: blank both the backslash and the escaped char.
          out.push(blank(c))
          if (i + 1 < n) out.push(blank(source[i + 1]))
          i += 2
          continue
        }
        if (c === quote) {
          out.push(' ')
          i++
          break
        }
        out.push(blank(c))
        i++
      }
      continue
    }

    out.push(ch)
    i++
  }

  return out.join('')
}

/**
 * Count direct `.toFixed(` call sites in source text, ignoring occurrences
 * inside comments and string/template literals.
 *
 * Pure and side-effect free: the same input always yields the same hits.
 *
 * @param source Raw source text.
 * @returns The `.toFixed(` call sites, in source order.
 */
export function countDirectToFixed(source: string): ToFixedHit[] {
  const scrubbed = stripCommentsAndStrings(source)
  const hits: ToFixedHit[] = []
  // Match `.toFixed(` allowing whitespace between the token and the paren, but
  // not preceded by an identifier char (so `myToFixed(` does not match).
  const re = /\.toFixed\s*\(/g
  let match: RegExpExecArray | null
  while ((match = re.exec(scrubbed)) !== null) {
    // Guard against an identifier char immediately before the dot, e.g. a
    // property named `xtoFixed` cannot arise, but `foo?.toFixed(` should still
    // match — the char before `.` is `o`/`)`/etc. which is fine.
    const upto = scrubbed.slice(0, match.index)
    const lastNewline = upto.lastIndexOf('\n')
    const line = (upto.match(/\n/g)?.length ?? 0) + 1
    const column = match.index - lastNewline
    hits.push({ line, column })
  }
  return hits
}

/**
 * Scan a single source string, tagging the result with its file path.
 *
 * @param file Path label to attach to the result.
 * @param source Raw source text.
 * @returns The file scan result.
 */
export function scanSource(file: string, source: string): FileScanResult {
  return { file, hits: countDirectToFixed(source) }
}

// ─────────────────────────────────────────────────────────────────────────────
// CLI / CI runner (impure convenience wrapper around the pure core above)
// ─────────────────────────────────────────────────────────────────────────────

/**
 * The migrated directories and files that must contain zero direct domain
 * `toFixed` calls (tasks 4.2 / 4.3 / 12.5), relative to the web project root.
 * Directories are scanned recursively for `.ts` / `.tsx` sources.
 */
export const MIGRATED_TARGETS: readonly string[] = [
  'src/components',
  'src/pages',
]

/** Whether a file should be scanned (source `.ts`/`.tsx`, excluding tests/stories). */
function isScannableSource(path: string): boolean {
  if (!/\.tsx?$/.test(path)) {
    return false
  }
  return !/\.(test|spec|stories)\.tsx?$/.test(path)
}

/**
 * Recursively collect scannable source files under a directory, or the file
 * itself when `target` is a file.
 *
 * @param absPath Absolute path to a file or directory.
 * @returns Absolute paths of scannable source files.
 */
function collectSourceFiles(absPath: string): string[] {
  const stat = statSync(absPath)
  if (stat.isFile()) {
    return isScannableSource(absPath) ? [absPath] : []
  }
  const files: string[] = []
  for (const entry of readdirSync(absPath)) {
    files.push(...collectSourceFiles(resolve(absPath, entry)))
  }
  return files
}

/** Resolve the web project root from this module's location (`scripts/`). */
function resolveWebRoot(): string {
  return resolve(dirname(fileURLToPath(import.meta.url)), '..')
}

/** Options for {@link runToFixedCheck}. */
export interface ToFixedCheckRunOptions {
  /** Web project root. Defaults to the directory above `scripts/`. */
  webRoot?: string
  /** Targets to scan. Defaults to {@link MIGRATED_TARGETS}. */
  targets?: readonly string[]
}

/** Outcome of a runner invocation. */
export interface ToFixedCheckResult {
  /** `true` when no direct `toFixed` calls were found in any target. */
  ok: boolean
  /** Total number of direct `toFixed` call sites across all scanned files. */
  totalHits: number
  /** Only the files that contain at least one hit, with their call sites. */
  violations: FileScanResult[]
  /** Number of source files scanned. */
  scannedFiles: number
}

/**
 * Read the migrated targets from disk and run the pure {@link scanSource} over
 * each source file. Performs no mutation; the caller inspects `ok` to decide how
 * to fail (Req 15.5).
 */
export function runToFixedCheck(options: ToFixedCheckRunOptions = {}): ToFixedCheckResult {
  const webRoot = options.webRoot ?? resolveWebRoot()
  const targets = options.targets ?? MIGRATED_TARGETS

  const violations: FileScanResult[] = []
  let totalHits = 0
  let scannedFiles = 0

  for (const target of targets) {
    const files = collectSourceFiles(resolve(webRoot, target))
    for (const absFile of files) {
      scannedFiles++
      const source = readFileSync(absFile, 'utf8')
      const result = scanSource(relative(webRoot, absFile), source)
      if (result.hits.length > 0) {
        totalHits += result.hits.length
        violations.push(result)
      }
    }
  }

  return { ok: totalHits === 0, totalHits, violations, scannedFiles }
}

/** Render a human-readable report for CLI/CI logs. */
export function formatToFixedReport(result: ToFixedCheckResult): string {
  if (result.ok) {
    return `toFixed check passed: 0 direct domain toFixed calls across ${result.scannedFiles} migrated file(s).`
  }
  const lines = [
    `toFixed check FAILED: ${result.totalHits} direct domain toFixed call(s) in ${result.violations.length} migrated file(s).`,
    '',
    'Route metric formatting through formatMetric / formatMetricByKey instead (Req 15.5).',
    '',
  ]
  for (const violation of result.violations) {
    for (const hit of violation.hits) {
      lines.push(`  • ${violation.file}:${hit.line}:${hit.column}`)
    }
  }
  return lines.join('\n')
}

/** Entry point used when this module is executed directly. */
function main(): void {
  const result = runToFixedCheck()
  process.stdout.write(formatToFixedReport(result) + '\n')
  if (!result.ok) {
    // Non-zero exit so CI fails when a direct domain toFixed call reappears.
    process.exitCode = 1
  }
}

// Only run the CLI when invoked directly, not when imported by tests.
if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  main()
}
