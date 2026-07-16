/**
 * Design document structure check (Req 17.1).
 *
 * The design document (`DESIGN.md`) must be split into five separately
 * addressable sections — `principles`, `tokens`, `components`, `metrics` and
 * `decisions`. Each section is "addressable" when the Markdown exposes an
 * anchor that resolves to the section's id, so a link such as `DESIGN.md#tokens`
 * lands on it. IF any of the five sections is absent, THE structure check fails.
 *
 * `checkStructure` is a PURE function: given the raw Markdown text it returns
 * which required sections are present and which are missing. It performs no I/O
 * and no mutation, so it is trivial to unit-test. A thin file-reading runner
 * ({@link runStructureCheck}) is exported for CLI/CI use.
 *
 * Addressability is detected from three anchor sources, in order of intent:
 *   1. Explicit heading anchors — `## Design Tokens {#tokens}` (the canonical,
 *      unambiguous mechanism used by `DESIGN.md`).
 *   2. Inline HTML anchors — `<a id="tokens">` / `<a name="tokens">`.
 *   3. GitHub-style slug of a heading's text — so a bare `## Tokens` heading is
 *      still addressable as `#tokens` even without an explicit anchor.
 */

import { readFileSync } from 'node:fs'
import { dirname, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

/** A required, separately addressable section of the design document. */
export interface RequiredSection {
  /** Canonical anchor id the section must be reachable by (e.g. `tokens`). */
  readonly id: string
  /** Human-readable label for reports. */
  readonly label: string
}

/**
 * The five sections `DESIGN.md` must expose as separately addressable chapters
 * (Req 17.1), in canonical order.
 */
export const REQUIRED_SECTIONS: readonly RequiredSection[] = [
  { id: 'principles', label: 'Principles' },
  { id: 'tokens', label: 'Design Tokens' },
  { id: 'components', label: 'Components' },
  { id: 'metrics', label: 'Metric Display Contract' },
  { id: 'decisions', label: 'Decision Records' },
]

/** Outcome of a structure check. */
export interface StructureCheckResult {
  /** `true` when every required section is present. */
  ok: boolean
  /** Ids of required sections that could not be found, in canonical order. */
  missing: string[]
  /** Ids of required sections that were found, in canonical order. */
  found: string[]
}

/**
 * Slugify heading text the way GitHub renders anchor ids: lower-case, drop
 * anything that is not a word char, space or hyphen, then collapse runs of
 * whitespace into single hyphens. Good enough to make a bare `## Tokens`
 * heading resolve to `#tokens`.
 *
 * @param text Raw heading text (anchor suffixes already removed).
 * @returns The GitHub-style slug.
 */
function slugify(text: string): string {
  return text
    .trim()
    .toLowerCase()
    .replace(/[^\w\s-]/g, '')
    .replace(/\s+/g, '-')
    .replace(/-+/g, '-')
    .replace(/^-|-$/g, '')
}

/**
 * Collect every anchor id the Markdown exposes: explicit heading anchors,
 * inline HTML anchors, and slugified heading text.
 *
 * @param md Raw Markdown text.
 * @returns The set of addressable anchor ids.
 */
function collectAnchors(md: string): Set<string> {
  const anchors = new Set<string>()
  const lines = md.split(/\r?\n/)
  const headingRe = /^#{1,6}\s+(.*)$/
  const explicitAnchorRe = /\{#([\w-]+)\}/g

  for (const line of lines) {
    const heading = line.match(headingRe)
    if (heading === null) {
      continue
    }
    let text = heading[1]

    // 1. Explicit `{#id}` anchors on the heading (there may be more than one).
    let match: RegExpExecArray | null
    explicitAnchorRe.lastIndex = 0
    while ((match = explicitAnchorRe.exec(text)) !== null) {
      anchors.add(match[1].toLowerCase())
    }

    // 3. GitHub-style slug of the heading text, with any `{#id}` suffix stripped.
    text = text.replace(explicitAnchorRe, '').trim()
    const slug = slugify(text)
    if (slug !== '') {
      anchors.add(slug)
    }
  }

  // 2. Inline HTML anchors anywhere in the document.
  const htmlAnchorRe = /<a\s+(?:id|name)\s*=\s*["']([\w-]+)["']/gi
  let htmlMatch: RegExpExecArray | null
  while ((htmlMatch = htmlAnchorRe.exec(md)) !== null) {
    anchors.add(htmlMatch[1].toLowerCase())
  }

  return anchors
}

/**
 * Check that the design document exposes every required addressable section.
 *
 * Pure and side-effect free (Req 17.1): the caller decides how to fail based on
 * the returned `ok` / `missing` fields.
 *
 * @param md Raw Markdown text (typically `DESIGN.md`).
 * @param required Required sections to look for. Defaults to {@link REQUIRED_SECTIONS}.
 * @returns Which required sections are present and which are missing.
 */
export function checkStructure(
  md: string,
  required: readonly RequiredSection[] = REQUIRED_SECTIONS,
): StructureCheckResult {
  const anchors = collectAnchors(md)
  const missing: string[] = []
  const found: string[] = []

  for (const section of required) {
    if (anchors.has(section.id.toLowerCase())) {
      found.push(section.id)
    } else {
      missing.push(section.id)
    }
  }

  return { ok: missing.length === 0, missing, found }
}

// ─────────────────────────────────────────────────────────────────────────────
// CLI / CI runner (impure convenience wrapper around the pure check above)
// ─────────────────────────────────────────────────────────────────────────────

/** Options for {@link runStructureCheck}. */
export interface StructureCheckRunOptions {
  /** Path to the design document. Defaults to the repo-root `DESIGN.md`. */
  mdPath?: string
  /** Required sections to check for. Defaults to {@link REQUIRED_SECTIONS}. */
  required?: readonly RequiredSection[]
}

/** Result of a runner invocation: the resolved path plus the check result. */
export interface StructureCheckRunResult extends StructureCheckResult {
  mdPath: string
}

/** Resolve the web project root from this module's location (`scripts/drift`). */
function resolveWebRoot(): string {
  return resolve(dirname(fileURLToPath(import.meta.url)), '..', '..')
}

/**
 * Read the design document from disk and run the pure {@link checkStructure}.
 * Performs no mutation; the caller inspects `ok` / `missing` to decide how to
 * fail.
 */
export function runStructureCheck(options: StructureCheckRunOptions = {}): StructureCheckRunResult {
  const webRoot = resolveWebRoot()
  const mdPath = options.mdPath ?? resolve(webRoot, '..', '..', 'DESIGN.md')
  const md = readFileSync(mdPath, 'utf8')
  const result = checkStructure(md, options.required ?? REQUIRED_SECTIONS)
  return { mdPath, ...result }
}

/** Render a human-readable structure report for CLI/CI logs. */
export function formatStructureReport(result: StructureCheckRunResult): string {
  if (result.ok) {
    return `Structure check passed: all ${result.found.length} required section(s) are addressable in DESIGN.md.`
  }
  const lines = [`Structure check FAILED: ${result.missing.length} required section(s) missing from DESIGN.md.`, '']
  for (const id of result.missing) {
    lines.push(`  • missing addressable section: #${id}`)
  }
  lines.push('', `  found: ${result.found.map((id) => `#${id}`).join(', ') || '(none)'}`)
  return lines.join('\n')
}

/** Entry point used when this module is executed directly (`node structureCheck.ts`). */
function main(): void {
  const result = runStructureCheck()
  process.stdout.write(formatStructureReport(result) + '\n')
  if (!result.ok) {
    // Non-zero exit so CI fails when a required section is absent (Req 17.1).
    process.exitCode = 1
  }
}

// Only run the CLI when invoked directly, not when imported by tests.
if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  main()
}
