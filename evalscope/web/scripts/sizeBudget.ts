/**
 * Initial-load chunk size budget guard (Requirement 16.6).
 *
 * Requirement 16.6 mandates that the initial-load JavaScript chunks stay within
 * a 150 KB gzip budget, verified against the real build output. This script is
 * run *after* `vite build` (locally via `npm run size-budget`, and in CI right
 * after the build step — wired by task 19.5).
 *
 * ## How the "initial load" set is identified
 *
 * Vite/Rollup split the app into chunks. Only a subset is fetched on first
 * paint: the entry chunk plus every chunk it (transitively) imports *statically*.
 * Chunks that are only reached through a dynamic `import()` (React.lazy, route
 * level code-splitting, …) are downloaded on demand and therefore excluded.
 *
 * Rather than fragile HTML/regex scraping, we rely on Vite's build manifest
 * (`dist/.vite/manifest.json`, enabled via `build.manifest` in `vite.config.ts`).
 * Each manifest record exposes:
 *   - `isEntry`       — the record is an HTML entry point,
 *   - `imports`       — keys of statically-imported chunks (part of initial load),
 *   - `dynamicImports`— keys of lazily-imported chunks (NOT part of initial load),
 *   - `file` / `css`  — emitted asset paths.
 *
 * We start from every `isEntry` record and walk `imports` transitively (never
 * `dynamicImports`), collecting the emitted `.js` files. Their combined gzip
 * size is compared against the budget.
 *
 * The core (`collectInitialChunkFiles`, `evaluateBudget`, `gzipSize`, …) is
 * PURE — it operates on a parsed manifest plus an injectable byte reader, so it
 * is fully testable without a real build. A thin runner ({@link runSizeBudget})
 * reads `dist/` from disk and wires the outcome to a non-zero exit code.
 */

import { gzipSync } from 'node:zlib'
import { existsSync, readFileSync } from 'node:fs'
import { dirname, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

/** Default initial-load gzip budget in bytes (150 KB, Req 16.6). */
export const DEFAULT_BUDGET_BYTES = 150 * 1024

/** A single record in a Vite build manifest (`.vite/manifest.json`). */
export interface ViteManifestChunk {
  /** Emitted file path, relative to the build output directory. */
  file: string
  /** `true` when this record is an HTML/JS entry point. */
  isEntry?: boolean
  /** `true` for chunks that are the target of a dynamic import. */
  isDynamicEntry?: boolean
  /** Manifest keys of statically-imported chunks (part of initial load). */
  imports?: string[]
  /** Manifest keys of dynamically-imported chunks (excluded from initial load). */
  dynamicImports?: string[]
  /** Emitted CSS asset paths for this chunk. */
  css?: string[]
  /** Original source path (informational). */
  src?: string
  /** Chunk name (informational). */
  name?: string
}

/** A parsed Vite build manifest: manifest key -> chunk record. */
export type ViteManifest = Record<string, ViteManifestChunk>

/** Measurement for one emitted chunk file. */
export interface ChunkMeasurement {
  /** Emitted file path, relative to the build output directory. */
  file: string
  /** Uncompressed size in bytes. */
  rawBytes: number
  /** Gzip-compressed size in bytes. */
  gzipBytes: number
}

/** Outcome of a budget evaluation. */
export interface BudgetResult {
  /** `true` when the total initial-load gzip size is within budget. */
  ok: boolean
  /** The budget the total was checked against, in bytes. */
  budgetBytes: number
  /** Sum of the gzip sizes of all initial-load chunk files, in bytes. */
  totalGzipBytes: number
  /** Sum of the raw sizes of all initial-load chunk files, in bytes. */
  totalRawBytes: number
  /** Per-file measurements, sorted by descending gzip size. */
  chunks: ChunkMeasurement[]
}

// ─────────────────────────────────────────────────────────────────────────────
// Pure core
// ─────────────────────────────────────────────────────────────────────────────

/** Gzip-compressed size, in bytes, of the given data. */
export function gzipSize(data: Buffer | Uint8Array | string): number {
  return gzipSync(data).length
}

/**
 * Collect the manifest keys that belong to the initial-load set: every entry
 * record plus the chunks reachable through *static* `imports` (transitively).
 * `dynamicImports` are never followed, so lazy chunks are excluded.
 *
 * Pure and deterministic; unknown keys (dangling references) are skipped.
 *
 * @param manifest Parsed Vite build manifest.
 * @returns Initial-load manifest keys, in deterministic (sorted) order.
 */
export function collectInitialChunkKeys(manifest: ViteManifest): string[] {
  const visited = new Set<string>()
  const queue: string[] = []

  // Seed with every entry point.
  for (const [key, chunk] of Object.entries(manifest)) {
    if (chunk.isEntry) {
      queue.push(key)
    }
  }

  while (queue.length > 0) {
    const key = queue.shift() as string
    if (visited.has(key)) {
      continue
    }
    const chunk = manifest[key]
    if (!chunk) {
      // Dangling reference (should not happen in a valid manifest) — skip.
      continue
    }
    visited.add(key)
    // Follow ONLY static imports; dynamic imports are lazy chunks.
    for (const dep of chunk.imports ?? []) {
      if (!visited.has(dep)) {
        queue.push(dep)
      }
    }
  }

  return [...visited].sort()
}

/**
 * Resolve the emitted `.js` files for the initial-load chunk set.
 *
 * @param manifest Parsed Vite build manifest.
 * @returns Deduplicated, sorted list of initial-load JS file paths.
 */
export function collectInitialChunkFiles(manifest: ViteManifest): string[] {
  const files = new Set<string>()
  for (const key of collectInitialChunkKeys(manifest)) {
    const file = manifest[key]?.file
    if (file && file.endsWith('.js')) {
      files.add(file)
    }
  }
  return [...files].sort()
}

/**
 * Measure the raw and gzip size of each initial-load chunk file.
 *
 * The byte reader is injected so the core stays pure and testable without a
 * real build: the runner supplies a disk-backed reader.
 *
 * @param manifest Parsed Vite build manifest.
 * @param readBytes Reader mapping an emitted file path to its bytes.
 * @returns Per-file measurements, sorted by descending gzip size.
 */
export function measureInitialChunks(
  manifest: ViteManifest,
  readBytes: (file: string) => Buffer | Uint8Array,
): ChunkMeasurement[] {
  const measurements = collectInitialChunkFiles(manifest).map((file): ChunkMeasurement => {
    const bytes = readBytes(file)
    return { file, rawBytes: bytes.length, gzipBytes: gzipSize(bytes) }
  })
  measurements.sort((a, b) => b.gzipBytes - a.gzipBytes)
  return measurements
}

/**
 * Evaluate per-file measurements against the gzip budget.
 *
 * @param chunks Per-file measurements.
 * @param budgetBytes Gzip budget in bytes. Defaults to {@link DEFAULT_BUDGET_BYTES}.
 * @returns The budget result; `ok` is `false` when the total exceeds the budget.
 */
export function evaluateBudget(
  chunks: ChunkMeasurement[],
  budgetBytes: number = DEFAULT_BUDGET_BYTES,
): BudgetResult {
  const totalGzipBytes = chunks.reduce((sum, c) => sum + c.gzipBytes, 0)
  const totalRawBytes = chunks.reduce((sum, c) => sum + c.rawBytes, 0)
  return {
    ok: totalGzipBytes <= budgetBytes,
    budgetBytes,
    totalGzipBytes,
    totalRawBytes,
    chunks,
  }
}

/** Format a byte count as a human-readable KB string. */
export function formatKb(bytes: number): string {
  return `${(bytes / 1024).toFixed(2)} KB`
}

/**
 * Render a human-readable budget report for CLI/CI logs.
 *
 * @param result The budget result.
 * @returns Multi-line report string.
 */
export function formatBudgetReport(result: BudgetResult): string {
  const header = result.ok
    ? `Size budget passed: initial-load gzip ${formatKb(result.totalGzipBytes)} <= ${formatKb(result.budgetBytes)} (${result.chunks.length} chunk(s)).`
    : `Size budget FAILED: initial-load gzip ${formatKb(result.totalGzipBytes)} > ${formatKb(result.budgetBytes)} (${result.chunks.length} chunk(s)).`

  const lines = [header, '']
  for (const chunk of result.chunks) {
    lines.push(`  • ${chunk.file}  gzip ${formatKb(chunk.gzipBytes)} (raw ${formatKb(chunk.rawBytes)})`)
  }
  if (!result.ok) {
    lines.push(
      '',
      'Reduce the initial bundle: code-split heavy views behind dynamic import()/React.lazy,',
      'drop or lazy-load large dependencies, or move non-critical work off the entry chunk (Req 16.6).',
    )
  }
  return lines.join('\n')
}

// ─────────────────────────────────────────────────────────────────────────────
// CLI / CI runner (impure wrapper around the pure core above)
// ─────────────────────────────────────────────────────────────────────────────

/** Resolve the web project root from this module's location (`scripts/`). */
function resolveWebRoot(): string {
  return resolve(dirname(fileURLToPath(import.meta.url)), '..')
}

/** Options for {@link runSizeBudget}. */
export interface SizeBudgetRunOptions {
  /** Web project root. Defaults to the directory above `scripts/`. */
  webRoot?: string
  /** Build output directory name (relative to `webRoot`). Defaults to `dist`. */
  distDir?: string
  /** Gzip budget in bytes. Defaults to {@link DEFAULT_BUDGET_BYTES}. */
  budgetBytes?: number
}

/** A run outcome that could not produce a budget result (e.g. missing build). */
export interface SizeBudgetRunError {
  ok: false
  /** Discriminant so callers can distinguish this from a {@link BudgetResult}. */
  kind: 'error'
  /** Human-readable explanation for CLI/CI logs. */
  message: string
}

/** Locate the Vite manifest inside the build output directory. */
function resolveManifestPath(distPath: string): string | null {
  // Vite 5+ writes the manifest to `<outDir>/.vite/manifest.json`; older
  // versions wrote it to `<outDir>/manifest.json`. Support both.
  const candidates = [resolve(distPath, '.vite', 'manifest.json'), resolve(distPath, 'manifest.json')]
  return candidates.find((candidate) => existsSync(candidate)) ?? null
}

/**
 * Read `dist/` from disk, measure the initial-load chunks and evaluate them
 * against the budget. Performs no mutation; the caller inspects the result to
 * decide how to fail (Req 16.6).
 *
 * @returns A {@link BudgetResult} when the build output is present, otherwise a
 *   {@link SizeBudgetRunError} describing what is missing.
 */
export function runSizeBudget(options: SizeBudgetRunOptions = {}): BudgetResult | SizeBudgetRunError {
  const webRoot = options.webRoot ?? resolveWebRoot()
  const distDir = options.distDir ?? 'dist'
  const budgetBytes = options.budgetBytes ?? DEFAULT_BUDGET_BYTES
  const distPath = resolve(webRoot, distDir)

  if (!existsSync(distPath)) {
    return {
      ok: false,
      kind: 'error',
      message: `Build output not found at "${distPath}". Run \`npm run build\` before \`npm run size-budget\`.`,
    }
  }

  const manifestPath = resolveManifestPath(distPath)
  if (!manifestPath) {
    return {
      ok: false,
      kind: 'error',
      message:
        `Vite manifest not found under "${distPath}". Ensure \`build.manifest\` is enabled in vite.config.ts ` +
        'and re-run `npm run build`.',
    }
  }

  const manifest = JSON.parse(readFileSync(manifestPath, 'utf8')) as ViteManifest
  const chunks = measureInitialChunks(manifest, (file) => readFileSync(resolve(distPath, file)))
  return evaluateBudget(chunks, budgetBytes)
}

/** Type guard distinguishing a run error from a budget result. */
function isRunError(result: BudgetResult | SizeBudgetRunError): result is SizeBudgetRunError {
  return 'kind' in result && result.kind === 'error'
}

/** Entry point used when this module is executed directly. */
function main(): void {
  const result = runSizeBudget()
  if (isRunError(result)) {
    process.stderr.write(result.message + '\n')
    // Non-zero exit: a missing/invalid build is a failure in the CI pipeline.
    process.exitCode = 1
    return
  }
  process.stdout.write(formatBudgetReport(result) + '\n')
  if (!result.ok) {
    // Non-zero exit so CI fails when the initial-load budget is exceeded.
    process.exitCode = 1
  }
}

// Only run the CLI when invoked directly, not when imported by tests.
if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  main()
}
