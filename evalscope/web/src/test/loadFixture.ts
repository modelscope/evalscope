import { readFileSync } from 'node:fs'
import { dirname, join, normalize } from 'node:path'
import { fileURLToPath } from 'node:url'

/**
 * Absolute path to the local fixtures directory.
 *
 * Resolved from this module's own URL so the loader works regardless of the
 * process working directory. Uses `import.meta.url` because the project is an
 * ES module and `__dirname` is not available.
 */
const FIXTURES_DIR = join(dirname(fileURLToPath(import.meta.url)), 'fixtures')

/**
 * Synchronously load a JSON fixture from `src/test/fixtures/`.
 *
 * The loader is intentionally deterministic: it only reads a local JSON file
 * from disk and parses it. It never touches the network, the system clock, or
 * any source of randomness, so tests relying on it produce identical results
 * across runs.
 *
 * @param name Fixture file name relative to the fixtures directory. The
 *   `.json` extension is optional (e.g. both `reports` and `reports.json` work).
 * @returns The parsed fixture content, typed as `T`.
 * @throws If `name` escapes the fixtures directory, or the file is missing or
 *   contains invalid JSON.
 */
export function loadFixture<T>(name: string): T {
  const fileName = name.endsWith('.json') ? name : `${name}.json`
  const filePath = normalize(join(FIXTURES_DIR, fileName))

  // Guard against path traversal so the loader can only read local fixtures.
  if (!filePath.startsWith(FIXTURES_DIR)) {
    throw new Error(`Fixture name resolves outside the fixtures directory: ${name}`)
  }

  const raw = readFileSync(filePath, 'utf-8')
  return JSON.parse(raw) as T
}
