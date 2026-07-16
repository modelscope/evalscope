/**
 * Locale key drift checker (Requirement 17.5).
 *
 * Provides a pure function `checkLocaleKeys(base, target)` that computes the
 * symmetric difference between two locale key sets:
 *   - `missing`: keys present in `base` but absent from `target`.
 *   - `extra`:   keys present in `target` but absent from `base`.
 *
 * This check is intentionally self-contained: it depends only on its inputs and
 * never reads token drift state, so it can run independently of the token drift
 * checker and its pass/fail result is unaffected by token drift results
 * (Requirement 17.5).
 */

/**
 * A locale resource: a recursively nested map whose leaves are translation
 * strings. Mirrors the `Dict` shape used by `src/i18n/translations.ts`.
 */
export interface LocaleMap {
  [key: string]: string | LocaleMap
}

/** Symmetric difference between a base and a target locale key set. */
export interface LocaleKeyDiff {
  /** Keys present in `base` but missing from `target`, sorted ascending. */
  missing: string[]
  /** Keys present in `target` but not in `base` (extra), sorted ascending. */
  extra: string[]
}

/**
 * Flatten a nested locale map into the sorted set of its leaf keys expressed in
 * dot notation (e.g. `nav.dashboard`). Only paths that terminate at a string
 * value are emitted; intermediate object nodes are not treated as keys. A path
 * that resolves to a string in one locale but to an object in another therefore
 * yields different leaf keys and surfaces as a difference — which is the desired
 * behavior for locale consistency.
 *
 * @param map    The nested locale map to flatten.
 * @param prefix Internal accumulator for the current dot-notation prefix.
 * @returns Sorted, de-duplicated dot-notation leaf keys.
 */
export function flattenLocaleKeys(map: LocaleMap, prefix = ''): string[] {
  const keys: string[] = []

  for (const key of Object.keys(map)) {
    const value = map[key]
    const path = prefix ? `${prefix}.${key}` : key

    if (typeof value === 'string') {
      keys.push(path)
    } else {
      // Nested map: recurse to collect deeper leaf keys.
      keys.push(...flattenLocaleKeys(value, path))
    }
  }

  // Sort for deterministic, comparable output.
  return keys.sort()
}

/**
 * Compute the symmetric difference between the key sets of two locale maps.
 *
 * @param base   The reference locale map (source of truth for expected keys).
 * @param target The locale map to check against the base.
 * @returns A `LocaleKeyDiff` whose `missing`/`extra` arrays are sorted ascending.
 */
export function checkLocaleKeys(base: LocaleMap, target: LocaleMap): LocaleKeyDiff {
  const baseKeys = new Set(flattenLocaleKeys(base))
  const targetKeys = new Set(flattenLocaleKeys(target))

  const missing = [...baseKeys].filter((key) => !targetKeys.has(key)).sort()
  const extra = [...targetKeys].filter((key) => !baseKeys.has(key)).sort()

  return { missing, extra }
}

// ---------------------------------------------------------------------------
// CLI / CI runner
// ---------------------------------------------------------------------------

import { localeDictionaries } from '../../src/i18n/translations.ts'

/**
 * Run the locale key check against the project's `en` (base) and `zh` (target)
 * locales and print a human-readable report.
 *
 * @returns `true` when the locales are consistent (no missing/extra keys),
 *          `false` otherwise.
 */
export function runLocaleKeyCheck(): boolean {
  const base = localeDictionaries.en as LocaleMap
  const target = localeDictionaries.zh as LocaleMap
  const diff = checkLocaleKeys(base, target)

  const hasDrift = diff.missing.length > 0 || diff.extra.length > 0

  if (!hasDrift) {
    console.log('[locale-key-check] OK: en and zh key sets are consistent.')
    return true
  }

  console.error('[locale-key-check] FAIL: locale key sets diverge.')
  if (diff.missing.length > 0) {
    console.error(`\nMissing in zh (present in en) — ${diff.missing.length}:`)
    for (const key of diff.missing) console.error(`  - ${key}`)
  }
  if (diff.extra.length > 0) {
    console.error(`\nExtra in zh (absent from en) — ${diff.extra.length}:`)
    for (const key of diff.extra) console.error(`  + ${key}`)
  }
  return false
}

// Execute the runner only when this module is invoked directly (not imported),
// exiting non-zero on any drift so it can gate CI.
if (import.meta.url === `file://${process.argv[1]}`) {
  process.exit(runLocaleKeyCheck() ? 0 : 1)
}
