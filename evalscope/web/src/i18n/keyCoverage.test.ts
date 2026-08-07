/**
 * Static guards on the translation dictionaries.
 *
 * `lookupTranslation` returns the lookup path itself when a key is absent, so a missing key does not
 * throw, does not warn, and does not fail a render test that only checks structure -- the UI simply
 * displays `dashboard.spread` where a word belongs. That has already happened once in this codebase,
 * when a component reached for `col.metric`, a key belonging to the standalone HTML report's own
 * dictionary rather than to this app's.
 *
 * These assertions read the source tree because the property being protected is the absence of a
 * mismatch between what components ask for and what the dictionaries define.
 */

import { describe, expect, it } from 'vitest'
import { readFileSync, readdirSync, statSync } from 'node:fs'
import { join, relative } from 'node:path'

import { localeDictionaries, lookupTranslation } from './translations'
import type { Dict } from './translations'

const SRC_ROOT = join(__dirname, '..')

/** Test files may reference keys that do not exist, to assert the fallback behaviour itself. */
const SKIPPED_SUFFIXES = ['.test.ts', '.test.tsx', '__arbitraries__.ts']

function sourceFiles(dir: string): string[] {
  const files: string[] = []
  for (const entry of readdirSync(dir)) {
    const full = join(dir, entry)
    if (statSync(full).isDirectory()) {
      if (entry === 'node_modules' || entry === 'test') continue
      files.push(...sourceFiles(full))
      continue
    }
    if (!/\.tsx?$/.test(entry)) continue
    if (SKIPPED_SUFFIXES.some((suffix) => entry.endsWith(suffix))) continue
    files.push(full)
  }
  return files
}

/**
 * Collect every fully literal translation key used in the source tree.
 *
 * Keys assembled at runtime -- `` t(`dashboard.sort_${key}`) `` -- cannot be resolved statically and
 * are not collected. Their namespaces are covered by the dictionary-parity check below, which
 * compares whole key sets rather than individual lookups.
 */
function literalKeysInSource(): Map<string, string[]> {
  const keys = new Map<string, string[]>()
  for (const file of sourceFiles(SRC_ROOT)) {
    const content = readFileSync(file, 'utf8')
    // `t('namespace.key')`, single or double quoted, optionally followed by interpolation vars.
    for (const match of content.matchAll(/\bt\(\s*['"]([\w.]+)['"]/g)) {
      const key = match[1]
      const where = keys.get(key) ?? []
      where.push(relative(SRC_ROOT, file))
      keys.set(key, where)
    }
  }
  return keys
}

describe('translation keys referenced by components exist', () => {
  it('resolves every literal key to a translation, not to the key itself', () => {
    const unresolved: string[] = []
    for (const [key, files] of literalKeysInSource()) {
      if (lookupTranslation('en', key) === key) {
        unresolved.push(`${key}  (${files.join(', ')})`)
      }
    }
    expect(unresolved).toEqual([])
  })

  it('does not reach for the standalone HTML report dictionary', () => {
    // `col.*` and `card.*` belong to evalscope/report/template/js/i18n_eval.js. Those render fine in
    // the generated HTML report and as raw keys here, which is why this needs its own check: such a
    // key would satisfy the test above only by coincidence of naming.
    const offenders: string[] = []
    for (const file of sourceFiles(SRC_ROOT)) {
      if (/\bt\(\s*['"](col|card)\./.test(readFileSync(file, 'utf8'))) {
        offenders.push(relative(SRC_ROOT, file))
      }
    }
    expect(offenders).toEqual([])
  })
})

describe('dictionaries stay in step with each other', () => {
  it('defines the same keys in English and Chinese', () => {
    // `lookupTranslation` falls back to English before giving up, so a key missing from `zh` shows
    // English text rather than a raw key. That is a reasonable degradation and therefore invisible,
    // which is exactly why it needs a test.
    const flatten = (dict: Dict, prefix = ''): string[] =>
      Object.entries(dict).flatMap(([key, value]) =>
        typeof value === 'string' ? [`${prefix}${key}`] : flatten(value as Dict, `${prefix}${key}.`),
      )

    const en = flatten(localeDictionaries.en).sort()
    const zh = flatten(localeDictionaries.zh).sort()

    expect(en.filter((key) => !zh.includes(key))).toEqual([])
    expect(zh.filter((key) => !en.includes(key))).toEqual([])
  })
})
