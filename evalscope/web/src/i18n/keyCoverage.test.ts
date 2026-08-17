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

// The flattening and the symmetric difference are owned by the drift checker; reimplementing them
// here would give the same rule two definitions that could disagree.
import { checkLocaleKeys, flattenLocaleKeys, type LocaleMap } from '../../scripts/drift/localeKeyCheck'
import { localeDictionaries, lookupTranslation } from './translations'

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
 * Covers `t('a.b')` and the `labelKey` prop / field, which `Field` and the tab
 * descriptors resolve through the same dictionary.
 *
 * Keys assembled at runtime -- `` t(`dashboard.sort_${key}`) -- cannot be
 * resolved this way; `literalPrefixesInSource` covers those instead.
 */
function literalKeysInSource(): Map<string, string[]> {
  const keys = new Map<string, string[]>()
  const patterns = [
    // `t('namespace.key')`, single or double quoted, optionally followed by interpolation vars.
    /\bt\(\s*['"]([\w.]+)['"]/g,
    // `labelKey="namespace.key"` as a JSX prop, and `labelKey: 'namespace.key'` in a descriptor.
    /\blabelKey[=:]\s*['"]([\w.]+)['"]/g,
  ]
  for (const file of sourceFiles(SRC_ROOT)) {
    const content = readFileSync(file, 'utf8')
    for (const pattern of patterns) {
      for (const match of content.matchAll(pattern)) {
        const key = match[1]
        const where = keys.get(key) ?? []
        where.push(relative(SRC_ROOT, file))
        keys.set(key, where)
      }
    }
  }
  return keys
}

/**
 * Collect the static prefix of every runtime-assembled key.
 *
 * `` t(`perf.archive.status_${s}`) `` yields `perf.archive.status_`. The suffix is
 * unknowable statically, but the prefix is not: if no declared key begins with it,
 * every branch of that lookup renders a raw path. This is the gap a namespace
 * rename falls into -- the en/zh parity check below cannot see it, because a
 * rename applied to both locales keeps them in perfect step with each other while
 * both drift away from the call site.
 */
function literalPrefixesInSource(): Map<string, string[]> {
  const prefixes = new Map<string, string[]>()
  for (const file of sourceFiles(SRC_ROOT)) {
    const content = readFileSync(file, 'utf8')
    for (const match of content.matchAll(/\bt\(\s*`([\w.]*[\w.])\$\{/g)) {
      const prefix = match[1]
      // A prefix with no dot is a whole-namespace lookup; there is nothing to check.
      if (!prefix.includes('.')) continue
      const where = prefixes.get(prefix) ?? []
      where.push(relative(SRC_ROOT, file))
      prefixes.set(prefix, where)
    }
  }
  return prefixes
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

  it('resolves the static prefix of every runtime-assembled key', () => {
    const declared = flattenLocaleKeys(localeDictionaries.en as LocaleMap)

    const orphaned: string[] = []
    for (const [prefix, files] of literalPrefixesInSource()) {
      if (!declared.some((key) => key.startsWith(prefix))) {
        orphaned.push(`${prefix}*  (${files.join(', ')})`)
      }
    }
    expect(orphaned).toEqual([])
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
    // which is exactly why it needs a test. `checkLocaleKeys` is exercised against synthetic maps in
    // `scripts/drift/localeKeyCheck.test.ts`; this applies it to the dictionaries actually shipped.
    expect(checkLocaleKeys(localeDictionaries.en as LocaleMap, localeDictionaries.zh as LocaleMap)).toEqual({
      missing: [],
      extra: [],
    })
  })
})

describe('every declared key is reachable', () => {
  it('declares no key that no call site can ask for', () => {
    // The checks above run one way: a key a component asks for must exist. The
    // other way round is just as silent -- a declared key nothing asks for is
    // dead weight that reads as intentional, and translators keep paying for it.
    //
    // Limited by construction to keys outside a runtime-assembled prefix: under
    // `` t(`trace.submitSource.${source}`) `` any suffix might be produced, so
    // no static rule can call one dead. Those groups are only verifiable against
    // the producing enum on the Python side, which this test cannot see.
    const declared = flattenLocaleKeys(localeDictionaries.en as LocaleMap)
    const prefixes = [...literalPrefixesInSource().keys()]
    // Reachability is a weaker question than the checks above: a key counts as
    // asked for if it appears as a string anywhere, since call sites also carry
    // keys through descriptors, maps and variables rather than only `t('a.b')`.
    const namespaces = new Set(declared.map((key) => key.split('.')[0]))
    const referenced = new Set<string>()
    for (const file of sourceFiles(SRC_ROOT)) {
      const content = readFileSync(file, 'utf8')
      for (const match of content.matchAll(/['"`]([A-Za-z_]\w*(?:\.[A-Za-z0-9_]+)+)['"`]/g)) {
        if (namespaces.has(match[1].split('.')[0])) referenced.add(match[1])
      }
    }

    const unreachable = declared.filter(
      (key) => !referenced.has(key) && !prefixes.some((prefix) => key.startsWith(prefix)),
    )
    expect(unreachable).toEqual([])
  })
})
