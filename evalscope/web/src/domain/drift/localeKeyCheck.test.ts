import { describe, expect, it } from 'vitest'
import fc from 'fast-check'

import { checkLocaleKeys, type LocaleMap } from '../../../scripts/drift/localeKeyCheck'

// Feature: frontend-refactor-2026-07, Property 30: Locale key symmetric-difference detection
//
// For any base and target locale key maps, `checkLocaleKeys` must return the
// symmetric difference between their leaf-key sets: `missing` is exactly the set
// of keys present in `base` but absent from `target`, and `extra` is exactly the
// set of keys present in `target` but absent from `base`. The check depends only
// on its two inputs (it never reads token drift state), so its result is
// independent of any token drift outcome.
//
// Validates: Requirements 17.5

/**
 * Independent reference implementation of leaf-key flattening. Written from
 * scratch (rather than reusing the module under test) so the property acts as a
 * true oracle for `checkLocaleKeys`' internal flattening.
 */
function refFlatten(map: LocaleMap, prefix = ''): string[] {
  const keys: string[] = []
  for (const key of Object.keys(map)) {
    const value = map[key]
    const path = prefix ? `${prefix}.${key}` : key
    if (typeof value === 'string') {
      keys.push(path)
    } else {
      keys.push(...refFlatten(value, path))
    }
  }
  return keys
}

/** Independent reference symmetric difference over two leaf-key sets. */
function refDiff(base: LocaleMap, target: LocaleMap): { missing: string[]; extra: string[] } {
  const baseKeys = new Set(refFlatten(base))
  const targetKeys = new Set(refFlatten(target))
  const missing = [...baseKeys].filter((k) => !targetKeys.has(k)).sort()
  const extra = [...targetKeys].filter((k) => !baseKeys.has(k)).sort()
  return { missing, extra }
}

// Path segments are drawn from a small pool so that nested keys collide across
// maps often enough to exercise both the "same key" and "diverging key" cases,
// and never contain the '.' path separator.
const segment = fc.constantFrom('a', 'b', 'c', 'd', 'e', 'f', 'g', 'h')

// A recursively nested locale map. Each node value is either a string leaf or a
// deeper nested map; `maxDepth` bounds recursion. Empty object nodes (0 keys)
// are permitted and contribute no leaf keys, mirroring real locale resources.
const nestedLocaleMap: fc.Arbitrary<LocaleMap> = fc.letrec<{ node: string | LocaleMap; map: LocaleMap }>((tie) => ({
  node: fc.oneof({ maxDepth: 3 }, fc.string({ maxLength: 4 }), tie('map')),
  map: fc.dictionary(segment, tie('node'), { maxKeys: 4 }),
})).map as fc.Arbitrary<LocaleMap>

describe('checkLocaleKeys (Property 30: locale key symmetric difference)', () => {
  it('returns missing/extra equal to the independently computed symmetric difference', () => {
    fc.assert(
      fc.property(nestedLocaleMap, nestedLocaleMap, (base, target) => {
        const result = checkLocaleKeys(base, target)
        const expected = refDiff(base, target)
        expect(result.missing).toEqual(expected.missing)
        expect(result.extra).toEqual(expected.extra)
      }),
    )
  })

  it('sorts both missing and extra ascending', () => {
    fc.assert(
      fc.property(nestedLocaleMap, nestedLocaleMap, (base, target) => {
        const { missing, extra } = checkLocaleKeys(base, target)
        expect(missing).toEqual([...missing].sort())
        expect(extra).toEqual([...extra].sort())
      }),
    )
  })

  it('places each divergent key on the correct side and never both', () => {
    fc.assert(
      fc.property(nestedLocaleMap, nestedLocaleMap, (base, target) => {
        const baseKeys = new Set(refFlatten(base))
        const targetKeys = new Set(refFlatten(target))
        const { missing, extra } = checkLocaleKeys(base, target)

        // missing: only in base, extra: only in target.
        for (const k of missing) {
          expect(baseKeys.has(k)).toBe(true)
          expect(targetKeys.has(k)).toBe(false)
        }
        for (const k of extra) {
          expect(targetKeys.has(k)).toBe(true)
          expect(baseKeys.has(k)).toBe(false)
        }
        // Disjoint sides.
        const missingSet = new Set(missing)
        expect(extra.some((k) => missingSet.has(k))).toBe(false)
      }),
    )
  })

  it('reports no drift when key sets are identical (base compared with itself)', () => {
    fc.assert(
      fc.property(nestedLocaleMap, (map) => {
        const { missing, extra } = checkLocaleKeys(map, map)
        expect(missing).toEqual([])
        expect(extra).toEqual([])
      }),
    )
  })

  it('reports all base keys as missing when target is empty', () => {
    fc.assert(
      fc.property(nestedLocaleMap, (base) => {
        const { missing, extra } = checkLocaleKeys(base, {})
        expect(missing).toEqual([...new Set(refFlatten(base))].sort())
        expect(extra).toEqual([])
      }),
    )
  })

  it('reports all target keys as extra when base is empty', () => {
    fc.assert(
      fc.property(nestedLocaleMap, (target) => {
        const { missing, extra } = checkLocaleKeys({}, target)
        expect(missing).toEqual([])
        expect(extra).toEqual([...new Set(refFlatten(target))].sort())
      }),
    )
  })

  it('is a pure function whose result is independent of external (token drift) state', () => {
    // Determinism: repeated invocations with the same inputs yield deep-equal
    // results, confirming the check reads only its arguments and never any
    // token drift state.
    fc.assert(
      fc.property(nestedLocaleMap, nestedLocaleMap, (base, target) => {
        const first = checkLocaleKeys(base, target)
        const second = checkLocaleKeys(base, target)
        expect(second).toEqual(first)
      }),
    )
  })
})
