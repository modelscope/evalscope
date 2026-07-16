import { describe, expect, it } from 'vitest'
import fc from 'fast-check'

import {
  buildEmptyStateSpec,
  resolveEmptyState,
  resolveEmptyText,
  type EmptyLocaleMaps,
  type EmptyReason,
} from './emptyState'

// Feature: frontend-refactor-2026-07, Property 12: 空状态 locale 回退
//
// For any empty-state key whose current-locale string is missing (absent or
// blank), locale resolution must fall back to the default (fallback) locale and
// return a non-empty string, and recovery actions must remain usable (their
// `navigateTo` is preserved verbatim by the resolution step). When both the
// current and fallback locales are missing the key, resolution returns the key
// itself, which is non-empty by contract so the UI never renders blank.
//
// Validates: Requirements 6.5
describe('resolveEmptyText / resolveEmptyState (Property 12: empty-state locale fallback)', () => {
  // Locale codes drawn from a small, distinct pool so a "current" and a
  // "fallback" locale can differ.
  const localeCode = fc.constantFrom('zh', 'en', 'ja', 'fr')
  // Non-empty display text: at least one non-whitespace character after trimming.
  const nonEmptyText = fc.string({ minLength: 1, maxLength: 40 }).filter((s) => s.trim().length > 0)
  // Blank text: empty or whitespace-only, i.e. an unusable value in the map.
  const blankText = fc.constantFrom('', ' ', '   ', '\t', '\n', ' \t \n ')
  // A localizable key that is always non-empty by contract.
  const localeKey = fc.string({ minLength: 1, maxLength: 30 }).filter((s) => s.trim().length > 0)
  const reason: fc.Arbitrary<EmptyReason> = fc.constantFrom('no-data', 'load-error', 'no-match')

  it('falls back to the default locale (non-empty) when the current locale lacks the key', () => {
    fc.assert(
      fc.property(
        localeKey,
        localeCode,
        localeCode,
        nonEmptyText,
        // How the current locale "misses" the key: absent, or present-but-blank.
        fc.constantFrom<'absent' | 'blank'>('absent', 'blank'),
        blankText,
        (key, current, fallbackBase, fallbackValue, missKind, blank) => {
          // Ensure the two locales are distinct so the fallback is meaningful.
          fc.pre(current !== fallbackBase)

          const currentMap = missKind === 'blank' ? { [key]: blank } : {}
          const localeMaps: EmptyLocaleMaps = {
            [current]: currentMap,
            [fallbackBase]: { [key]: fallbackValue },
          }

          const resolved = resolveEmptyText(key, current, fallbackBase, localeMaps)
          // The fallback locale's non-empty value is returned.
          expect(resolved).toBe(fallbackValue)
          expect(resolved.trim().length).toBeGreaterThan(0)
        },
      ),
    )
  })

  it('returns the (non-empty) key when both current and fallback locales lack it', () => {
    fc.assert(
      fc.property(
        localeKey,
        localeCode,
        localeCode,
        // Independently decide how each locale misses the key.
        fc.constantFrom<'absent' | 'blank'>('absent', 'blank'),
        fc.constantFrom<'absent' | 'blank'>('absent', 'blank'),
        blankText,
        blankText,
        (key, current, fallbackBase, currentMiss, fallbackMiss, blankA, blankB) => {
          const currentMap = currentMiss === 'blank' ? { [key]: blankA } : {}
          const fallbackMap = fallbackMiss === 'blank' ? { [key]: blankB } : {}
          const localeMaps: EmptyLocaleMaps =
            current === fallbackBase
              ? { [current]: { ...fallbackMap, ...currentMap } }
              : { [current]: currentMap, [fallbackBase]: fallbackMap }

          const resolved = resolveEmptyText(key, current, fallbackBase, localeMaps)
          // Last resort: the key itself, which is non-empty by contract.
          expect(resolved).toBe(key)
          expect(resolved.trim().length).toBeGreaterThan(0)
        },
      ),
    )
  })

  it('resolves a full spec via fallback while preserving every action navigateTo', () => {
    fc.assert(
      fc.property(
        reason,
        localeCode,
        localeCode,
        nonEmptyText,
        (reasonValue, current, fallbackBase, fallbackValue) => {
          fc.pre(current !== fallbackBase)

          const spec = buildEmptyStateSpec(reasonValue, { view: 'reports' })

          // Build a fallback-only locale map: the current locale is entirely
          // empty, so every key must fall back to `fallbackBase`, which maps the
          // message key and every action label key to a non-empty value.
          const fallbackMap: Record<string, string> = { [spec.messageKey]: fallbackValue }
          for (const action of spec.actions) {
            fallbackMap[action.labelKey] = fallbackValue
          }
          const localeMaps: EmptyLocaleMaps = {
            [current]: {},
            [fallbackBase]: fallbackMap,
          }

          const resolved = resolveEmptyState(spec, current, fallbackBase, localeMaps)

          // Message falls back to a non-empty string.
          expect(resolved.message.trim().length).toBeGreaterThan(0)
          expect(resolved.reason).toBe(reasonValue)

          // Recovery stays available: same number of actions, each label
          // non-empty, and each navigateTo preserved verbatim in order.
          expect(resolved.actions).toHaveLength(spec.actions.length)
          resolved.actions.forEach((resolvedAction, index) => {
            const specAction = spec.actions[index]
            expect(resolvedAction.navigateTo).toBe(specAction.navigateTo)
            expect(resolvedAction.navigateTo.trim().length).toBeGreaterThan(0)
            expect(resolvedAction.label.trim().length).toBeGreaterThan(0)
          })
        },
      ),
    )
  })

  it('preserves navigateTo even when no locale provides any string (labels fall back to keys)', () => {
    fc.assert(
      fc.property(reason, localeCode, localeCode, (reasonValue, current, fallbackBase) => {
        const spec = buildEmptyStateSpec(reasonValue, { view: 'performance' })
        // Empty locale maps: nothing resolves, so labels fall back to keys and
        // the message falls back to its key. Actions must still be navigable.
        const localeMaps: EmptyLocaleMaps = {}

        const resolved = resolveEmptyState(spec, current, fallbackBase, localeMaps)

        expect(resolved.message).toBe(spec.messageKey)
        expect(resolved.actions).toHaveLength(spec.actions.length)
        resolved.actions.forEach((resolvedAction, index) => {
          expect(resolvedAction.navigateTo).toBe(spec.actions[index].navigateTo)
          expect(resolvedAction.navigateTo.trim().length).toBeGreaterThan(0)
          // Label falls back to the (non-empty) key.
          expect(resolvedAction.label).toBe(spec.actions[index].labelKey)
        })
      }),
    )
  })
})
