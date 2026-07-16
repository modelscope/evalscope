import { describe, expect, it } from 'vitest'
import fc from 'fast-check'

import {
  MAX_EMPTY_ACTIONS,
  MIN_EMPTY_ACTIONS,
  buildEmptyStateSpec,
  type EmptyReason,
  type EmptyStateAction,
  type EmptyStateContext,
  type EmptyStateView,
} from './emptyState'

// Feature: frontend-refactor-2026-07, Property 11: 空状态动作数量与可导航性
//
// For any empty-state reason (and any optional view context — including one that
// supplies blank, duplicate or otherwise invalid extra actions), the spec built
// by buildEmptyStateSpec must expose between MIN_EMPTY_ACTIONS (1) and
// MAX_EMPTY_ACTIONS (3) recovery actions, and every action must carry a
// non-empty in-product navigateTo (non-blank after trimming). This guarantees
// the user is never stranded on an empty state without a navigable way out.
//
// Validates: Requirements 6.2
describe('buildEmptyStateSpec (Property 11: action count & navigability)', () => {
  const reasonArb: fc.Arbitrary<EmptyReason> = fc.constantFrom('no-data', 'load-error', 'no-match')

  const viewArb: fc.Arbitrary<EmptyStateView> = fc.constantFrom(
    'dashboard',
    'reports',
    'evaluations',
    'compare',
    'performance',
    'perf-compare',
    'benchmarks',
  )

  // Routes may be blank / whitespace to exercise the sanitize path that drops
  // invalid navigateTo targets and falls back to safe in-product routes.
  const routeArb: fc.Arbitrary<string> = fc.oneof(
    fc.constant(''),
    fc.constant('   '),
    fc.string({ maxLength: 12 }),
    fc.constantFrom('/tasks', '/reports', '/compare', '/dashboard', '/tasks?tab=eval'),
  )

  // Extra actions intentionally include blank and duplicate navigateTo values so
  // that de-duplication and empty-filtering in sanitizeActions are exercised.
  const extraActionArb: fc.Arbitrary<EmptyStateAction> = fc.record({
    labelKey: fc.string({ maxLength: 12 }),
    navigateTo: fc.oneof(
      fc.constant(''),
      fc.constant('   '),
      // A small pool of routes maximizes the chance of duplicates across actions.
      fc.constantFrom('/tasks', '/reports', '/compare', '/dashboard'),
      fc.string({ maxLength: 12 }),
    ),
  })

  const contextArb: fc.Arbitrary<EmptyStateContext> = fc.record(
    {
      view: viewArb,
      retryTo: routeArb,
      clearFiltersTo: routeArb,
      createTaskTo: routeArb,
      extraActions: fc.array(extraActionArb, { maxLength: 6 }),
    },
    { requiredKeys: [] },
  )

  it('produces 1..3 actions each with a non-empty navigateTo (with context)', () => {
    fc.assert(
      fc.property(reasonArb, contextArb, (reason, context) => {
        const spec = buildEmptyStateSpec(reason, context)

        expect(spec.reason).toBe(reason)
        expect(spec.actions.length).toBeGreaterThanOrEqual(MIN_EMPTY_ACTIONS)
        expect(spec.actions.length).toBeLessThanOrEqual(MAX_EMPTY_ACTIONS)

        for (const action of spec.actions) {
          expect(typeof action.navigateTo).toBe('string')
          expect(action.navigateTo.trim().length).toBeGreaterThan(0)
        }
      }),
    )
  })

  it('produces 1..3 navigable actions for any reason without context', () => {
    fc.assert(
      fc.property(reasonArb, (reason) => {
        const spec = buildEmptyStateSpec(reason)

        expect(spec.actions.length).toBeGreaterThanOrEqual(MIN_EMPTY_ACTIONS)
        expect(spec.actions.length).toBeLessThanOrEqual(MAX_EMPTY_ACTIONS)
        for (const action of spec.actions) {
          expect(action.navigateTo.trim().length).toBeGreaterThan(0)
        }
      }),
    )
  })

  it('never emits duplicate navigateTo targets', () => {
    fc.assert(
      fc.property(reasonArb, contextArb, (reason, context) => {
        const targets = buildEmptyStateSpec(reason, context).actions.map((a) => a.navigateTo)
        expect(new Set(targets).size).toBe(targets.length)
      }),
    )
  })
})
