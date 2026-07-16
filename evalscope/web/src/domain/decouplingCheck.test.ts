// Unit tests for the data-model / rendering decoupling guard (Task 16.3,
// Req 15.1, 15.2).
//
// Two layers are exercised:
//   1. The pure core (`isRenderingSpecifier` / `findRenderingImports`) must flag
//      only rendering-layer imports (react, @/components, @/pages, react-router)
//      and leave pure-domain imports (@/api/types, @/utils/*, sibling domain
//      modules) untouched.
//   2. An integration guard runs the real check over the Compare and Predictions
//      data-model modules and asserts they import no rendering layer, i.e. the
//      data model holds no rendering logic (Req 15.1) and the prediction
//      presentation selector does not reference render internals (Req 15.2).

import { describe, expect, it } from 'vitest'

import {
  findRenderingImports,
  isRenderingSpecifier,
  runDecouplingCheck,
} from '../../scripts/checkDecoupling'

describe('isRenderingSpecifier (pure core)', () => {
  it('flags react / react-dom and their subpaths', () => {
    expect(isRenderingSpecifier('react')).toBe(true)
    expect(isRenderingSpecifier('react/jsx-runtime')).toBe(true)
    expect(isRenderingSpecifier('react-dom')).toBe(true)
    expect(isRenderingSpecifier('react-dom/client')).toBe(true)
  })

  it('flags the component and pages trees (alias and relative)', () => {
    expect(isRenderingSpecifier('@/components/ui/Card')).toBe(true)
    expect(isRenderingSpecifier('@/pages/ComparePage')).toBe(true)
    expect(isRenderingSpecifier('../../components/single/ChatView')).toBe(true)
    expect(isRenderingSpecifier('./pages/PerfComparePage')).toBe(true)
  })

  it('flags routing as a render concern', () => {
    expect(isRenderingSpecifier('react-router')).toBe(true)
    expect(isRenderingSpecifier('react-router-dom')).toBe(true)
  })

  it('does not flag pure-domain imports', () => {
    expect(isRenderingSpecifier('@/api/types')).toBe(false)
    expect(isRenderingSpecifier('@/utils/reportParser')).toBe(false)
    expect(isRenderingSpecifier('../metric/metricFormat')).toBe(false)
    expect(isRenderingSpecifier('fast-check')).toBe(false)
    // "react" only as a substring of an unrelated package must not match.
    expect(isRenderingSpecifier('preact-signals')).toBe(false)
  })
})

describe('findRenderingImports (pure core)', () => {
  it('finds a react import and reports its line', () => {
    const src = ["import { useState } from 'react'", "const x = 1"].join('\n')
    const hits = findRenderingImports(src)
    expect(hits).toHaveLength(1)
    expect(hits[0]).toEqual({ specifier: 'react', line: 1 })
  })

  it('recognises type-only and side-effect import forms', () => {
    const src = [
      "import type { ReactNode } from 'react'",
      "import '@/components/ui/Card'",
    ].join('\n')
    const specifiers = findRenderingImports(src).map((h) => h.specifier)
    expect(specifiers).toEqual(['react', '@/components/ui/Card'])
  })

  it('returns nothing for a purely domain module', () => {
    const src = [
      "import type { ReportData } from '@/api/types'",
      "import { parseReportName } from '@/utils/reportParser'",
      "import { formatMetric } from '../metric/metricFormat'",
    ].join('\n')
    expect(findRenderingImports(src)).toEqual([])
  })
})

describe('runDecouplingCheck (data-model guard, Req 15.1, 15.2)', () => {
  it('finds zero rendering imports in the Compare and Predictions data models', () => {
    const result = runDecouplingCheck()

    // Sanity: the guarded modules actually resolved and were scanned.
    expect(result.scannedModules).toBeGreaterThan(0)

    // The core assertion: data models import no rendering layer.
    expect(result.violations).toEqual([])
    expect(result.ok).toBe(true)
  })
})
