// Unit tests for the direct-domain `toFixed` guard (Task 4.4, Req 15.5).
//
// Two layers are exercised:
//   1. The pure core (`countDirectToFixed` / `stripCommentsAndStrings`) must
//      count only real `.toFixed(` call sites and ignore occurrences inside
//      comments and string/template literals — otherwise doc comments that
//      merely mention `toFixed` would trip the check.
//   2. An integration guard runs the real check over the migrated targets and
//      asserts the count is 0, i.e. tasks 4.2 / 4.3 / 12.5 removed every direct
//      domain `toFixed` call and no regression has crept back in.

import { describe, expect, it } from 'vitest'

import {
  countDirectToFixed,
  runToFixedCheck,
  stripCommentsAndStrings,
} from '../../../scripts/checkToFixed'

describe('countDirectToFixed (pure core)', () => {
  it('counts a real .toFixed( call', () => {
    const hits = countDirectToFixed('const s = score.toFixed(2)')
    expect(hits).toHaveLength(1)
    expect(hits[0].line).toBe(1)
  })

  it('counts multiple calls and reports their line numbers', () => {
    const src = ['const a = x.toFixed(1)', 'const b = y.toFixed(3)'].join('\n')
    const hits = countDirectToFixed(src)
    expect(hits.map((h) => h.line)).toEqual([1, 2])
  })

  it('tolerates whitespace before the paren', () => {
    expect(countDirectToFixed('x.toFixed (2)')).toHaveLength(1)
  })

  it('ignores .toFixed inside a line comment', () => {
    expect(countDirectToFixed('// use score.toFixed(2) here')).toHaveLength(0)
  })

  it('ignores .toFixed inside a block/JSDoc comment', () => {
    const src = ['/**', ' * replaces value.toFixed(n) with formatMetric', ' */', 'const y = 1'].join('\n')
    expect(countDirectToFixed(src)).toHaveLength(0)
  })

  it('ignores .toFixed inside string and template literals', () => {
    expect(countDirectToFixed('const s = "call .toFixed(2)"')).toHaveLength(0)
    expect(countDirectToFixed("const s = 'x.toFixed(1)'")).toHaveLength(0)
    expect(countDirectToFixed('const s = `${x}.toFixed(1)`')).toHaveLength(0)
  })

  it('still counts a real call sitting next to a comment mention', () => {
    const src = '// score.toFixed(2) is banned\nconst s = score.toFixed(2)'
    expect(countDirectToFixed(src)).toHaveLength(1)
  })

  it('preserves newlines when scrubbing so line numbers stay aligned', () => {
    const src = '/* multi\nline */\nx.toFixed(2)'
    const scrubbed = stripCommentsAndStrings(src)
    expect(scrubbed.split('\n')).toHaveLength(3)
    expect(countDirectToFixed(src)[0].line).toBe(3)
  })
})

describe('runToFixedCheck (migrated targets guard, Req 15.5)', () => {
  it('finds zero direct domain toFixed calls in migrated components/pages', () => {
    const result = runToFixedCheck()
    // Sanity: the targets actually resolved to real source files.
    expect(result.scannedFiles).toBeGreaterThan(0)
    // The core assertion of Req 15.5: no direct domain toFixed remains.
    expect(result.violations).toEqual([])
    expect(result.totalHits).toBe(0)
    expect(result.ok).toBe(true)
  })
})
