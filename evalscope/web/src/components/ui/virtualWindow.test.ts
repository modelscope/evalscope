import { describe, it, expect } from 'vitest'
import { computeVirtualWindow, type VirtualWindowInput } from './virtualWindow'

/** Convenience: run computeVirtualWindow with sensible defaults. */
function run(partial: Partial<VirtualWindowInput> & Pick<VirtualWindowInput, 'heights'>) {
  return computeVirtualWindow({
    gap: 0,
    scrollTop: 0,
    viewportHeight: 0,
    overscan: 0,
    ...partial,
  })
}

describe('computeVirtualWindow — offsets & total height', () => {
  it('produces a zero-based prefix sum with gaps between items', () => {
    const { offsets, totalHeight } = run({ heights: [100, 200, 50], gap: 10 })
    expect(offsets).toEqual([0, 110, 320])
    // 100 + 10 + 200 + 10 + 50, no trailing gap.
    expect(totalHeight).toBe(370)
  })

  it('adds no gap for a single item and no trailing gap overall', () => {
    expect(run({ heights: [120], gap: 8 })).toMatchObject({ offsets: [0], totalHeight: 120 })
  })

  it('returns empty geometry for an empty list', () => {
    expect(run({ heights: [] })).toEqual({ offsets: [], totalHeight: 0, startIndex: 0, endIndex: 0 })
  })

  it('treats invalid or negative heights/gaps as zero', () => {
    const { offsets, totalHeight } = run({ heights: [NaN, -5, 100], gap: -3 })
    expect(offsets).toEqual([0, 0, 0])
    expect(totalHeight).toBe(100)
  })
})

describe('computeVirtualWindow — visible range', () => {
  const heights = Array.from({ length: 10 }, () => 100) // offsets: 0,100,...,900

  it('selects only items intersecting the viewport with no overscan', () => {
    // Viewport [250, 450): touches items 2 (200-300), 3 (300-400), 4 (400-500).
    const { startIndex, endIndex } = computeVirtualWindow({
      heights,
      gap: 0,
      scrollTop: 250,
      viewportHeight: 200,
      overscan: 0,
    })
    expect(startIndex).toBe(2)
    expect(endIndex).toBe(5)
  })

  it('covers every item whose interval intersects the viewport', () => {
    const scrollTop = 250
    const viewportHeight = 200
    const { startIndex, endIndex, offsets } = computeVirtualWindow({
      heights,
      gap: 0,
      scrollTop,
      viewportHeight,
      overscan: 0,
    })
    const viewportBottom = scrollTop + viewportHeight
    for (let i = 0; i < heights.length; i++) {
      const top = offsets[i]
      const bottom = top + heights[i]
      const intersects = bottom > scrollTop && top < viewportBottom
      if (intersects) {
        expect(i).toBeGreaterThanOrEqual(startIndex)
        expect(i).toBeLessThan(endIndex)
      }
    }
  })

  it('widens the range by overscan on both sides and clamps to bounds', () => {
    const { startIndex, endIndex } = computeVirtualWindow({
      heights,
      gap: 0,
      scrollTop: 450,
      viewportHeight: 100,
      overscan: 2,
    })
    // Core viewport [450,550) touches items 4,5 -> widened by 2 -> [2,8).
    expect(startIndex).toBe(2)
    expect(endIndex).toBe(8)
  })

  it('clamps overscan at the list edges', () => {
    const top = computeVirtualWindow({ heights, gap: 0, scrollTop: 0, viewportHeight: 100, overscan: 5 })
    expect(top.startIndex).toBe(0)
    const bottom = computeVirtualWindow({ heights, gap: 0, scrollTop: 900, viewportHeight: 100, overscan: 5 })
    expect(bottom.endIndex).toBe(10)
  })

  it('renders at least one item when scrolled past measured content', () => {
    const { startIndex, endIndex } = computeVirtualWindow({
      heights,
      gap: 0,
      scrollTop: 100000,
      viewportHeight: 200,
      overscan: 0,
    })
    expect(startIndex).toBeLessThan(endIndex)
    expect(endIndex).toBeLessThanOrEqual(heights.length)
  })

  it('keeps 0 <= startIndex <= endIndex <= count across varied inputs', () => {
    const cases: VirtualWindowInput[] = [
      { heights: [50, 60, 70], gap: 4, scrollTop: 0, viewportHeight: 0, overscan: 0 },
      { heights: [10], gap: 0, scrollTop: 5, viewportHeight: 3, overscan: 3 },
      { heights: [100, 100, 100, 100], gap: 12, scrollTop: 130, viewportHeight: 90, overscan: 1 },
      { heights: [0, 0, 0], gap: 0, scrollTop: 0, viewportHeight: 10, overscan: 0 },
    ]
    for (const input of cases) {
      const { startIndex, endIndex } = computeVirtualWindow(input)
      expect(startIndex).toBeGreaterThanOrEqual(0)
      expect(endIndex).toBeGreaterThanOrEqual(startIndex)
      expect(endIndex).toBeLessThanOrEqual(input.heights.length)
    }
  })
})
