/**
 * Pure windowing math for the self-implemented list virtualization.
 *
 * The rendering component ({@link ../common/VirtualList}) owns the scroll
 * container, item-height measurement and React state; this module contains only
 * the deterministic geometry so it can be unit tested without a DOM.
 *
 * The model is a vertical stack of `count` items separated by a fixed `gap`.
 * Each item has a resolved height in px (a real measured value once observed, or
 * an estimate before measurement). Given the current scroll offset and viewport
 * height we compute the slice of items that intersect the viewport (expanded by
 * `overscan` items on each side) plus the absolute top offset of every item.
 */

export interface VirtualWindowInput {
  /** Resolved height in px for each item (measured value or estimate). */
  heights: number[]
  /** Vertical gap in px inserted between consecutive items. */
  gap: number
  /** Current scroll offset of the viewport in px. */
  scrollTop: number
  /** Height of the visible viewport in px. */
  viewportHeight: number
  /** Number of extra items to render on each side of the visible range. */
  overscan: number
}

export interface VirtualWindowResult {
  /** Top offset in px for each item (prefix sum of heights + gaps). */
  offsets: number[]
  /** Total scrollable content height in px. */
  totalHeight: number
  /** Index of the first item to render (inclusive). */
  startIndex: number
  /** Index one past the last item to render (exclusive). */
  endIndex: number
}

/** Clamp a possibly-invalid px value to a finite, non-negative number. */
function nonNegative(value: number): number {
  return Number.isFinite(value) && value > 0 ? value : 0
}

/**
 * Compute the visible window and item offsets for a virtualized list.
 *
 * Invariants (all verified by the accompanying tests):
 * - `offsets[0] === 0` and `offsets[i+1] === offsets[i] + heights[i] + gap`.
 * - `totalHeight === sum(heights) + gap * (count - 1)` (no trailing gap).
 * - `0 <= startIndex <= endIndex <= count`.
 * - Every item whose vertical interval intersects the viewport is contained in
 *   `[startIndex, endIndex)`; the range is then widened by `overscan` on each
 *   side and clamped to the list bounds.
 * - For a non-empty list at least one item is always rendered.
 */
export function computeVirtualWindow(input: VirtualWindowInput): VirtualWindowResult {
  const gap = nonNegative(input.gap)
  const overscan = Math.max(0, Math.floor(input.overscan))
  const count = input.heights.length

  const offsets: number[] = new Array(count)
  let acc = 0
  for (let i = 0; i < count; i++) {
    offsets[i] = acc
    acc += nonNegative(input.heights[i])
    if (i < count - 1) acc += gap
  }
  const totalHeight = acc

  if (count === 0) {
    return { offsets, totalHeight: 0, startIndex: 0, endIndex: 0 }
  }

  const scrollTop = nonNegative(input.scrollTop)
  const viewportBottom = scrollTop + nonNegative(input.viewportHeight)

  // First item whose bottom edge is strictly past the top of the viewport.
  let start = 0
  while (start < count && offsets[start] + nonNegative(input.heights[start]) <= scrollTop) {
    start++
  }
  // First item whose top edge is at or beyond the bottom of the viewport.
  let end = start
  while (end < count && offsets[end] < viewportBottom) {
    end++
  }

  start = Math.max(0, start - overscan)
  end = Math.min(count, end + overscan)

  // Guard: keep at least one item rendered even when the scroll offset lands
  // past the measured content (e.g. before heights settle).
  if (start >= count) start = count - 1
  if (end <= start) end = Math.min(count, start + 1)

  return { offsets, totalHeight, startIndex: start, endIndex: end }
}
