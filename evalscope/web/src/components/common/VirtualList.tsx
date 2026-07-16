import { useCallback, useEffect, useRef, useState, type CSSProperties, type ReactNode } from 'react'
import { computeVirtualWindow } from './virtualWindow'

/**
 * Minimal self-implemented list virtualization (no external dependency).
 *
 * Long Predictions / agent-Trace lists can contain hundreds of message rows or
 * step blocks; rendering them all is expensive. This component windows the list
 * so only the items intersecting the viewport (plus a small overscan) are
 * mounted, while a sized spacer preserves the real scroll height.
 *
 * Design notes:
 * - Variable row heights are supported: each rendered item is measured (initial
 *   layout + a {@link ResizeObserver} for async growth such as Markdown / image
 *   loads or a disclosure toggling open) and unmeasured items fall back to
 *   `estimateHeight`. The pure windowing math lives in `computeVirtualWindow`.
 * - Virtualization only kicks in past `threshold` items, so short lists render
 *   exactly as before (no inner scroll container, no behavioural change). It
 *   also degrades gracefully to a plain list when `ResizeObserver` is
 *   unavailable, keeping the change additive and reversible.
 */
export interface VirtualListProps<T> {
  /** Items to render. */
  items: T[]
  /** Stable key for an item (used as React key). */
  getKey: (item: T, index: number) => string | number
  /** Render a single item's content. */
  renderItem: (item: T, index: number) => ReactNode
  /** Estimated row height in px used before an item is measured. Default 160. */
  estimateHeight?: number
  /** Vertical gap in px between consecutive items. Default 8. */
  gap?: number
  /** Extra items rendered above/below the viewport. Default 6. */
  overscan?: number
  /** Only virtualize when item count exceeds this threshold. Default 30. */
  threshold?: number
  /** Max height in px of the internal scroll container when virtualizing. Default 800. */
  maxHeight?: number
  /** Class applied to the outer container. */
  className?: string
  /** Extra styles merged onto the outer container. */
  style?: CSSProperties
}

export default function VirtualList<T>({
  items,
  getKey,
  renderItem,
  estimateHeight = 160,
  gap = 8,
  overscan = 6,
  threshold = 30,
  maxHeight = 800,
  className,
  style,
}: VirtualListProps<T>) {
  const canVirtualize = typeof ResizeObserver !== 'undefined'
  const active = canVirtualize && items.length > threshold

  const scrollRef = useRef<HTMLDivElement | null>(null)
  const itemEls = useRef<Map<number, HTMLElement>>(new Map())
  const [scrollTop, setScrollTop] = useState(0)
  const [viewportHeight, setViewportHeight] = useState(maxHeight)
  // Measured heights per index; missing entries fall back to `estimateHeight`.
  const [heights, setHeights] = useState<number[]>([])

  const handleScroll = useCallback(() => {
    const el = scrollRef.current
    if (el) setScrollTop(el.scrollTop)
  }, [])

  const setItemRef = useCallback(
    (index: number) => (el: HTMLElement | null) => {
      const map = itemEls.current
      if (el) {
        map.set(index, el)
      } else {
        map.delete(index)
      }
    },
    [],
  )

  // Read the mounted items' heights (refs are read here, inside a callback
  // invoked from effects — never during render) and merge any changes into
  // state. Converges once heights settle since state only updates on a change.
  const measure = useCallback(() => {
    const collected: Array<[number, number]> = []
    itemEls.current.forEach((el, index) => {
      const h = el.offsetHeight
      if (h > 0) collected.push([index, h])
    })
    setHeights((prev) => {
      let changed = false
      const next = prev.slice()
      for (const [index, h] of collected) {
        if (next[index] !== h) {
          next[index] = h
          changed = true
        }
      }
      return changed ? next : prev
    })
  }, [])

  // Track the viewport height so the visible window reacts to container resizes.
  useEffect(() => {
    if (!active) return
    const el = scrollRef.current
    if (!el) return
    const update = () => setViewportHeight(el.clientHeight || maxHeight)
    update()
    const ro = new ResizeObserver(update)
    ro.observe(el)
    return () => ro.disconnect()
  }, [active, maxHeight])

  // Measure mounted items. The ResizeObserver delivers an initial notification
  // for every observed target, so this covers both the first measurement after
  // render and later async growth (Markdown / image load, disclosure toggling).
  // Re-observes the currently mounted items every commit; the mounted set is
  // bounded by the window size so this stays cheap.
  useEffect(() => {
    if (!active) return
    const ro = new ResizeObserver(() => measure())
    itemEls.current.forEach((el) => ro.observe(el))
    return () => ro.disconnect()
  })

  if (!active) {
    // Short list (or no ResizeObserver): render everything in normal flow.
    return (
      <div className={className} style={{ display: 'flex', flexDirection: 'column', gap: `${gap}px`, ...style }}>
        {items.map((item, i) => (
          <div key={getKey(item, i)}>{renderItem(item, i)}</div>
        ))}
      </div>
    )
  }

  const resolvedHeights = items.map((_, i) => heights[i] ?? estimateHeight)
  const { offsets, totalHeight, startIndex, endIndex } = computeVirtualWindow({
    heights: resolvedHeights,
    gap,
    scrollTop,
    viewportHeight: viewportHeight || maxHeight,
    overscan,
  })

  const visible: ReactNode[] = []
  for (let i = startIndex; i < endIndex; i++) {
    visible.push(
      <div
        key={getKey(items[i], i)}
        data-vindex={i}
        ref={setItemRef(i)}
        style={{ position: 'absolute', top: offsets[i], left: 0, right: 0 }}
      >
        {renderItem(items[i], i)}
      </div>,
    )
  }

  return (
    <div
      ref={scrollRef}
      onScroll={handleScroll}
      className={className}
      style={{ overflowY: 'auto', maxHeight, position: 'relative', ...style }}
    >
      <div style={{ height: totalHeight, position: 'relative', width: '100%' }}>{visible}</div>
    </div>
  )
}
