/**
 * Roving tabindex navigation (pure logic, no DOM dependency).
 *
 * Provides the index math used by an accessible tablist implementing the WAI-ARIA
 * roving tabindex pattern. Given the current index, the number of tabs and a
 * movement direction, these helpers return a valid index within `[0, count)` and
 * wrap around when moving past the first / last tab (Req 11.5).
 */

/** Movement direction along the tablist. */
export type RovingDirection = 'next' | 'prev'

/** Orientation of the tablist, controls which arrow keys move the focus. */
export type RovingOrientation = 'horizontal' | 'vertical'

/**
 * Compute the next roving index when moving in a given direction.
 *
 * The result is always a valid index within `[0, count)`, wrapping from the last
 * tab back to the first (and vice versa). The `current` index is clamped into a
 * safe range first so out-of-bounds inputs still yield a valid result.
 *
 * Edge cases:
 * - `count <= 0`: there is no valid index, so `-1` is returned.
 * - `count === 1`: the only valid index `0` is always returned.
 *
 * @param current - The currently focused tab index.
 * @param count - The total number of tabs.
 * @param direction - Whether to move to the `next` or `prev` tab.
 * @returns A valid index within `[0, count)`, or `-1` when `count <= 0`.
 */
export function nextRovingIndex(current: number, count: number, direction: RovingDirection): number {
  if (count <= 0) {
    return -1
  }
  if (count === 1) {
    return 0
  }

  // Normalize `current` into `[0, count)` so out-of-range inputs are safe.
  const normalized = ((Math.trunc(current) % count) + count) % count
  const step = direction === 'next' ? 1 : -1
  return (normalized + step + count) % count
}

/**
 * Map a keyboard key to a roving index according to the tablist orientation.
 *
 * Supported keys:
 * - Horizontal orientation: `ArrowRight` -> next, `ArrowLeft` -> prev.
 * - Vertical orientation: `ArrowDown` -> next, `ArrowUp` -> prev.
 * - `Home` -> first tab, `End` -> last tab (available in both orientations).
 *
 * Keys that do not affect navigation return `null` so the caller can ignore them.
 *
 * @param current - The currently focused tab index.
 * @param count - The total number of tabs.
 * @param key - The `KeyboardEvent.key` value.
 * @param orientation - The tablist orientation (defaults to `horizontal`).
 * @returns The next valid index within `[0, count)`, or `null` when the key is
 *   not a navigation key. Returns `-1` when `count <= 0`.
 */
export function moveRovingIndex(
  current: number,
  count: number,
  key: string,
  orientation: RovingOrientation = 'horizontal',
): number | null {
  if (count <= 0) {
    return -1
  }

  const nextKey = orientation === 'vertical' ? 'ArrowDown' : 'ArrowRight'
  const prevKey = orientation === 'vertical' ? 'ArrowUp' : 'ArrowLeft'

  switch (key) {
    case nextKey:
      return nextRovingIndex(current, count, 'next')
    case prevKey:
      return nextRovingIndex(current, count, 'prev')
    case 'Home':
      return 0
    case 'End':
      return count - 1
    default:
      return null
  }
}
