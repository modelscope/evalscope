import type { PredictionRow } from '@/api/types'

/**
 * One sample as answered by every report under comparison.
 *
 * `Input` and `Gold` come from whichever report was seen first: they describe the
 * sample, not the answer, so they are identical across reports by construction.
 */
export interface MergedPrediction {
  Index: string
  Input: string
  Gold: string
  /** Keyed by report reference; a sample is only shown when every report answered it. */
  models: Record<string, PredictionRow>
}

/** Per-model view filter, relative to the score threshold. */
export type PerModelFilter = 'any' | 'above' | 'below'

/** Locale translate contract, narrowed to what the compare surfaces need. */
export type Translate = (path: string, vars?: Record<string, string | number>) => string

/** The colour tokens of one compare slot. */
export interface SlotPalette {
  dot: string
  border: string
  bg: string
  headerBg: string
}

/**
 * Distinct accent colour per model column — DESIGN.md §Compare Slots.
 *
 * Do NOT add a fourth entry: only three brand-colour slots exist, and extra
 * models must collapse to a numbered legend instead. Iteration paths use
 * `MODEL_PALETTE[i] ?? MODEL_PALETTE[0]` as a guard in case slicing is bypassed
 * upstream.
 */
export const MODEL_PALETTE: SlotPalette[] = [
  {
    dot: 'var(--compare-0-dot)',
    border: 'var(--compare-0-border)',
    bg: 'var(--compare-0-bg)',
    headerBg: 'var(--compare-0-bg-header)',
  },
  {
    dot: 'var(--compare-1-dot)',
    border: 'var(--compare-1-border)',
    bg: 'var(--compare-1-bg)',
    headerBg: 'var(--compare-1-bg-header)',
  },
  {
    dot: 'var(--compare-2-dot)',
    border: 'var(--compare-2-border)',
    bg: 'var(--compare-2-bg)',
    headerBg: 'var(--compare-2-bg-header)',
  },
]
