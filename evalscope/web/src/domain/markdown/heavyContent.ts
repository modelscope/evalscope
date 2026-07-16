/**
 * Pure logic for Markdown heavy-content rendering decisions (Req 16.3, 16.4).
 *
 * "Heavy content" is content that is comparatively expensive to render: math
 * formula blocks, syntax-highlighted code blocks, and large tables (more than
 * {@link LARGE_TABLE_LINE_THRESHOLD} source lines). While such content is
 * collapsed the Web_Console must not render it (Req 16.3); once expanded it is
 * rendered again (Req 16.4).
 *
 * This module holds the deterministic, side-effect-free decision logic so it
 * can be exercised directly by property-based tests (Property 28). It has no
 * DOM, network, clock or randomness dependencies. `MarkdownRenderer` consumes
 * {@link shouldRenderHeavy} to gate each heavy region, and the detection
 * helpers to decide which optional modules to load on demand (Req 16.1).
 */

/**
 * Kinds of content a Markdown region can be classified as.
 *
 * - `'text'`        — ordinary prose / light markup; never gated by collapse.
 * - `'math'`        — a math formula block rendered via KaTeX (heavy).
 * - `'code'`        — a syntax-highlighted fenced code block (heavy).
 * - `'large-table'` — a table exceeding {@link LARGE_TABLE_LINE_THRESHOLD}
 *   source lines (heavy).
 */
export type ContentKind = 'text' | 'math' | 'code' | 'large-table'

/** Content kinds considered heavy and therefore gated while collapsed. */
export const HEAVY_CONTENT_KINDS: readonly ContentKind[] = ['math', 'code', 'large-table']

/**
 * A table is "heavy" once its source spans strictly more than this many lines
 * (Req 16.3). A table at or below the threshold renders eagerly.
 */
export const LARGE_TABLE_LINE_THRESHOLD = 50

/** Whether a content kind is heavy (expensive to render). */
export function isHeavyContent(contentType: ContentKind): boolean {
  return HEAVY_CONTENT_KINDS.includes(contentType)
}

/**
 * Decide whether a content region should be rendered right now.
 *
 * The rule is total and deterministic (Property 28): heavy content is rendered
 * only when it is NOT collapsed; light content (`'text'`) is always rendered
 * regardless of collapse state.
 *
 *   shouldRenderHeavy(collapsed, contentType) =
 *     !isHeavyContent(contentType) || !collapsed
 *
 * @param collapsed - Whether the region is currently collapsed.
 * @param contentType - The classified kind of the region.
 * @returns `true` when the region should be rendered, `false` when a collapsed
 *   heavy region should be withheld (Req 16.3).
 */
export function shouldRenderHeavy(collapsed: boolean, contentType: ContentKind): boolean {
  if (!isHeavyContent(contentType)) {
    return true
  }
  return !collapsed
}

/**
 * Decide whether a table with the given source-line span is heavy.
 *
 * @param lineCount - Number of Markdown source lines the table spans.
 * @returns `true` when the table exceeds {@link LARGE_TABLE_LINE_THRESHOLD}.
 */
export function isLargeTable(lineCount: number): boolean {
  if (!Number.isFinite(lineCount)) {
    return false
  }
  return lineCount > LARGE_TABLE_LINE_THRESHOLD
}

// Detection helpers used to decide which optional modules to load (Req 16.1).
// These are intentionally lightweight lexical scans, not full Markdown parses:
// a false positive only costs an unnecessary (still lazy) module load, never a
// rendering error.

/** Matches a fenced code block opener (``` or ~~~), the syntax-highlight trigger. */
const FENCED_CODE_RE = /(^|\n)[ \t]*(```|~~~)/

/** Matches inline `$...$` or block `$$...$$` math delimiters. */
const INLINE_MATH_RE = /\$[^\n$]+\$/
const BLOCK_MATH_RE = /\$\$[\s\S]+?\$\$/

/**
 * Whether the content contains math that would require the KaTeX pipeline.
 *
 * When this returns `false` the renderer must not load `remark-math` /
 * `rehype-katex` (Req 16.1).
 */
export function containsMath(content: string): boolean {
  if (!content) {
    return false
  }
  return BLOCK_MATH_RE.test(content) || INLINE_MATH_RE.test(content)
}

/**
 * Whether the content contains a fenced code block that would require the
 * syntax highlighter.
 *
 * When this returns `false` the renderer must not load
 * `react-syntax-highlighter` (Req 16.1).
 */
export function containsCodeBlock(content: string): boolean {
  if (!content) {
    return false
  }
  return FENCED_CODE_RE.test(content)
}
