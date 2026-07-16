/**
 * Pure logic for classifying chart preflight responses (Chart_Renderer).
 *
 * The preflight request itself lives in `ChartFrame`; this module only holds the
 * deterministic, side-effect-free classification of an HTTP status code so it
 * can be covered by property-based tests.
 */

/**
 * Failure categories a chart preflight can end up in.
 *
 * `'timeout'` and `'network'` are derived from the request lifecycle (no
 * response / aborted), not from a status code, and are therefore produced by
 * `ChartFrame` rather than {@link classifyChartResponse}.
 */
export type ChartFailureKind = '4xx' | '5xx' | 'timeout' | 'network'

/** Classification of a chart preflight response derived purely from its status. */
export type ChartResponseClass = 'success' | '4xx' | '5xx'

/**
 * Classify an HTTP status code returned by a chart preflight request.
 *
 * The mapping is total and deterministic: the same status always yields the
 * same class (Property 5). `5xx` server errors and `4xx` client errors are
 * surfaced as distinct failure classes so the UI can react differently, while
 * any other status (including `2xx` and redirects) is treated as a success for
 * the purposes of rendering the chart.
 *
 * @param status - HTTP status code of the preflight response.
 * @returns `'5xx'` for server errors, `'4xx'` for client errors, otherwise
 *   `'success'`.
 */
export function classifyChartResponse(status: number): ChartResponseClass {
  if (status >= 500 && status < 600) return '5xx'
  if (status >= 400 && status < 500) return '4xx'
  return 'success'
}
