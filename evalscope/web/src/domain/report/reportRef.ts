import type { ReportSummary } from '@/api/types'

/**
 * Identity of one model's report inside one evaluation run.
 *
 * A report lives at `<root>/<runId>/reports/<modelId>/`, so it is addressed by exactly two names.
 * The datasets it covers are data read from that directory, never part of its identity.
 */
export interface ReportRef {
  runId: string
  modelId: string
}

/** Separator between the two names in the flat form used for cache keys, URLs and selections. */
const REF_SEPARATOR = '/'

/**
 * Serialize a reference to its flat `"{runId}/{modelId}"` form.
 *
 * This flat string is the single identifier passed between components, used as a cache key, held in
 * selection sets and carried in the `report` query parameter. Only request builders split it apart.
 */
export function formatReportRef(ref: ReportRef): string {
  return `${ref.runId}${REF_SEPARATOR}${ref.modelId}`
}

/**
 * Parse the flat `"{runId}/{modelId}"` form.
 *
 * Splits on the first separator so a `modelId` is never truncated (model ids are sanitized to a
 * single path segment server-side, but splitting on the first separator is the safe contract).
 * When no separator is present the whole value is treated as the `runId` with an empty `modelId`,
 * mirroring how the backend rejects such a reference.
 */
export function parseReportRef(value: string): ReportRef {
  const index = value.indexOf(REF_SEPARATOR)
  if (index === -1) {
    return { runId: value, modelId: '' }
  }
  return { runId: value.slice(0, index), modelId: value.slice(index + REF_SEPARATOR.length) }
}

/** Build a reference from a report-list summary item. */
export function reportRefFromSummary(summary: Pick<ReportSummary, 'run_id' | 'model_id'>): ReportRef {
  return { runId: summary.run_id, modelId: summary.model_id }
}
