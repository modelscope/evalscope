/** Tokens used in report name encoding (must match Python constants). */
export const REPORT_TOKEN = '@@'
export const MODEL_TOKEN = '::'
export const DATASET_TOKEN = ', '

/**
 * Parse a report identifier string into its constituent parts.
 * Format: `{timestamp}@@{model_name}::{dataset1}, {dataset2}`
 */
export function parseReportName(name: string): { prefix: string; model: string; datasets: string[] } {
  const [prefix, rest] = name.split(REPORT_TOKEN)
  if (!rest) return { prefix: name, model: '', datasets: [] }
  const [model, dsStr] = rest.split(MODEL_TOKEN)
  const datasets = dsStr ? dsStr.split(DATASET_TOKEN) : []
  return { prefix, model, datasets }
}

/** Extract just the model name portion of a report identifier. */
export function modelFromReport(name: string): string {
  return parseReportName(name).model
}

/**
 * Generate a display name that is unique among a list of report names.
 * If multiple reports share the same model name, append the timestamp prefix to distinguish them.
 */
export function getDisplayNames(names: string[]): Record<string, string> {
  const modelCounts: Record<string, number> = {}
  for (const n of names) {
    const m = parseReportName(n).model || n
    modelCounts[m] = (modelCounts[m] ?? 0) + 1
  }
  const result: Record<string, string> = {}
  for (const n of names) {
    const { prefix, model } = parseReportName(n)
    const base = model || n
    result[n] = modelCounts[base] > 1 ? `${base} (${prefix})` : base
  }
  return result
}
