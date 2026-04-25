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
