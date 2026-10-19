import type { ReportData } from '@/api/types'

function record(value: unknown): Record<string, unknown> | null {
  return value != null && typeof value === 'object' && !Array.isArray(value)
    ? value as Record<string, unknown>
    : null
}

function finite(value: unknown): value is number {
  return typeof value === 'number' && Number.isFinite(value)
}

/** Read the existing score metadata for presentation, without recomputing probabilities. */
export function decisionObservation(metadata: unknown) {
  const data = record(record(metadata)?.decision)
  if (!data || !['choice', 'noul', 'score'].includes(String(data.kind))) return null
  const { labels, probabilities } = data
  if (!Array.isArray(labels) || !Array.isArray(probabilities) || !labels.length
    || labels.length !== probabilities.length || !labels.every((v): v is string => typeof v === 'string')
    || !probabilities.every((v): v is number => finite(v) && v >= 0 && v <= 1)) return null
  const raw = record(data.raw_probabilities)
  return {
    kind: String(data.kind),
    predicted: String(data.predicted ?? ''),
    gold: String(data.gold ?? ''),
    expected: finite(data.expected) ? data.expected : null,
    confidence: finite(data.confidence) ? data.confidence : null,
    threshold: finite(data.threshold) ? data.threshold : null,
    continuous: data.continuous === true,
    rows: labels.map((label, index) => ({
      label,
      probability: probabilities[index] as number,
      raw: finite(raw?.[label]) ? raw[label] as number : null,
    })),
  }
}

/** Select saved curves by their complete metric identity and subset; never combine scales. */
export function decisionCurves(report?: ReportData) {
  return (report?.metrics ?? []).filter((metric) => metric.identity.name === 'aurc').flatMap((metric) => {
    const dims = metric.identity.dimensions
    const basis = String(dims.confidence ?? '')
    return metric.categories.flatMap((category) => category.subsets.flatMap((subset) => {
      const saved = subset.metadata?.[`risk_coverage_${basis}`]
      if (!Array.isArray(saved) || !saved.length) return []
      const points = saved.map(record)
      if (!points.every((point) => point && finite(point.coverage) && finite(point.risk)
        && point.coverage > 0 && point.coverage <= 1 && point.risk >= 0 && point.risk <= 1)) return []
      return [{
        id: JSON.stringify([metric.identity, category.name, subset.name]),
        subset: subset.name,
        kind: String(dims.kind ?? ''),
        levels: dims.levels,
        basis,
        risk: String(dims.risk ?? ''),
        aurc: subset.score,
        semantics: metric.semantics,
        num: subset.num,
        unavailable: record(subset.metadata?.unavailable),
        points: points.map((point) => ({ coverage: point!.coverage as number, risk: point!.risk as number })),
      }]
    }))
  })
}
