import { describe, expect, it } from 'vitest'

import { loadFixture } from '@/test/loadFixture'
import { loadReportResponseSchema } from './reports.schema'

describe('loadReportResponseSchema real report compatibility', () => {
  const fixture = loadFixture<unknown>('report-real-single-sample')

  it('accepts single-sample null standard deviations and reports without perf metrics', () => {
    const result = loadReportResponseSchema.safeParse(fixture)

    expect(result.success).toBe(true)
    if (!result.success) return
    expect(result.data.report_list[0].perf_metrics?.summary.latency.std).toBeNull()
    expect(result.data.report_list[1].perf_metrics).toBeNull()
  })

  it('continues to reject null for defined statistics such as the mean', () => {
    const invalid = structuredClone(fixture) as {
      report_list: Array<{ perf_metrics?: { summary: { latency: { mean: number | null } } } }>
    }
    invalid.report_list[0].perf_metrics!.summary.latency.mean = null

    expect(loadReportResponseSchema.safeParse(invalid).success).toBe(false)
  })
})
