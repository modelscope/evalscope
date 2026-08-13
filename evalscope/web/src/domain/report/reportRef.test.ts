import { describe, expect, it } from 'vitest'
import fc from 'fast-check'

import { formatReportRef, parseReportRef, reportRefFromSummary } from './reportRef'

describe('reportRef', () => {
  it('round-trips a reference through format and parse', () => {
    const segment = fc.stringMatching(/^[A-Za-z0-9._-]{1,20}$/)
    fc.assert(
      fc.property(segment, segment, (runId, modelId) => {
        const ref = formatReportRef({ runId, modelId })
        expect(ref).toBe(`${runId}/${modelId}`)
        expect(parseReportRef(ref)).toEqual({ runId, modelId })
      }),
    )
  })

  it('splits on the first separator so a model id keeps trailing separators', () => {
    // Model ids are single path segments server-side, but splitting on the first separator is the
    // safe contract: everything after the first slash is the model id.
    expect(parseReportRef('run/model/extra')).toEqual({ runId: 'run', modelId: 'model/extra' })
  })

  it('treats a value with no separator as a run id with an empty model id', () => {
    expect(parseReportRef('run-only')).toEqual({ runId: 'run-only', modelId: '' })
  })

  it('builds a reference from a report summary', () => {
    expect(reportRefFromSummary({ run_id: '20260811_152001', model_id: 'qwen-plus' })).toEqual({
      runId: '20260811_152001',
      modelId: 'qwen-plus',
    })
  })
})
