import { describe, expect, it } from 'vitest'
import fc from 'fast-check'

import { classifyChartResponse } from './chartPreflight'

// Feature: frontend-refactor-2026-07, Property 5: 图表响应分类
//
// For any HTTP status code, the chart preflight classification function must
// classify 2xx as success, 4xx as '4xx' and 5xx as '5xx', and the result must
// be deterministic (the same status always yields the same class).
//
// Validates: Requirements 2.4
describe('classifyChartResponse (Property 5: chart response classification)', () => {
  it('classifies any 2xx status as success', () => {
    fc.assert(
      fc.property(fc.integer({ min: 200, max: 299 }), (status) => {
        expect(classifyChartResponse(status)).toBe('success')
      }),
    )
  })

  it("classifies any 4xx status as '4xx'", () => {
    fc.assert(
      fc.property(fc.integer({ min: 400, max: 499 }), (status) => {
        expect(classifyChartResponse(status)).toBe('4xx')
      }),
    )
  })

  it("classifies any 5xx status as '5xx'", () => {
    fc.assert(
      fc.property(fc.integer({ min: 500, max: 599 }), (status) => {
        expect(classifyChartResponse(status)).toBe('5xx')
      }),
    )
  })

  it('is deterministic: the same status always yields the same class', () => {
    fc.assert(
      fc.property(fc.integer({ min: 0, max: 599 }), (status) => {
        const first = classifyChartResponse(status)
        const second = classifyChartResponse(status)
        expect(second).toBe(first)
      }),
    )
  })
})
