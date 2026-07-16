// Feature: frontend-refactor-2026-07, Property 7: deterministic chat presentation selection

import fc from 'fast-check'
import { describe, expect, it } from 'vitest'

import { selectChatPresentation } from './chatPresentation'

describe('selectChatPresentation (Property 7)', () => {
  it('selects traced, structured, or legacy mode solely from available structured data', () => {
    fc.assert(
      fc.property(fc.boolean(), fc.boolean(), (hasMessages, hasTrace) => {
        const prediction = {
          Messages: hasMessages ? [{ role: 'user' as const, content: 'hello' }] : [],
          AgentTrace: hasTrace
            ? { max_steps: 1, events: [{ type: 'run_start' as const, step: 0, timestamp: 0, payload: {} }] }
            : undefined,
        }

        const expected = hasMessages && hasTrace ? 'traced' : hasMessages ? 'structured' : 'legacy'
        expect(selectChatPresentation(prediction)).toBe(expected)
      }),
    )
  })
})
