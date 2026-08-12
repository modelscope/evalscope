import { describe, expect, it, vi } from 'vitest'
import { z } from 'zod'

import { apiValidated } from './client'

/** Stub `fetch` to return `body` as a successful JSON response. */
function mockJsonResponse(body: unknown): void {
  vi.stubGlobal(
    'fetch',
    vi.fn(async () => ({ ok: true, json: async () => body }) as unknown as Response),
  )
}

describe('apiValidated backend-null normalization', () => {
  it('accepts explicit nulls for .optional() fields nested at any depth', async () => {
    // Mirrors how the backend emits Optional[...] = None (Pydantic) and NaN
    // cells (pandas to_json): explicit JSON null rather than an absent key.
    const schema = z.object({
      top: z.number().optional(),
      nested: z.object({
        inner: z.string().optional(),
        items: z.array(z.object({ token: z.number().optional() })),
      }),
    })
    mockJsonResponse({
      top: null,
      nested: { inner: null, items: [{ token: null }, { token: 7 }] },
    })

    const result = await apiValidated('/api/v1/test', schema)

    expect(result).toEqual({
      top: undefined,
      nested: { inner: undefined, items: [{ token: undefined }, { token: 7 }] },
    })
  })

  it('still rejects a null where the schema requires a concrete value', async () => {
    const schema = z.object({ required: z.number() })
    mockJsonResponse({ required: null })

    await expect(apiValidated('/api/v1/test', schema)).rejects.toMatchObject({ kind: 'validation' })
  })
})
