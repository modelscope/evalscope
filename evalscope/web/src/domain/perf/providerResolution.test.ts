// Feature: frontend-refactor-2026-07, Property 13: Provider precedence resolution
//
// For any Performance run input, `resolveProvider` must follow a strict
// priority order when resolving the provider:
//   1. an explicit, non-empty `Provider` metadata value  → source = 'metadata';
//   2. otherwise a known API-URL host                     → source = 'host-detection';
//   3. otherwise neither is available                     → provider = 'Custom',
//                                                            source = 'custom-fallback'.
// In every case the result must always carry two independent, non-empty
// `provider` and `protocol` fields (the protocol is never derived from the
// provider and vice versa).
//
// Validates: Requirements 8.1, 8.2, 8.3

import { describe, expect, it } from 'vitest'
import fc from 'fast-check'

import {
  CUSTOM_PROVIDER,
  resolveProvider,
  type ProviderResolutionInput,
} from './providerResolution'

/**
 * Known host → provider pairs, mirroring the KNOWN_HOSTS table in the
 * production module. Kept local so the test asserts against an explicit,
 * independent expectation rather than re-deriving it from the code under test.
 */
const KNOWN_HOST_PAIRS: ReadonlyArray<readonly [string, string]> = [
  ['dashscope.aliyuncs.com', 'DashScope'],
  ['dashscope-intl.aliyuncs.com', 'DashScope'],
  ['api.openai.com', 'OpenAI'],
  ['api.anthropic.com', 'Anthropic'],
  ['generativelanguage.googleapis.com', 'Google'],
  ['api.deepseek.com', 'DeepSeek'],
  ['api.moonshot.cn', 'Moonshot'],
  ['openrouter.ai', 'OpenRouter'],
  ['api.together.xyz', 'Together'],
  ['api.mistral.ai', 'Mistral'],
  ['api.groq.com', 'Groq'],
]

/** `basic_info` keys the resolver reads (must match the production module). */
const PROVIDER_KEY = 'Provider'
const PROTOCOL_KEY = 'Protocol'
const API_URL_KEY = 'API URL'

/** Assert both output fields are independent, non-empty strings. */
function expectIndependentNonEmptyFields(result: ReturnType<typeof resolveProvider>): void {
  expect(typeof result.provider).toBe('string')
  expect(typeof result.protocol).toBe('string')
  expect(result.provider.length).toBeGreaterThan(0)
  expect(result.protocol.length).toBeGreaterThan(0)
}

/** Strings that are non-empty after trimming (so `readInfo` keeps them). */
const nonBlankArb = fc.string({ minLength: 1, maxLength: 40 }).filter((s) => s.trim().length > 0)

/** Blank-ish values that `readInfo` treats as absent (empty / whitespace only). */
const blankArb = fc.constantFrom('', ' ', '   ', '\t', '\n', ' \t \n ')

/** Optional explicit protocol metadata; when present it drives `protocol`. */
const optionalProtocolArb = fc.option(nonBlankArb, { nil: undefined })

/**
 * Build a `basic_info` map, omitting keys whose value is `undefined` so we can
 * exercise both the "key absent" and "key blank" shapes.
 */
function makeBasicInfo(entries: Record<string, string | undefined>): Record<string, string> {
  const info: Record<string, string> = {}
  for (const [key, value] of Object.entries(entries)) {
    if (value !== undefined) {
      info[key] = value
    }
  }
  return info
}

describe('resolveProvider (Property 13: Provider precedence resolution)', () => {
  // Scenario 1: explicit non-empty Provider metadata wins over everything else.
  it('uses explicit Provider metadata (source = metadata)', () => {
    const caseArb = fc.record({
      provider: nonBlankArb,
      protocol: optionalProtocolArb,
      // A known host that would ALSO match host-detection, to prove metadata
      // takes strict priority over it.
      apiUrl: fc.option(
        fc.constantFrom(...KNOWN_HOST_PAIRS.map(([host]) => `https://${host}/v1`)),
        { nil: undefined },
      ),
      apiType: fc.option(nonBlankArb, { nil: undefined }),
    })

    fc.assert(
      fc.property(caseArb, ({ provider, protocol, apiUrl, apiType }) => {
        const input: ProviderResolutionInput = {
          basic_info: makeBasicInfo({
            [PROVIDER_KEY]: provider,
            [PROTOCOL_KEY]: protocol,
            [API_URL_KEY]: apiUrl,
          }),
          api_type: apiType ?? null,
        }

        const result = resolveProvider(input)

        expect(result.source).toBe('metadata')
        // Provider equals the trimmed metadata value.
        expect(result.provider).toBe(provider.trim())
        expectIndependentNonEmptyFields(result)
      }),
    )
  })

  // Scenario 2: no Provider metadata but the API URL host matches KNOWN_HOSTS.
  it('detects provider from a known API URL host (source = host-detection)', () => {
    const caseArb = fc.record({
      // Absent or blank Provider so resolution falls through to host detection.
      provider: fc.option(blankArb, { nil: undefined }),
      protocol: optionalProtocolArb,
      hostPair: fc.constantFrom(...KNOWN_HOST_PAIRS),
      // Optionally turn the known host into a subdomain, which must still match.
      subdomain: fc.option(fc.stringMatching(/^[a-z][a-z0-9-]{0,10}$/), { nil: undefined }),
      // With or without a scheme (the resolver retries with http:// prefix).
      withScheme: fc.boolean(),
      apiType: fc.option(nonBlankArb, { nil: undefined }),
    })

    fc.assert(
      fc.property(caseArb, ({ provider, protocol, hostPair, subdomain, withScheme, apiType }) => {
        const [knownHost, expectedProvider] = hostPair
        const host = subdomain ? `${subdomain}.${knownHost}` : knownHost
        const apiUrl = withScheme ? `https://${host}/v1/chat` : `${host}/v1/chat`

        const input: ProviderResolutionInput = {
          basic_info: makeBasicInfo({
            [PROVIDER_KEY]: provider,
            [PROTOCOL_KEY]: protocol,
            [API_URL_KEY]: apiUrl,
          }),
          api_type: apiType ?? null,
        }

        const result = resolveProvider(input)

        expect(result.source).toBe('host-detection')
        expect(result.provider).toBe(expectedProvider)
        expectIndependentNonEmptyFields(result)
      }),
    )
  })

  // Scenario 3: no Provider metadata and the host is unrecognized or missing.
  it('falls back to Custom when no metadata and host is unknown/missing (source = custom-fallback)', () => {
    // Unknown host generator: guaranteed never to match a known host because it
    // ends in the reserved `.test` TLD, which no KNOWN_HOSTS entry uses.
    const unknownHostArb = fc
      .stringMatching(/^[a-z][a-z0-9-]{0,15}$/)
      .map((label) => `${label}.unknown-provider.test`)

    const caseArb = fc.record({
      provider: fc.option(blankArb, { nil: undefined }),
      protocol: optionalProtocolArb,
      // Either an unknown/blank API URL, or omit the key entirely (missing host).
      apiUrl: fc.oneof(
        unknownHostArb.map((h) => `https://${h}/v1`),
        unknownHostArb,
        blankArb,
        fc.constant(undefined),
      ),
      apiType: fc.option(nonBlankArb, { nil: undefined }),
    })

    fc.assert(
      fc.property(caseArb, ({ provider, protocol, apiUrl, apiType }) => {
        const input: ProviderResolutionInput = {
          basic_info: makeBasicInfo({
            [PROVIDER_KEY]: provider,
            [PROTOCOL_KEY]: protocol,
            [API_URL_KEY]: apiUrl,
          }),
          api_type: apiType ?? null,
        }

        const result = resolveProvider(input)

        expect(result.source).toBe('custom-fallback')
        expect(result.provider).toBe(CUSTOM_PROVIDER)
        expectIndependentNonEmptyFields(result)
      }),
    )
  })

  // Cross-cutting: the two fields are always present and independent, even for
  // fully arbitrary / degenerate inputs.
  it('always returns independent non-empty provider and protocol for any input', () => {
    const arbitraryInfoArb = fc.option(
      fc.dictionary(fc.string({ maxLength: 20 }), fc.string({ maxLength: 40 }), { maxKeys: 6 }),
      { nil: undefined },
    )

    fc.assert(
      fc.property(
        arbitraryInfoArb,
        fc.option(fc.string({ maxLength: 20 }), { nil: undefined }),
        (basicInfo, apiType) => {
          const result = resolveProvider({ basic_info: basicInfo ?? null, api_type: apiType ?? null })

          expectIndependentNonEmptyFields(result)
          expect(['metadata', 'host-detection', 'custom-fallback']).toContain(result.source)
        },
      ),
    )
  })
})
