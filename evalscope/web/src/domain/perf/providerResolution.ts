/**
 * Pure logic for Performance provider/protocol resolution (Performance_View).
 *
 * These helpers have no DOM, network, timer or randomness dependencies so they
 * can be exercised directly by property-based tests. They back the two
 * independent, individually-labelled `Provider` and `Protocol` fields shown in
 * the Performance views.
 *
 * Provider is resolved by a strict priority order:
 *   1. explicit provider metadata,
 *   2. known-host detection from the API URL,
 *   3. a `Custom` fallback when neither is available.
 *
 * Protocol is always resolved independently of provider so the two never
 * collapse into a single combined field.
 */

/** Which resolution step produced the returned provider. */
export type ProviderSource = 'metadata' | 'host-detection' | 'custom-fallback'

/**
 * Result of resolving a run's identity.
 *
 * `provider` and `protocol` are always populated and are independent of each
 * other: the protocol is never derived from the provider and the
 * provider is never derived from the protocol.
 */
export interface ProviderResolution {
  /** Human-facing provider label (e.g. `DashScope`, `OpenAI`, `Custom`). */
  provider: string
  /** Protocol label shown separately from provider (e.g. `OpenAI-compatible`). */
  protocol: string
  /** Which priority step produced `provider`. */
  source: ProviderSource
}

/**
 * Structural input accepted by {@link resolveProvider}.
 *
 * The provider metadata, protocol and API URL that drive resolution live in a
 * perf run's `basic_info` map (see `PerfDetailResponse`), with `api_type` as a
 * secondary hint for the protocol. This interface captures exactly the fields
 * the resolver reads, so any object carrying them (including
 * `PerfDetailResponse`) satisfies it structurally.
 */
export interface ProviderResolutionInput {
  /** Native `basic_info` map from the perf run (keys such as `Provider`, `Protocol`, `API URL`). */
  basic_info?: Record<string, string> | null
  /** Optional explicit provider returned by the archive list endpoint. */
  provider?: string | null
  /** Optional explicit protocol returned by the archive list endpoint. */
  protocol?: string | null
  /** Sanitized endpoint hostname returned by the archive list endpoint. */
  api_host?: string | null
  /** Backend api_type (e.g. `openai_api`), used only as a protocol fallback hint. */
  api_type?: string | null
}

/** Fallback provider label when no explicit metadata or known host matches. */
export const CUSTOM_PROVIDER = 'Custom'

/** Fallback protocol label when no explicit protocol metadata is present. */
export const DEFAULT_PROTOCOL = 'OpenAI-compatible'

/** `basic_info` keys the resolver reads. */
const PROVIDER_KEY = 'Provider'
const PROTOCOL_KEY = 'Protocol'
const API_URL_KEY = 'API URL'
const API_HOST_KEY = 'API Host'

/**
 * Known host → provider mapping used for host detection.
 *
 * Keys are bare hostnames (no scheme/port). A run's API URL host matches a key
 * when it equals the key exactly or is a subdomain of it.
 */
const KNOWN_HOSTS: Record<string, string> = {
  'dashscope.aliyuncs.com': 'DashScope',
  'dashscope-intl.aliyuncs.com': 'DashScope',
  'api.openai.com': 'OpenAI',
  'api.anthropic.com': 'Anthropic',
  'generativelanguage.googleapis.com': 'Google',
  'api.deepseek.com': 'DeepSeek',
  'api.moonshot.cn': 'Moonshot',
  'openrouter.ai': 'OpenRouter',
  'api.together.xyz': 'Together',
  'api.mistral.ai': 'Mistral',
  'api.groq.com': 'Groq',
}

/** Read a trimmed, non-empty string from `basic_info`, or `undefined`. */
function readInfo(basicInfo: Record<string, string> | null | undefined, key: string): string | undefined {
  const raw = basicInfo?.[key]
  if (typeof raw !== 'string') {
    return undefined
  }
  const trimmed = raw.trim()
  return trimmed.length > 0 ? trimmed : undefined
}

/**
 * Extract a normalized hostname from an API URL string.
 *
 * Handles URLs missing a scheme (e.g. `localhost:8000/v1`) by retrying with an
 * `http://` prefix. Returns `undefined` when the value cannot be parsed into a
 * host. The result is lowercased with any port stripped.
 */
function extractHost(apiUrl: string | undefined): string | undefined {
  if (!apiUrl) {
    return undefined
  }
  const candidates = [apiUrl, `http://${apiUrl}`]
  for (const candidate of candidates) {
    try {
      const { hostname } = new URL(candidate)
      if (hostname) {
        return hostname.toLowerCase()
      }
    } catch {
      // Try the next candidate form.
    }
  }
  return undefined
}

/**
 * Detect a known provider from an API URL host.
 *
 * Matches when the host equals a known host exactly or is a subdomain of one.
 * Returns `undefined` when the host is absent or unrecognized so the caller can
 * fall through to the `Custom` fallback.
 */
function detectProviderFromHost(apiUrl: string | undefined): string | undefined {
  const host = extractHost(apiUrl)
  if (!host) {
    return undefined
  }
  for (const [knownHost, provider] of Object.entries(KNOWN_HOSTS)) {
    if (host === knownHost || host.endsWith(`.${knownHost}`)) {
      return provider
    }
  }
  return undefined
}

/**
 * Resolve the protocol label independently of the provider.
 *
 * Prefers explicit `Protocol` metadata, then falls back to a default protocol.
 * `api_type` is reserved as a hint but all currently supported backends speak
 * the OpenAI-compatible protocol.
 */
function resolveProtocol(input: ProviderResolutionInput): string {
  const explicit = input.protocol?.trim() || readInfo(input.basic_info, PROTOCOL_KEY)
  if (explicit) {
    return explicit
  }
  return DEFAULT_PROTOCOL
}

/**
 * Resolve a run's provider and protocol into two independent fields.
 *
 * Provider resolution follows the priority order:
 *   1. explicit `Provider` metadata → `source = 'metadata'`;
 *   2. otherwise a known host detected from the API URL → `source = 'host-detection'`;
 *   3. otherwise `Custom` → `source = 'custom-fallback'`.
 *
 * The returned `protocol` is always resolved independently of `provider`, and
 * both fields are always present.
 *
 * @param run - Structural perf run input carrying `basic_info` and `api_type`.
 * @returns The resolved provider, protocol and the source that produced the provider.
 */
export function resolveProvider(run: ProviderResolutionInput): ProviderResolution {
  const protocol = resolveProtocol(run)

  // Priority 1: explicit provider metadata.
  const explicitProvider = run.provider?.trim() || readInfo(run.basic_info, PROVIDER_KEY)
  if (explicitProvider) {
    return { provider: explicitProvider, protocol, source: 'metadata' }
  }

  // Priority 2: known-host detection from the API URL.
  const detected = detectProviderFromHost(
    run.api_host?.trim()
      || readInfo(run.basic_info, API_HOST_KEY)
      || readInfo(run.basic_info, API_URL_KEY),
  )
  if (detected) {
    return { provider: detected, protocol, source: 'host-detection' }
  }

  // Priority 3: Custom fallback.
  return { provider: CUSTOM_PROVIDER, protocol, source: 'custom-fallback' }
}
