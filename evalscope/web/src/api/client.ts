/**
 * HTTP API client for the Web Console.
 *
 * Every exported request helper parses JSON from responses that have already
 * been validated by the backend's Pydantic response contract.
 *
 * All outward methods accept an optional `AbortSignal` for cancellation.
 * Failures are normalised to {@link DomainError} categories:
 * `http-4xx` / `http-5xx` for HTTP error statuses, `network` for transport
 * failures, and `aborted` when a request is cancelled via its signal.
 */
import { DomainError } from './errors'

/** Options accepted by every outward request method. */
export interface RequestOptions {
  /** Query-string parameters appended to the request URL. */
  params?: Record<string, unknown>
  /** Optional signal used to cancel the in-flight request. */
  signal?: AbortSignal
}

/** Options accepted by POST request methods (adds custom headers). */
export interface PostRequestOptions extends RequestOptions {
  /** Extra request headers merged over the default JSON content type. */
  headers?: Record<string, string>
}

/**
 * Build an absolute request URL from a path and optional query params.
 *
 * Empty and `undefined` values are skipped so callers can pass optional
 * parameters without polluting the query string.
 */
function buildUrl(path: string, params?: Record<string, unknown>): string {
  const url = new URL(path, window.location.origin)
  if (params) {
    for (const [k, v] of Object.entries(params)) {
      if (v !== undefined && v !== '') url.searchParams.set(k, String(v))
    }
  }
  return url.toString()
}

/** Narrow an unknown thrown value to a cancellation (`AbortError`). */
function isAbortError(err: unknown): boolean {
  return (
    (typeof DOMException !== 'undefined' && err instanceof DOMException && err.name === 'AbortError') ||
    (err instanceof Error && err.name === 'AbortError')
  )
}

/**
 * Perform the fetch and map transport-level failures onto {@link DomainError}.
 *
 * `AbortError` becomes `kind='aborted'`; any other rejection (DNS failure,
 * connection refused, offline, CORS) becomes `kind='network'`.
 */
async function doFetch(url: string, init: RequestInit): Promise<Response> {
  try {
    return await fetch(url, init)
  } catch (err) {
    if (isAbortError(err)) {
      throw new DomainError('aborted', 'Request was aborted')
    }
    throw new DomainError('network', err instanceof Error ? err.message : 'Network request failed')
  }
}

/**
 * Validate that a response is OK; otherwise throw a typed HTTP error.
 *
 * 5xx statuses map to `kind='http-5xx'`, all other non-OK statuses map to
 * `kind='http-4xx'`. The server error message (when present in the JSON body)
 * is preserved for display.
 */
async function ensureOk(res: Response): Promise<void> {
  if (res.ok) return
  const body = await res.json().catch(() => ({ error: res.statusText }))
  const message: string = (body && typeof body.error === 'string' && body.error) || `HTTP ${res.status}`
  const kind = res.status >= 500 ? 'http-5xx' : 'http-4xx'
  throw new DomainError(kind, message, res.status)
}

/**
 * Parse a successful response as JSON, mapping malformed bodies to a network
 * error (the transport delivered something that is not the expected JSON).
 */
async function parseJson<T>(res: Response): Promise<T> {
  try {
    return (await res.json()) as T
  } catch {
    throw new DomainError('network', 'Failed to parse response body as JSON')
  }
}

export async function apiValidated<T>(path: string, options?: RequestOptions): Promise<T> {
  const res = await doFetch(buildUrl(path, options?.params), { signal: options?.signal })
  await ensureOk(res)
  return parseJson<T>(res)
}

export async function apiDeleteValidated<T>(path: string, options?: RequestOptions): Promise<T> {
  const res = await doFetch(buildUrl(path, options?.params), {
    method: 'DELETE',
    signal: options?.signal,
  })
  await ensureOk(res)
  return parseJson<T>(res)
}

/**
 * POST a JSON body and parse the successful response.
 *
 * Mirrors {@link apiValidated} for mutating endpoints.
 */
export async function apiPostValidated<T>(
  path: string,
  body: unknown,
  options?: PostRequestOptions,
): Promise<T> {
  const res = await doFetch(buildUrl(path, options?.params), {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', ...options?.headers },
    body: JSON.stringify(body),
    signal: options?.signal,
  })
  await ensureOk(res)
  return parseJson<T>(res)
}
