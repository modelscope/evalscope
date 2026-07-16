/**
 * HTTP API client for the Web Console.
 *
 * Every exported request helper validates the parsed response with a zod schema
 * and rejects with a typed {@link DomainError} on failure, so consumers never
 * receive unvalidated data.
 *
 * All outward methods accept an optional `AbortSignal` for cancellation.
 * Failures are normalised to {@link DomainError} categories:
 * `http-4xx` / `http-5xx` for HTTP error statuses, `network` for transport
 * failures, and `aborted` when a request is cancelled via its signal.
 *
 * See design section 11 and Requirement 13.
 */
import type { ZodType } from 'zod'

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

/**
 * Run a parsed value through a zod schema, converting validation failures into
 * a typed {@link DomainError} (`kind='validation'`) so consumers never receive
 * unvalidated data and no uncaught exception escapes.
 */
function validate<T>(schema: ZodType<T>, data: unknown): T {
  const result = schema.safeParse(data)
  if (!result.success) {
    throw new DomainError('validation', `Response validation failed: ${result.error.message}`)
  }
  return result.data
}

/**
 * GET a JSON resource and validate it against `schema` at runtime.
 *
 * On schema mismatch this rejects with `DomainError(kind='validation')` rather
 * than returning unvalidated data. HTTP, network, and abort
 * failures reject with their respective {@link DomainError} kinds. It never
 * throws synchronously, so consumers can handle all failures via the returned
 * promise.
 */
export async function apiValidated<T>(path: string, schema: ZodType<T>, options?: RequestOptions): Promise<T> {
  const res = await doFetch(buildUrl(path, options?.params), { signal: options?.signal })
  await ensureOk(res)
  const data = await parseJson<unknown>(res)
  return validate(schema, data)
}

/**
 * POST a JSON body and validate the response against `schema` at runtime.
 *
 * Mirrors {@link apiValidated} for mutating endpoints: schema mismatch rejects
 * with `DomainError(kind='validation')`.
 */
export async function apiPostValidated<T>(
  path: string,
  body: unknown,
  schema: ZodType<T>,
  options?: PostRequestOptions,
): Promise<T> {
  const res = await doFetch(buildUrl(path, options?.params), {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', ...options?.headers },
    body: JSON.stringify(body),
    signal: options?.signal,
  })
  await ensureOk(res)
  const data = await parseJson<unknown>(res)
  return validate(schema, data)
}
