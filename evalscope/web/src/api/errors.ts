/**
 * Typed domain errors for the API client.
 *
 * These provide a machine-discriminable error category (`kind`) so consumers can
 * branch on the failure mode (e.g. show a validation error state, silently ignore
 * an aborted request) without parsing error messages. See design section 11.
 */

/**
 * Machine-discriminable category of a {@link DomainError}.
 *
 * - `validation`: response failed runtime schema validation.
 * - `http-4xx`: server responded with a 4xx client error status.
 * - `http-5xx`: server responded with a 5xx server error status.
 * - `network`: the request never completed (e.g. connection failure).
 * - `aborted`: the request was cancelled via an `AbortSignal`.
 */
export type DomainErrorKind = 'validation' | 'http-4xx' | 'http-5xx' | 'network' | 'aborted'

/**
 * Error type raised (or returned) by the API client with a machine-discriminable
 * `kind` and an optional HTTP `status`.
 */
export class DomainError extends Error {
  /** Machine-discriminable error category. */
  readonly kind: DomainErrorKind
  /** HTTP status code when the failure originated from an HTTP response. */
  readonly status?: number

  constructor(kind: DomainErrorKind, message: string, status?: number) {
    super(message)
    this.name = 'DomainError'
    this.kind = kind
    this.status = status
    // Restore the prototype chain: extending built-ins (Error) breaks `instanceof`
    // when compiled down to ES5-style output, so pin it explicitly.
    Object.setPrototypeOf(this, DomainError.prototype)
  }
}

/**
 * Type guard narrowing an unknown value to {@link DomainError}.
 */
export function isDomainError(e: unknown): e is DomainError {
  return e instanceof DomainError
}
