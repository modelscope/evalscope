/** Schemes that already point at something the browser can load directly. */
const REMOTE_SCHEME_RE = /^(?:https?|data):/i

/**
 * Resolve a server-side path / URL / base64 payload to a browser-loadable src.
 *
 * - http(s) and data: URIs are returned as-is (the scheme match is case-insensitive).
 * - Absolute POSIX/Windows paths are proxied through the media file endpoint.
 * - Anything else is treated as base64 and wrapped in a data: URI with `mimeType`.
 *
 * Shared by both rendering chains so that a local media path shows up the same
 * way in structured `ContentBlock` messages and in Markdown text (e.g. the
 * `Input` column produced by `messages_to_markdown`).
 */
export function resolveMediaSrc(src: string, mimeType: string): string {
  if (REMOTE_SCHEME_RE.test(src)) return src
  if (src.startsWith('/') || /^[A-Za-z]:[/\\]/.test(src)) {
    return `/api/v1/reports/media/file?path=${encodeURIComponent(src)}`
  }
  return `data:${mimeType};base64,${src}`
}

/**
 * Resolve an image src taken from rendered Markdown.
 *
 * A Markdown parser percent-encodes the link destination, so a local path
 * containing spaces, non-ASCII characters or backslashes arrives here already
 * encoded (`/tmp/my img.jpg` -> `/tmp/my%20img.jpg`).  It has to be decoded back
 * to the real filesystem path before the media endpoint can look it up,
 * otherwise the proxy receives the literal escape sequences.
 *
 * Markdown content is authored text rather than a media field, so anything that
 * looks like a URL is passed through untouched - including protocol-relative
 * URLs and intentionally escaped remote URLs.
 */
export function resolveMarkdownMediaSrc(src: string, mimeType: string): string {
  if (REMOTE_SCHEME_RE.test(src) || src.startsWith('//')) return src
  try {
    return resolveMediaSrc(decodeURI(src), mimeType)
  } catch {
    // Malformed escape sequence – fall back to the raw value.
    return resolveMediaSrc(src, mimeType)
  }
}
