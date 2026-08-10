/**
 * Resolve a server-side path / URL / base64 payload to a browser-loadable src.
 *
 * - http(s) and data: URIs are returned as-is.
 * - Absolute POSIX/Windows paths are proxied through the media file endpoint.
 * - Anything else is treated as base64 and wrapped in a data: URI with `mimeType`.
 *
 * Shared by both rendering chains so that a local media path shows up the same
 * way in structured `ContentBlock` messages and in Markdown text (e.g. the
 * `Input` column produced by `messages_to_markdown`).
 */
export function resolveMediaSrc(src: string, mimeType: string): string {
  if (src.startsWith('http://') || src.startsWith('https://') || src.startsWith('data:')) return src
  if (src.startsWith('/') || /^[A-Za-z]:[/\\]/.test(src)) {
    return `/api/v1/reports/media/file?path=${encodeURIComponent(src)}`
  }
  return `data:${mimeType};base64,${src}`
}
