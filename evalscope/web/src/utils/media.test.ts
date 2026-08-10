import { describe, expect, it } from 'vitest'

import { resolveMarkdownMediaSrc, resolveMediaSrc } from './media'

const PROXY = '/api/v1/reports/media/file?path='

describe('resolveMediaSrc', () => {
  it('proxies absolute posix and windows paths', () => {
    expect(resolveMediaSrc('/tmp/a.png', 'image/jpeg')).toBe(`${PROXY}${encodeURIComponent('/tmp/a.png')}`)
    expect(resolveMediaSrc('C:\\tmp\\a.png', 'image/jpeg')).toBe(`${PROXY}${encodeURIComponent('C:\\tmp\\a.png')}`)
  })

  it('passes remote sources through regardless of scheme case', () => {
    for (const src of ['http://e.com/a.png', 'HTTPS://e.com/a.png', 'data:image/png;base64,x', 'blob:abc-123']) {
      expect(resolveMediaSrc(src, 'image/jpeg')).toBe(src)
    }
  })

  it('treats anything else as a base64 payload', () => {
    expect(resolveMediaSrc('aGVsbG8=', 'image/png')).toBe('data:image/png;base64,aGVsbG8=')
  })
})

describe('resolveMarkdownMediaSrc', () => {
  it('decodes a percent-encoded destination before proxying it', () => {
    const raw = '/tmp/my dir/\u56fe.png'
    expect(resolveMarkdownMediaSrc(encodeURI(raw), 'image/jpeg')).toBe(`${PROXY}${encodeURIComponent(raw)}`)
  })

  it('round-trips a path whose name contains a percent sign', () => {
    const raw = '/tmp/100%.png'
    expect(resolveMarkdownMediaSrc(encodeURI(raw), 'image/jpeg')).toBe(`${PROXY}${encodeURIComponent(raw)}`)
  })

  it('leaves a protocol-relative url alone instead of proxying it as a path', () => {
    expect(resolveMarkdownMediaSrc('//cdn.example.com/a.png', 'image/jpeg')).toBe('//cdn.example.com/a.png')
  })

  it('falls back to the raw value when the escape sequence is malformed', () => {
    // A lone '%' makes decodeURI throw; the src must still be resolved.
    expect(resolveMarkdownMediaSrc('/tmp/a%zz.png', 'image/jpeg')).toBe(`${PROXY}${encodeURIComponent('/tmp/a%zz.png')}`)
  })
})
