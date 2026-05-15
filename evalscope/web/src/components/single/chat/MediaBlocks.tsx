import React, { useState } from 'react'
import { createPortal } from 'react-dom'
import { ChevronDown, ChevronRight, X } from 'lucide-react'
import type { ContentBlock } from '@/api/types'
import { useLocale } from '@/contexts/LocaleContext'
import MarkdownRenderer from '@/components/common/MarkdownRenderer'

/**
 * Convert a server-side local file path or URL to a browser-accessible URL.
 *
 * - HTTP(S) URLs and data-URIs are returned as-is.
 * - Base64 strings (no path separator) are returned as-is for the caller to
 *   wrap in a data-URI.
 * - Absolute local paths (starting with '/' or a Windows drive letter) are
 *   proxied through the server's media file endpoint so the browser can
 *   fetch them over HTTP.
 */
function toMediaUrl(src: string): string {
  if (
    src.startsWith('http://') ||
    src.startsWith('https://') ||
    src.startsWith('data:')
  ) {
    return src
  }
  // Absolute POSIX path or Windows path → proxy through server
  if (src.startsWith('/') || /^[A-Za-z]:[/\\]/.test(src)) {
    return `/api/v1/reports/media/file?path=${encodeURIComponent(src)}`
  }
  // Otherwise treat as base64 payload
  return src
}

export function ImageBlock({ src }: { src: string }) {
  const [open, setOpen] = useState(false)
  const resolved = toMediaUrl(src)
  const imgSrc = resolved.startsWith('data:') || resolved.startsWith('http') || resolved.startsWith('/')
    ? resolved
    : `data:image/jpeg;base64,${resolved}`
  return (
    <div style={{ marginTop: '0.5rem', marginBottom: '0.25rem' }}>
      <img
        src={imgSrc}
        alt=""
        onClick={() => setOpen(true)}
        className="cursor-zoom-in hover:scale-[1.02] transition-transform"
        style={{
          maxWidth: '100%',
          maxHeight: '360px',
          borderRadius: '0.5rem',
          border: '1px solid var(--color-border-subtle)',
          display: 'block',
        }}
      />
      {open && createPortal(
        <div
          className="fixed inset-0 z-[9999] flex items-center justify-center"
          style={{ background: 'var(--overlay-bg)', backdropFilter: 'blur(6px)' }}
          onClick={() => setOpen(false)}
        >
          <div className="relative max-w-[90vw] max-h-[90vh]" onClick={e => e.stopPropagation()}>
            <button
              onClick={() => setOpen(false)}
              className="absolute -top-3 -right-3 z-10 rounded-full p-1 bg-[var(--color-surface)] border border-[var(--color-border)] hover:bg-[var(--color-surface-hover)] transition-colors"
            >
              <X size={16} />
            </button>
            <img
              src={imgSrc}
              alt=""
              className="max-w-full max-h-[85vh] rounded-xl object-contain shadow-2xl"
            />
          </div>
        </div>,
        document.body
      )}
    </div>
  )
}

export function AudioBlock({ src, format }: { src: string; format?: string }) {
  const mimeType = format === 'mp3' ? 'audio/mpeg' : format === 'wav' ? 'audio/wav' : 'audio/mpeg'
  const resolved = toMediaUrl(src)
  const audioSrc = resolved.startsWith('data:') || resolved.startsWith('http') || resolved.startsWith('/')
    ? resolved
    : `data:${mimeType};base64,${resolved}`
  return (
    <div style={{ marginTop: '0.5rem', marginBottom: '0.25rem' }}>
      {/* eslint-disable-next-line jsx-a11y/media-has-caption */}
      <audio controls src={audioSrc} style={{ width: '100%', borderRadius: '0.4rem' }} />
    </div>
  )
}

export function VideoBlock({ src, format }: { src: string; format?: string }) {
  const resolved = toMediaUrl(src)
  const mimeType = format === 'webm' ? 'video/webm'
    : format === 'ogg' || format === 'ogv' ? 'video/ogg'
    : 'video/mp4'
  const videoSrc = resolved.startsWith('data:') || resolved.startsWith('http') || resolved.startsWith('/')
    ? resolved
    : `data:${mimeType};base64,${resolved}`
  return (
    <div style={{ marginTop: '0.5rem', marginBottom: '0.25rem' }}>
      {/* eslint-disable-next-line jsx-a11y/media-has-caption */}
      <video
        controls
        src={videoSrc}
        style={{
          maxWidth: '100%',
          maxHeight: '360px',
          borderRadius: '0.5rem',
          border: '1px solid var(--color-border-subtle)',
          display: 'block',
          background: 'var(--media-video-bg)',
        }}
      />
    </div>
  )
}

/** Collapsible reasoning block rendered above the main answer. */
export function ReasoningBlock({ text, tokens }: { text: string; tokens?: number }) {
  const { t } = useLocale()
  const [open, setOpen] = useState(false)
  return (
    <div
      style={{
        marginBottom: '0.5rem',
        borderRadius: '0.5rem',
        border: '1px solid var(--bubble-reasoning-border)',
        background: 'var(--bubble-reasoning-bg)',
        overflow: 'hidden',
      }}
    >
      <button
        onClick={() => setOpen(o => !o)}
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: '0.35rem',
          width: '100%',
          padding: '0.35rem 0.7rem',
          background: 'none',
          border: 'none',
          cursor: 'pointer',
          color: 'var(--bubble-bot-color)',
          fontSize: '0.7rem',
          fontWeight: 600,
          letterSpacing: '0.04em',
        }}
      >
        {open ? <ChevronDown size={12} /> : <ChevronRight size={12} />}
        {open ? t('prediction.hideReasoning') : t('prediction.showReasoning')}
        <span style={{ opacity: 0.5, fontWeight: 400 }}>
          · {tokens != null ? `${tokens} tokens` : `${text.length} chars`}
        </span>
      </button>
      {open && (
        <div
          style={{
            padding: '0.5rem 0.75rem 0.75rem',
            fontSize: '0.8rem',
            color: 'var(--text)',
            borderTop: '1px solid var(--bubble-reasoning-border)',
          }}
        >
          <MarkdownRenderer content={text} />
        </div>
      )}
    </div>
  )
}

/** Render ContentBlock[] into React nodes. */
export function renderContentBlocks(
  blocks: ContentBlock[],
  opts: { includeReasoning?: boolean } = {},
): React.ReactNode[] {
  const nodes: React.ReactNode[] = []
  blocks.forEach((b, i) => {
    if (b.type === 'reasoning' && opts.includeReasoning) {
      nodes.push(<ReasoningBlock key={`r${i}`} text={b.reasoning ?? ''} tokens={b.reasoning_tokens} />)
    } else if (b.type === 'text') {
      if (b.text) nodes.push(<MarkdownRenderer key={`t${i}`} content={b.text} />)
    } else if (b.type === 'image') {
      if (b.image) nodes.push(<ImageBlock key={`img${i}`} src={b.image} />)
    } else if (b.type === 'audio') {
      if (b.audio) nodes.push(<AudioBlock key={`aud${i}`} src={b.audio} format={b.format} />)
    } else if (b.type === 'video') {
      if (b.video) nodes.push(<VideoBlock key={`vid${i}`} src={b.video} format={b.format} />)
    }
  })
  return nodes
}
