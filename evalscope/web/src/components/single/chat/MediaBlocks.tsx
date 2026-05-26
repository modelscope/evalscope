import React, { useState } from 'react'
import { ChevronDown, ChevronRight } from 'lucide-react'
import type { ContentBlock } from '@/api/types'
import { useLocale } from '@/contexts/LocaleContext'
import MarkdownRenderer from '@/components/common/MarkdownRenderer'
import ImageLightbox from '@/components/common/ImageLightbox'
import ChatBubble, { bubbleAccent } from '@/components/ui/ChatBubble'

/**
 * Resolve a server-side path / URL / base64 payload to a browser-loadable src.
 *
 * - http(s) and data: URIs are returned as-is.
 * - Absolute POSIX/Windows paths are proxied through the media file endpoint.
 * - Anything else is treated as base64 and wrapped in a data: URI with `mimeType`.
 */
function resolveMediaSrc(src: string, mimeType: string): string {
  if (src.startsWith('http://') || src.startsWith('https://') || src.startsWith('data:')) return src
  if (src.startsWith('/') || /^[A-Za-z]:[/\\]/.test(src)) {
    return `/api/v1/reports/media/file?path=${encodeURIComponent(src)}`
  }
  return `data:${mimeType};base64,${src}`
}

const IMG_INLINE_CLASS = 'cursor-zoom-in hover:scale-[1.02] transition-transform max-w-full max-h-[360px] rounded-lg border border-[var(--border)] block'

export function ImageBlock({ src }: { src: string }) {
  return (
    <div className="mt-2 mb-1">
      <ImageLightbox
        src={resolveMediaSrc(src, 'image/jpeg')}
        className={IMG_INLINE_CLASS}
      />
    </div>
  )
}

const AUDIO_MIMES: Record<string, string> = { mp3: 'audio/mpeg', wav: 'audio/wav' }

export function AudioBlock({ src, format }: { src: string; format?: string }) {
  const mimeType = AUDIO_MIMES[format ?? ''] ?? 'audio/mpeg'
  return (
    <div className="mt-2 mb-1">
      {/* eslint-disable-next-line jsx-a11y/media-has-caption */}
      <audio controls src={resolveMediaSrc(src, mimeType)} className="w-full rounded-[0.4rem]" />
    </div>
  )
}

const VIDEO_MIMES: Record<string, string> = { webm: 'video/webm', ogg: 'video/ogg', ogv: 'video/ogg' }

export function VideoBlock({ src, format }: { src: string; format?: string }) {
  const mimeType = VIDEO_MIMES[format ?? ''] ?? 'video/mp4'
  return (
    <div className="mt-2 mb-1">
      {/* eslint-disable-next-line jsx-a11y/media-has-caption */}
      <video
        controls
        src={resolveMediaSrc(src, mimeType)}
        className="max-w-full max-h-[360px] rounded-lg border border-[var(--border)] block bg-[var(--media-video-bg)]"
      />
    </div>
  )
}

/** Collapsible reasoning block rendered above the main answer. */
export function ReasoningBlock({ text, tokens }: { text: string; tokens?: number }) {
  const { t } = useLocale()
  const [open, setOpen] = useState(false)
  return (
    <ChatBubble role="reasoning" variant="card" className="mb-2 overflow-hidden">
      <button
        onClick={() => setOpen(o => !o)}
        className="flex items-center gap-1.5 w-full px-2.5 py-1.5 bg-transparent border-none cursor-pointer text-xs font-semibold"
        style={{ color: bubbleAccent('reasoning'), letterSpacing: '0.04em' }}
      >
        {open ? <ChevronDown size={12} /> : <ChevronRight size={12} />}
        {open ? t('prediction.hideReasoning') : t('prediction.showReasoning')}
        <span className="opacity-50 font-normal">
          · {tokens != null ? `${tokens} tokens` : `${text.length} chars`}
        </span>
      </button>
      {open && (
        <div className="px-3 pb-3 pt-2 text-[0.8rem] text-[var(--text)] border-t border-[var(--border)]">
          <MarkdownRenderer content={text} />
        </div>
      )}
    </ChatBubble>
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
