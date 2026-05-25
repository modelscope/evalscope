import { useState, type CSSProperties } from 'react'
import { createPortal } from 'react-dom'
import { X } from 'lucide-react'

interface Props {
  src: string
  alt?: string
  /** Inline style for the inline thumbnail. */
  style?: CSSProperties
  /** Tailwind class for the inline thumbnail. */
  className?: string
}

const DEFAULT_INLINE = 'rounded-lg cursor-zoom-in border border-[var(--border)] hover:border-[var(--border-strong)] transition-all hover:scale-[1.02]'

/** Click-to-zoom image with portal-rendered overlay. Used by both Markdown img and ContentBlock images. */
export default function ImageLightbox({ src, alt = '', style, className }: Props) {
  const [open, setOpen] = useState(false)
  if (!src) return null
  return (
    <>
      <img
        src={src}
        alt={alt}
        onClick={() => setOpen(true)}
        className={className ?? DEFAULT_INLINE}
        style={style}
      />
      {open && createPortal(
        <div
          className="fixed inset-0 z-[9999] flex items-center justify-center"
          style={{ background: 'var(--overlay-bg)', backdropFilter: 'blur(6px)' }}
          onClick={() => setOpen(false)}
        >
          <div className="relative max-w-[90vw] max-h-[90vh]" onClick={(e) => e.stopPropagation()}>
            <button
              onClick={() => setOpen(false)}
              className="absolute -top-3 -right-3 z-10 rounded-full p-1 bg-[var(--bg-card)] border border-[var(--border)] hover:bg-[var(--bg-card2)] transition-colors"
            >
              <X size={16} />
            </button>
            <img
              src={src}
              alt={alt}
              className="max-w-full max-h-[85vh] rounded-xl object-contain shadow-2xl"
            />
          </div>
        </div>,
        document.body,
      )}
    </>
  )
}
