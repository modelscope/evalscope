import { useState, type CSSProperties, type ReactNode } from 'react'
import { ChevronDown, ChevronRight } from 'lucide-react'

interface Props {
  /** Header content rendered to the right of the chevron. May be a render fn for open-aware headers. */
  header: ReactNode | ((open: boolean) => ReactNode)
  children: ReactNode
  /** Initial open state when uncontrolled. */
  defaultOpen?: boolean
  /** Wrapper style (border, background, etc.). */
  style?: CSSProperties
  /** Header button style. */
  headerStyle?: CSSProperties
  /** Body wrapper style. */
  bodyStyle?: CSSProperties
  /** Chevron icon size (px). */
  chevronSize?: number
  /** Chevron color (CSS color or var). */
  chevronColor?: string
}

/** Lightweight chevron + content collapsible. The 4 chat collapsibles in chat/ all share this shape. */
export default function Collapsible({
  header,
  children,
  defaultOpen = false,
  style,
  headerStyle,
  bodyStyle,
  chevronSize = 12,
  chevronColor = 'var(--text-muted)',
}: Props) {
  const [open, setOpen] = useState(defaultOpen)
  const Chevron = open ? ChevronDown : ChevronRight
  return (
    <div style={style}>
      <button
        onClick={() => setOpen((v) => !v)}
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: '0.4rem',
          width: '100%',
          background: 'none',
          border: 'none',
          cursor: 'pointer',
          padding: '0.2rem 0',
          textAlign: 'left',
          ...headerStyle,
        }}
      >
        <Chevron size={chevronSize} style={{ color: chevronColor, flexShrink: 0 }} />
        {typeof header === 'function' ? header(open) : header}
      </button>
      {open && <div style={bodyStyle}>{children}</div>}
    </div>
  )
}
