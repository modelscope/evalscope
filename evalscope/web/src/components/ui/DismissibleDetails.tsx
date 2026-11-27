import { useEffect, useRef, type ReactNode } from 'react'

/** A native disclosure for menus that close on outside click or Escape. */
export default function DismissibleDetails({ className, children }: { className?: string; children: ReactNode }) {
  const ref = useRef<HTMLDetailsElement>(null)

  useEffect(() => {
    const details = ref.current
    if (!details) return
    const document = details.ownerDocument

    const dismissOnOutsideClick = (event: MouseEvent) => {
      if (details.open && event.target instanceof Node && !details.contains(event.target)) {
        details.open = false
      }
    }

    document.addEventListener('click', dismissOnOutsideClick, true)
    return () => document.removeEventListener('click', dismissOnOutsideClick, true)
  }, [])

  return (
    <details
      ref={ref}
      className={className}
      onKeyDown={(event) => {
        if (event.key === 'Escape') event.currentTarget.open = false
      }}
    >
      {children}
    </details>
  )
}
