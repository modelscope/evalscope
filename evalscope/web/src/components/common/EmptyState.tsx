import { Inbox } from 'lucide-react'

export default function EmptyState({ text }: { text?: string }) {
  return (
    <div className="flex flex-col items-center justify-center gap-2 py-16 text-[var(--color-ink-muted)]">
      <Inbox size={40} strokeWidth={1} />
      <span className="text-sm">{text || 'No data available'}</span>
    </div>
  )
}
