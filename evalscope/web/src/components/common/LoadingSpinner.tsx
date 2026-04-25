export default function LoadingSpinner({ text }: { text?: string }) {
  return (
    <div className="flex items-center justify-center gap-2 py-12 text-[var(--color-ink-muted)]">
      <div className="h-5 w-5 animate-spin rounded-full border-2 border-current border-t-transparent" />
      {text && <span className="text-sm">{text}</span>}
    </div>
  )
}
