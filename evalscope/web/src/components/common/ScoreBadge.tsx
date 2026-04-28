interface Props {
  score: number
  threshold?: number
}

export default function ScoreBadge({ score, threshold = 0.99 }: Props) {
  const pass = score >= threshold
  return (
    <span
      className="inline-block px-2 py-0.5 rounded text-sm font-mono"
      style={{ backgroundColor: pass ? 'var(--color-pass)' : 'var(--color-fail)', color: '#fff' }}
    >
      {score.toFixed(4)}
    </span>
  )
}
