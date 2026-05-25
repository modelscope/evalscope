import type { ReportSummary } from '@/api/types'
import ScoreBadge from '@/components/ui/ScoreBadge'
import ScoreChip from '@/components/ui/ScoreChip'

interface EvalRunCardProps {
  report: ReportSummary
  onClick: () => void
}

/** Format ISO-ish timestamp `2025-05-22T14:30:00…` → `2025-05-22 14:30:00`. */
function formatTimestamp(ts: string): string {
  return ts.replace('T', ' ').slice(0, 19)
}

/**
 * EvalRunCard — single eval-row in the dashboard timeline view.
 * DESIGN.md `{components.eval-run-card}`: borderless row-card with model name + ScoreBadge on top,
 * timestamp + dataset ScoreChips on the second row. Hover changes border tint only (no lift).
 */
export default function EvalRunCard({ report, onClick }: EvalRunCardProps) {
  const dsScores = report.dataset_scores

  return (
    <button
      onClick={onClick}
      className="rounded-[var(--radius)] border border-[var(--border)] bg-[var(--bg-card)] hover:border-[var(--border-md)] transition-colors p-4 sm:p-5 cursor-pointer w-full text-left"
    >
      <div className="flex items-center gap-2">
        <span className="type-title-md text-[var(--text)] truncate flex-1 min-w-0">
          {report.model_name}
        </span>
        <ScoreBadge score={report.score} className="shrink-0" />
      </div>

      <div className="flex items-center gap-2 mt-2 flex-wrap">
        {report.timestamp && (
          <span className="type-caption-mono text-[var(--text-muted)] shrink-0">
            {formatTimestamp(report.timestamp)}
          </span>
        )}
        <div className="flex flex-wrap gap-1.5">
          {dsScores && Object.keys(dsScores).length > 0 ? (
            Object.entries(dsScores).map(([ds, s]) => <ScoreChip key={ds} label={ds} score={s} />)
          ) : (
            <span className="type-caption-mono text-[var(--text-muted)]">{report.dataset_name}</span>
          )}
        </div>
      </div>
    </button>
  )
}
