import { useState } from 'react'
import { useLocale } from '@/contexts/LocaleContext'
import { decisionObservation } from '@/domain/report/decisionDiagnostics'
import { formatMetric } from '@/domain/metric'
import { RATIO_PERCENT_SEMANTICS } from '@/domain/report/primaryMetrics'
import Card from '@/components/ui/Card'
import Badge from '@/components/ui/Badge'
import SegmentedControl from '@/components/ui/SegmentedControl'

export default function DecisionPanel({ metadata }: { metadata: unknown }) {
  const { t } = useLocale()
  const [distribution, setDistribution] = useState('scoring')
  const decision = decisionObservation(metadata)
  if (!decision) return null
  const rawAvailable = decision.rows.every((row) => row.raw !== null)

  return (
    <Card title={t('reportDetail.decisionProbabilities')} badge={<Badge>{decision.kind.toUpperCase()}</Badge>}>
      <div className="flex flex-col gap-5">
        <dl className="grid grid-cols-2 gap-4 rounded-[var(--radius-sm)] bg-[var(--bg-card2)] p-4">
          <div className="min-w-0">
            <dt className="text-xs text-[var(--text-muted)]">{t(decision.kind === 'score' ? 'reportDetail.expectedScore' : 'reportDetail.decisionPrediction')}</dt>
            <dd className="mt-1 break-words text-xl font-semibold tabular-nums text-[var(--accent)]">{decision.expected !== null ? formatMetric(decision.expected, null).primary : decision.predicted}</dd>
          </div>
          <div className="min-w-0 border-l border-[var(--border)] pl-4">
            <dt className="text-xs text-[var(--text-muted)]">{t('reportDetail.decisionGold')}</dt>
            <dd className="mt-1 break-words text-xl font-semibold tabular-nums">{decision.gold}</dd>
          </div>
        </dl>
        <div className="flex flex-wrap items-center justify-between gap-2">
          <span className="text-xs text-[var(--text-muted)]">{t('reportDetail.distribution')}</span>
          {rawAvailable ? (
            <SegmentedControl
              ariaLabel={t('reportDetail.distribution')}
              value={distribution}
              onChange={setDistribution}
              size="sm"
              options={[
                { value: 'scoring', label: t('reportDetail.scoringView'), accessibleLabel: t('reportDetail.scoringProbabilities') },
                { value: 'raw', label: t('reportDetail.rawView'), accessibleLabel: t('reportDetail.rawProbabilities') },
              ]}
            />
          ) : <span className="text-xs">{t('reportDetail.scoringProbabilities')}</span>}
        </div>
        <div className="flex flex-col gap-4">
          {decision.rows.map((row) => {
            const probability = distribution === 'raw' && rawAvailable ? row.raw! : row.probability
            const formatted = formatMetric(probability, RATIO_PERCENT_SEMANTICS).primary
            return (
              <div key={row.label} className="min-w-0">
                <div className="mb-1.5 flex items-start justify-between gap-3 text-sm">
                  <div className="flex min-w-0 flex-wrap items-center gap-2">
                    <span className="break-words font-medium">{row.label}</span>
                    {row.label === decision.predicted && <Badge>{t(decision.kind === 'score' ? 'reportDetail.mostLikelyLevel' : 'reportDetail.decisionPrediction')}</Badge>}
                    {!decision.continuous && row.label === decision.gold && <span className="rounded-full border border-[var(--border-md)] px-2 py-0.5 text-xs text-[var(--text-muted)]">{t('reportDetail.decisionGold')}</span>}
                  </div>
                  <span className="shrink-0 font-mono tabular-nums">{formatted}</span>
                </div>
                <div
                  role="meter" aria-label={row.label}
                  aria-valuemin={0} aria-valuemax={1} aria-valuenow={probability} aria-valuetext={formatted}
                  className="h-2 overflow-hidden rounded-full bg-[var(--bg-deep)]"
                >
                  <div className="h-full rounded-full bg-[var(--accent)]" style={{ width: `${probability * 100}%` }} />
                </div>
              </div>
            )
          })}
        </div>
        <div className="flex flex-col gap-1.5 border-t border-[var(--border)] pt-3 text-xs text-[var(--text-muted)]">
          {decision.kind === 'noul' && decision.threshold !== null && (
            <p>{t('reportDetail.noulThreshold')}: {decision.threshold}</p>
          )}
          {decision.confidence !== null && (
            <p>{t('reportDetail.providerConfidence')}: <span className="font-mono tabular-nums">{formatMetric(decision.confidence, null).primary}</span> · {t('reportDetail.confidenceHint')}</p>
          )}
          <p>{t('reportDetail.probabilityScaleHint')}</p>
        </div>
      </div>
    </Card>
  )
}
