import { useState } from 'react'
import { useLocale } from '@/contexts/LocaleContext'
import { decisionObservation } from '@/domain/report/decisionDiagnostics'
import { formatMetric } from '@/domain/metric'
import { RATIO_PERCENT_SEMANTICS } from '@/domain/report/primaryMetrics'
import Card from '@/components/ui/Card'
import Select from '@/components/ui/Select'

export default function DecisionPanel({ metadata }: { metadata: unknown }) {
  const { t } = useLocale()
  const [distribution, setDistribution] = useState('scoring')
  const decision = decisionObservation(metadata)
  if (!decision) return null
  const rawAvailable = decision.rows.every((row) => row.raw !== null)

  return (
    <Card title={t('reportDetail.decisionProbabilities')}>
      <div className="flex flex-col gap-4">
        <p className="text-sm break-words">
          {decision.kind.toUpperCase()} · {t(decision.kind === 'score' ? 'reportDetail.expectedScore' : 'reportDetail.decisionPrediction')}: {decision.expected ?? decision.predicted}
          {' · '}{t('reportDetail.decisionGold')}: {decision.gold}
        </p>
        {decision.confidence !== null && (
          <p className="text-xs text-[var(--text-muted)]">
            {t('reportDetail.providerConfidence')}: {formatMetric(decision.confidence, null).primary} · {t('reportDetail.confidenceHint')}
          </p>
        )}
        {decision.kind === 'noul' && decision.threshold !== null && (
          <p className="text-xs text-[var(--text-muted)]">{t('reportDetail.noulThreshold')}: {decision.threshold}</p>
        )}
        <Select
          label={t('reportDetail.distribution')}
          value={distribution}
          onChange={setDistribution}
          options={[
            { value: 'scoring', label: t('reportDetail.scoringProbabilities') },
            ...(rawAvailable ? [{ value: 'raw', label: t('reportDetail.rawProbabilities') }] : []),
          ]}
        />
        <div className="flex flex-col gap-2">
          {decision.rows.map((row) => {
            const probability = distribution === 'raw' && rawAvailable ? row.raw! : row.probability
            return (
              <div key={row.label} className="grid grid-cols-[minmax(3rem,1fr)_2fr_5rem] items-center gap-3 text-sm">
                <span className="break-words">{row.label}</span>
                <meter className="w-full" min={0} max={1} value={probability} aria-label={row.label} />
                <span className="text-right font-mono tabular-nums">{formatMetric(probability, RATIO_PERCENT_SEMANTICS).primary}</span>
              </div>
            )
          })}
        </div>
      </div>
    </Card>
  )
}
