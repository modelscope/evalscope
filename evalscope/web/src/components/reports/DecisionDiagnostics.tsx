import { useState } from 'react'
import type { ReportData } from '@/api/types'
import { useLocale } from '@/contexts/LocaleContext'
import { decisionCurves } from '@/domain/report/decisionDiagnostics'
import { formatMetric } from '@/domain/metric'
import { RATIO_PERCENT_SEMANTICS } from '@/domain/report/primaryMetrics'
import Card from '@/components/ui/Card'
import Select from '@/components/ui/Select'

export default function DecisionDiagnostics({ report }: { report?: ReportData }) {
  const { t } = useLocale()
  const curves = decisionCurves(report)
  const [picked, setPicked] = useState('')
  const curve = curves.find((item) => item.id === picked) ?? curves[0]
  if (!curve) return null
  return (
    <Card title={t('reportDetail.riskCoverage')}>
      <div className="flex flex-col gap-4">
        <Select
          label={t('reportDetail.decisionGroup')}
          value={curve.id}
          onChange={setPicked}
          options={curves.map((item) => ({
            value: item.id,
            label: `${item.subset} · ${item.kind}${item.levels ? ` · ${item.levels}` : ''} · ${t(`reportDetail.${item.basis === 'provider_confidence' ? 'providerConfidence' : 'topProbability'}`)}`,
          }))}
        />
        <p className="text-sm">AURC ↓: {formatMetric(curve.aurc, curve.semantics).primary} · {t('reportDetail.samples')}: {curve.num}</p>
        <p className="text-xs text-[var(--text-muted)]">
          {t(curve.risk === 'normalized_mae' ? 'reportDetail.scoreRiskHint' : 'reportDetail.classRiskHint')}
          {' '}{t('reportDetail.tiedCoverageHint')}
        </p>
        <CoveragePoint key={curve.id} points={curve.points} />
        {curve.unavailable && (
          <p className="text-sm text-[var(--text-muted)]">
            {t('reportDetail.unavailableMetrics')}: {Object.entries(curve.unavailable).map(([name, reason]) => `${name}: ${String(reason)}`).join('; ')}
          </p>
        )}
      </div>
    </Card>
  )
}

function CoveragePoint({ points }: { points: { coverage: number; risk: number }[] }) {
  const { t } = useLocale()
  const [index, setIndex] = useState(points.length - 1)
  const point = points[Math.min(index, points.length - 1)]
  return (
    <div className="flex flex-col gap-3">
      <label className="text-sm" htmlFor="decision-coverage">{t('reportDetail.coveragePoint')}</label>
      <input
        id="decision-coverage" type="range" min={0} max={points.length - 1} step={1}
        value={index} onChange={(event) => setIndex(Number(event.target.value))}
        aria-valuetext={`${t('reportDetail.coverage')}: ${formatMetric(point.coverage, RATIO_PERCENT_SEMANTICS).primary}, ${t('reportDetail.risk')}: ${formatMetric(point.risk, RATIO_PERCENT_SEMANTICS).primary}`}
        className="w-full accent-[var(--accent)]"
      />
      <output htmlFor="decision-coverage" aria-live="polite" className="text-sm font-mono tabular-nums">
        {t('reportDetail.coverage')}: {formatMetric(point.coverage, RATIO_PERCENT_SEMANTICS).primary} · {t('reportDetail.risk')}: {formatMetric(point.risk, RATIO_PERCENT_SEMANTICS).primary}
      </output>
    </div>
  )
}
