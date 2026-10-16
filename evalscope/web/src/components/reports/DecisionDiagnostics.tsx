import { useId, useState } from 'react'
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
        <dl className="flex flex-wrap gap-x-8 gap-y-3">
          <div><dt className="text-xs text-[var(--text-muted)]">AURC ↓</dt><dd className="mt-1 text-lg font-semibold tabular-nums">{formatMetric(curve.aurc, curve.semantics).primary}</dd></div>
          <div><dt className="text-xs text-[var(--text-muted)]">{t('reportDetail.decisionSamples')}</dt><dd className="mt-1 text-lg font-semibold tabular-nums">{curve.num}</dd></div>
        </dl>
        <CoveragePoint key={curve.id} points={curve.points} />
        <p className="text-xs leading-relaxed text-[var(--text-muted)]">
          {t(curve.risk === 'normalized_mae' ? 'reportDetail.scoreRiskHint' : 'reportDetail.classRiskHint')}
          {' '}{t('reportDetail.tiedCoverageHint')}
        </p>
        {curve.unavailable && (
          <details className="border-t border-[var(--border)] pt-3 text-xs text-[var(--text-muted)]">
            <summary className="cursor-pointer">{t('reportDetail.unavailableMetrics')}</summary>
            <ul className="mt-2 space-y-1 break-words">
              {Object.entries(curve.unavailable).map(([name, reason]) => <li key={name}>{name}: {String(reason)}</li>)}
            </ul>
          </details>
        )}
      </div>
    </Card>
  )
}

function CoveragePoint({ points }: { points: { coverage: number; risk: number }[] }) {
  const { t } = useLocale()
  const id = useId()
  const [index, setIndex] = useState(points.length - 1)
  const point = points[Math.min(index, points.length - 1)]
  const coverage = formatMetric(point.coverage, RATIO_PERCENT_SEMANTICS).primary
  const risk = formatMetric(point.risk, RATIO_PERCENT_SEMANTICS).primary
  const summary = `${t('reportDetail.coverage')}: ${coverage} · ${t('reportDetail.risk')}: ${risk}`
  return (
    <div className="flex flex-col gap-3 rounded-[var(--radius-sm)] bg-[var(--bg-card2)] p-4">
      <output htmlFor={id} aria-live="polite" aria-label={summary} className="grid grid-cols-2 gap-4">
        <span className="min-w-0"><span className="block text-xs text-[var(--text-muted)]">{t('reportDetail.coverage')}</span><span className="mt-1 block text-2xl font-semibold tabular-nums text-[var(--accent)]">{coverage}</span></span>
        <span className="min-w-0 border-l border-[var(--border)] pl-4"><span className="block text-xs text-[var(--text-muted)]">{t('reportDetail.risk')} ↓</span><span className="mt-1 block text-2xl font-semibold tabular-nums">{risk}</span></span>
      </output>
      <label className="text-xs text-[var(--text-muted)]" htmlFor={id}>{t('reportDetail.coveragePoint')}</label>
      <input
        id={id} type="range" min={0} max={points.length - 1} step={1}
        value={index} onChange={(event) => setIndex(Number(event.target.value))}
        aria-valuetext={summary}
        disabled={points.length === 1}
        className="decision-coverage-slider h-5 w-full cursor-pointer disabled:cursor-default"
      />
    </div>
  )
}
