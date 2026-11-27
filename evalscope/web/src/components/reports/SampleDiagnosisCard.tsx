import type { SampleDiagnosis } from '@/api/types'
import { useLocale } from '@/contexts/LocaleContext'

export default function SampleDiagnosisCard({ diagnosis, showFindings = true }: { diagnosis: SampleDiagnosis; showFindings?: boolean }) {
  const { t } = useLocale()
  const hasModelAnalysis = diagnosis.llm_status !== 'not_selected'
  if (!showFindings && !hasModelAnalysis) return null
  return (
    <details
      data-model-proposal={diagnosis.llm_status === 'ok' || undefined}
      className="rounded-[var(--radius)] border border-[var(--border)] bg-[var(--bg-card)]"
    >
      <summary className="cursor-pointer p-4 text-sm font-medium">
        {t(hasModelAnalysis ? 'diagnosis.modelAnalysis' : 'diagnosis.relatedEvidence')}
      </summary>
      <section className="border-t border-[var(--border)] p-4" aria-label={t('diagnosis.sample')}>
        {diagnosis.llm_status && diagnosis.llm_status !== 'not_selected' && (
          <div className="mb-4 space-y-3">
            <p className="text-sm font-medium">{t('diagnosis.modelAnalysis')} · {t(`diagnosis.llmStatus.${diagnosis.llm_status}`)}</p>
            {diagnosis.llm_status === 'ok' && diagnosis.llm && <>
              <p className="text-sm whitespace-pre-wrap">{diagnosis.llm.explanation}</p>
              <p className="text-xs text-[var(--text-muted)]">{t('diagnosis.proposalNotice')}</p>
              {diagnosis.llm.gold_suspect && <p className="text-sm text-[var(--warning-text)]">{t('diagnosis.goldSuspect')}</p>}
              {diagnosis.llm.evidence.map((reference, index) => {
                const source = diagnosis.sources?.find(s => s.id === reference.source_id)
                if (!source || reference.start == null || reference.end == null) return null
                const chars = Array.from(source.text)
                return <details key={index} className="rounded border border-[var(--border)] p-3">
                  <summary className="cursor-pointer text-xs">{t('diagnosis.evidenceNumber', { number: index + 1 })}
                    {source.step != null ? ` · ${t('trace.step')} ${source.step}` : ''}</summary>
                  <p className="mt-2 text-sm whitespace-pre-wrap break-words">
                    {chars.slice(Math.max(0, reference.start - 80), reference.start).join('')}
                    <mark className="bg-[var(--warning-bg)] text-[var(--text)]">{reference.quote}</mark>
                    {chars.slice(reference.end, reference.end + 80).join('')}
                  </p>
                </details>
              })}
            </>}
          </div>
        )}
        {diagnosis.stop_reason && (
          <p className="text-sm">
            {t('diagnosis.stop')}:{' '}
            {t(`diagnosis.stopReason.${diagnosis.stop_reason}`) ===
            `diagnosis.stopReason.${diagnosis.stop_reason}`
              ? t('diagnosis.stopReason.other')
              : t(`diagnosis.stopReason.${diagnosis.stop_reason}`)}
          </p>
        )}
        {diagnosis.context_omitted && <p className="my-2 text-xs text-[var(--text-muted)]">{t(diagnosis.sources?.some(source => source.id === 'omitted-content') ? 'diagnosis.unsupportedContent' : 'diagnosis.contextOmitted')}</p>}
        {showFindings && <ul className="space-y-1 text-sm">
          {diagnosis.findings.map((finding, i) => (
            <li key={i}>
              <span className="font-medium">{t(`diagnosis.finding.${finding.category}`)}</span>
              <span className="ml-2 text-xs text-[var(--text-muted)]">
                {t('diagnosis.confidenceLabel')}: {t(`diagnosis.confidence.${finding.confidence}`)}
              </span>
              {finding.evidence?.map((e, j) => (
                <blockquote
                  key={j}
                  className="my-1 border-l-2 border-[var(--border)] pl-2 whitespace-pre-wrap break-words text-[var(--text-muted)]"
                >
                  {e.excerpt}
                </blockquote>
              ))}
            </li>
          ))}
        </ul>}
      </section>
    </details>
  )
}
