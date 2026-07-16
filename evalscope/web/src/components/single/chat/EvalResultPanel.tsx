import React, { useState } from 'react'
import {
  ChevronDown,
  ChevronRight,
  ClipboardCheck,
  Scissors,
  Target,
  Gauge,
  FileJson,
  Database,
} from 'lucide-react'
import { useLocale } from '@/contexts/LocaleContext'
import MarkdownRenderer from '@/components/common/MarkdownRenderer'
import ScoreBadge from '@/components/ui/ScoreBadge'
import JsonViewer from '@/components/common/JsonViewer'

/* ─── CollapsibleJson ──────────────────────────────────────── */

export function CollapsibleJson({
  label,
  value,
  maxHeight = 200,
  defaultOpen = false,
  icon,
}: {
  label: string
  value: unknown
  maxHeight?: number
  defaultOpen?: boolean
  icon?: React.ReactNode
}) {
  const [open, setOpen] = useState(defaultOpen)
  const isEmpty =
    value == null ||
    (typeof value === 'object' && Object.keys(value as object).length === 0) ||
    value === '{}'
  if (isEmpty) return null
  return (
    <div className="mt-[0.4rem]">
      <button
        onClick={() => setOpen(v => !v)}
        aria-expanded={open}
        className="inline-flex min-h-8 items-center gap-1.5 bg-transparent px-0 py-1 text-xs font-medium text-[var(--text-muted)] hover:text-[var(--text)]"
      >
        {open ? <ChevronDown size={11} /> : <ChevronRight size={11} />}
        {icon && <span className="inline-flex items-center">{icon}</span>}
        <span className="uppercase tracking-wide font-semibold">{label}</span>
      </button>
      {open && (
        <div className="mt-[0.3rem] rounded-[0.4rem] overflow-hidden border border-[var(--border)]">
          <JsonViewer value={value} maxHeight={maxHeight} />
        </div>
      )}
    </div>
  )
}

/* ─── EvalResultPanel ──────────────────────────────────────── */

export interface EvalResultPanelProps {
  pred: string
  gold: string
  nScore: number
  score: Record<string, unknown>
  metadata: unknown
  threshold: number
  showPred: boolean
}

const eyebrowBase = 'type-label-xs mb-2 flex items-center gap-1.5'

export function EvalResultPanel({
  pred,
  gold,
  nScore,
  score,
  metadata,
  threshold,
  showPred,
}: EvalResultPanelProps) {
  const { t } = useLocale()
  const isSameAsGenerated = pred === '*Same as Generated*' || pred === ''
  const metaStr = (() => {
    if (metadata == null) return ''
    if (typeof metadata === 'object') return JSON.stringify(metadata)
    return String(metadata)
  })()
  const hasMetadata = metaStr && metaStr !== '{}' && metaStr !== 'null'

  return (
    <section
      className="overflow-hidden rounded-xl border border-[var(--border-md)] bg-[var(--bg-card)] shadow-[var(--shadow-sm)]"
      style={{ animation: 'fadeInUp 300ms ease-out 160ms both' }}
    >
      <div className="flex items-center gap-2 border-b border-[var(--border)] bg-[var(--bg-card2)] px-4 py-3">
        <ClipboardCheck size={15} className="text-[var(--text-muted)]" />
        <span className="type-label-xs">
          {t('prediction.evalResult')}
        </span>
      </div>

      <div
        className={`grid grid-cols-1 gap-0 px-4 py-4 ${
          showPred
            ? 'md:grid-cols-[minmax(0,1fr)_minmax(0,1fr)_minmax(9rem,auto)]'
            : 'md:grid-cols-[minmax(0,1fr)_minmax(9rem,auto)]'
        }`}
      >
        {showPred && (
          <div className="pb-4 md:border-r md:border-[var(--border)] md:pb-0 md:pr-4">
            <div className={eyebrowBase}>
              <Scissors size={13} />
              {t('prediction.extractedAnswer')}
            </div>
            {isSameAsGenerated ? (
              <span className="text-xs text-[var(--text-muted)] opacity-50 italic">
                = Generated
              </span>
            ) : (
              <div className="type-body-sm text-[var(--text)]">
                <MarkdownRenderer content={pred} />
              </div>
            )}
          </div>
        )}

        <div className={`${showPred ? 'border-t border-[var(--border)] py-4 md:border-t-0 md:px-4 md:py-0' : 'pb-4 md:border-r md:border-[var(--border)] md:pb-0 md:pr-4'} md:border-r`}>
          <div className={eyebrowBase}>
            <Target size={13} />
            {t('prediction.expectedAnswer')}
          </div>
          <div className="type-body-sm text-[var(--text)]">
            <MarkdownRenderer content={gold} />
          </div>
        </div>

        <div className="border-t border-[var(--border)] pt-4 md:border-t-0 md:pl-4 md:pt-0">
          <div className={eyebrowBase}>
            <Gauge size={13} />
            {t('prediction.score')}
          </div>
          <div className="flex flex-col gap-1">
            <ScoreBadge score={nScore} threshold={threshold} className="self-start" />
            <span className="type-body-xs text-[var(--text-muted)]">
              {nScore >= threshold ? t('prediction.aboveFilter') : t('prediction.belowFilter')} · {threshold}
            </span>
          </div>
        </div>
      </div>

      {(Object.keys(score).length > 0 || hasMetadata) && (
        <div className="flex flex-col gap-1 border-t border-[var(--border)] px-4 py-3">
          {Object.keys(score).length > 0 && (
            <CollapsibleJson
              label={t('prediction.scoreJson')}
              value={score}
              maxHeight={200}
              icon={<FileJson size={11} />}
            />
          )}
          {hasMetadata && (
            <CollapsibleJson
              label={t('prediction.metadata')}
              value={metadata}
              maxHeight={250}
              icon={<Database size={11} />}
            />
          )}
        </div>
      )}
    </section>
  )
}
