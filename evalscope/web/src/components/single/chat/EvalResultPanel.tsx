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
        className="inline-flex items-center gap-[0.3rem] text-[0.7rem] text-[var(--text-muted)] opacity-70 bg-transparent border-0 cursor-pointer py-[0.1rem] px-0"
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

const eyebrowBase = 'text-[0.65rem] font-bold uppercase tracking-[0.08em] mb-[0.35rem] flex items-center gap-[0.25rem]'

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
    <div
      className="rounded-xl border border-[var(--border-md)] bg-[var(--bg-card2)] overflow-hidden shadow-[var(--shadow-sm)]"
      style={{ animation: 'fadeInUp 300ms ease-out 160ms both' }}
    >
      <div className="flex items-center gap-[0.4rem] px-4 py-2 border-b border-[var(--border-md)] bg-[var(--bg-deep)]">
        <ClipboardCheck size={13} className="text-[var(--text-muted)] opacity-60" />
        <span className="text-[0.65rem] font-bold uppercase tracking-[0.1em] text-[var(--text-muted)] opacity-70">
          {t('prediction.evalResult')}
        </span>
      </div>

      <div
        className="grid gap-0 px-4 py-3"
        style={{
          gridTemplateColumns: showPred ? 'minmax(80px,auto) 1fr minmax(100px,auto)' : '1fr minmax(100px,auto)',
        }}
      >
        {showPred && (
          <div className="pr-4 border-r border-[var(--border-md)]">
            <div className={`${eyebrowBase} text-[var(--purple)] opacity-80`}>
              <Scissors size={11} />
              {t('prediction.extractedAnswer')}
            </div>
            {isSameAsGenerated ? (
              <span className="text-xs text-[var(--text-muted)] opacity-50 italic">
                = Generated
              </span>
            ) : (
              <div className="text-sm">
                <MarkdownRenderer content={pred} />
              </div>
            )}
          </div>
        )}

        <div className={`${showPred ? 'px-4' : 'pr-4'} border-r border-[var(--border-md)]`}>
          <div className={`${eyebrowBase} text-[var(--yellow)] opacity-90`}>
            <Target size={11} />
            {t('prediction.expectedAnswer')}
          </div>
          <div className="text-sm">
            <MarkdownRenderer content={gold} />
          </div>
        </div>

        <div className="pl-4">
          <div className={`${eyebrowBase} text-[var(--cyan)] opacity-90`}>
            <Gauge size={11} />
            {t('prediction.score')}
          </div>
          <div className="flex flex-col gap-1">
            <ScoreBadge score={nScore} threshold={threshold} />
            <span className="text-[0.65rem] text-[var(--text-muted)] opacity-50">
              thr: {threshold}
            </span>
          </div>
        </div>
      </div>

      {(Object.keys(score).length > 0 || hasMetadata) && (
        <div className="flex flex-col gap-1 px-4 pb-3">
          {Object.keys(score).length > 0 && (
            <CollapsibleJson
              label={t('prediction.scoreJson')}
              value={score}
              maxHeight={200}
              defaultOpen
              icon={<FileJson size={11} />}
            />
          )}
          {hasMetadata && (
            <CollapsibleJson
              label={t('prediction.metadata')}
              value={metadata}
              maxHeight={250}
              defaultOpen
              icon={<Database size={11} />}
            />
          )}
        </div>
      )}
    </div>
  )
}
