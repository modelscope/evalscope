import { useState, useCallback } from 'react'
import { Edit3, Cpu, Target, BarChart2, Database, Copy, Check, ChevronDown, ChevronRight } from 'lucide-react'
import type { PredictionRow } from '@/api/types'
import { useLocale } from '@/contexts/LocaleContext'
import MarkdownRenderer from '@/components/common/MarkdownRenderer'
import ScoreBadge from '@/components/common/ScoreBadge'
import { prettyJson } from '@/utils/formatUtils'

interface Props {
  prediction: PredictionRow
  threshold?: number
}

interface SectionCardProps {
  title: string
  icon: React.ReactNode
  borderColor: string
  accentColor: string
  children: React.ReactNode
  copyText?: string
  collapsible?: boolean
  defaultOpen?: boolean
}

function SectionCard({
  title,
  icon,
  borderColor,
  accentColor,
  children,
  copyText,
  collapsible = false,
  defaultOpen = true,
}: SectionCardProps) {
  const { t } = useLocale()
  const [copied, setCopied] = useState(false)
  const [open, setOpen] = useState(defaultOpen)

  const handleCopy = useCallback(async () => {
    if (!copyText) return
    try {
      await navigator.clipboard.writeText(copyText)
      setCopied(true)
      setTimeout(() => setCopied(false), 1800)
    } catch { /* noop */ }
  }, [copyText])

  return (
    <div
      className="glass-card card-hover rounded-2xl overflow-hidden"
      style={{ borderLeftWidth: 3, borderLeftColor: borderColor, borderLeftStyle: 'solid' }}
    >
      {/* Header */}
      <div
        className="flex items-center gap-2 px-4 py-3"
        style={{ borderBottom: open ? '1px solid var(--color-border-subtle)' : 'none' }}
      >
        <span style={{ color: accentColor }}>{icon}</span>
        <h5
          className="text-xs font-semibold uppercase tracking-wider flex-1"
          style={{ color: accentColor }}
        >
          {title}
        </h5>

        {copyText && (
          <button
            onClick={handleCopy}
            title={t('prediction.copyContent')}
            className="flex items-center gap-1 px-2 py-1 rounded-lg text-xs transition-all hover:bg-[var(--color-surface-hover)]"
            style={{ color: 'var(--color-ink-muted)' }}
          >
            {copied ? (
              <>
                <Check size={12} />
                <span>{t('prediction.copySuccess')}</span>
              </>
            ) : (
              <>
                <Copy size={12} />
                <span>{t('prediction.copyContent')}</span>
              </>
            )}
          </button>
        )}

        {collapsible && (
          <button
            onClick={() => setOpen((v) => !v)}
            className="p-1 rounded-lg hover:bg-[var(--color-surface-hover)] transition-colors"
            style={{ color: 'var(--color-ink-muted)' }}
          >
            {open ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
          </button>
        )}
      </div>

      {/* Body */}
      {open && (
        <div className="px-4 py-3 overflow-auto max-h-[400px]">
          {children}
        </div>
      )}
    </div>
  )
}

function formatValue(val: unknown): string {
  if (val === null || val === undefined) return ''
  if (typeof val === 'object') return JSON.stringify(val, null, 2)
  return String(val)
}

/* ─── main export ─────────────────────────────────────────────── */

export default function FlatView({ prediction, threshold = 0.99 }: Props) {
  const { t } = useLocale()

  const showPred =
    prediction.Pred &&
    prediction.Generated &&
    prediction.Pred.trim() !== prediction.Generated.trim()

  const metaStr = formatValue(prediction.Metadata)
  const scoreStr = prettyJson(prediction.Score)
  const isLargeMetadata = metaStr.length > 300

  return (
    <div className="flex flex-col gap-3 stagger-children">
      {/* Input */}
      <SectionCard
        title={t('prediction.input')}
        icon={<Edit3 size={14} />}
        borderColor="#6366f1"
        accentColor="#818cf8"
        copyText={formatValue(prediction.Input)}
      >
        <MarkdownRenderer content={formatValue(prediction.Input)} />
      </SectionCard>

      {/* Generated */}
      <SectionCard
        title={t('prediction.generated')}
        icon={<Cpu size={14} />}
        borderColor="#10b981"
        accentColor="#34d399"
        copyText={formatValue(prediction.Generated)}
      >
        <MarkdownRenderer content={formatValue(prediction.Generated)} />
      </SectionCard>

      {/* Prediction (only if different from Generated) */}
      {showPred && (
        <SectionCard
          title={t('prediction.prediction')}
          icon={<Cpu size={14} />}
          borderColor="#8b5cf6"
          accentColor="#a78bfa"
          copyText={formatValue(prediction.Pred)}
        >
          <MarkdownRenderer content={formatValue(prediction.Pred)} />
        </SectionCard>
      )}

      {/* Expected Answer */}
      <SectionCard
        title={t('prediction.expectedAnswer')}
        icon={<Target size={14} />}
        borderColor="#f59e0b"
        accentColor="#fbbf24"
        copyText={formatValue(prediction.Gold)}
      >
        <MarkdownRenderer content={formatValue(prediction.Gold)} />
      </SectionCard>

      {/* Score */}
      <SectionCard
        title={t('prediction.score')}
        icon={<BarChart2 size={14} />}
        borderColor="#06b6d4"
        accentColor="#22d3ee"
        copyText={scoreStr}
      >
        <div className="flex flex-col gap-3">
          <div className="flex items-center gap-3">
            <span className="text-xs text-[var(--color-ink-muted)]">NScore</span>
            <ScoreBadge score={prediction.NScore} threshold={threshold} />
            <span className="text-xs text-[var(--color-ink-muted)] ml-auto">
              threshold: {threshold}
            </span>
          </div>
          <pre
            className="text-xs font-mono whitespace-pre-wrap break-all p-3 rounded-xl"
            style={{ background: 'var(--color-surface)', color: 'var(--color-ink-muted)' }}
          >
            {scoreStr}
          </pre>
        </div>
      </SectionCard>

      {/* Metadata */}
      {prediction.Metadata !== null && prediction.Metadata !== undefined && metaStr && metaStr !== '{}' && (
        <SectionCard
          title={t('prediction.metadata')}
          icon={<Database size={14} />}
          borderColor="#94a3b8"
          accentColor="#94a3b8"
          copyText={metaStr}
          collapsible={isLargeMetadata}
          defaultOpen={!isLargeMetadata}
        >
          <pre
            className="text-xs font-mono whitespace-pre-wrap break-all"
            style={{ color: 'var(--color-ink-muted)' }}
          >
            {metaStr}
          </pre>
        </SectionCard>
      )}
    </div>
  )
}
