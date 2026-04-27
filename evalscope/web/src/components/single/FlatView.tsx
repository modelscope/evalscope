import { useState, useCallback } from 'react'
import { Edit3, Cpu, Target, Copy, Check, ChevronDown, ChevronRight, ClipboardCheck } from 'lucide-react'
import type { PredictionRow } from '@/api/types'
import { useLocale } from '@/contexts/LocaleContext'
import MarkdownRenderer from '@/components/common/MarkdownRenderer'
import ScoreBadge from '@/components/common/ScoreBadge'
import { prettyJson } from '@/utils/formatUtils'
import JsonViewer from '@/components/common/JsonViewer'
import PerfChip from './PerfChip'

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
    } catch {
      const el = document.createElement('textarea')
      el.value = copyText
      document.body.appendChild(el)
      el.select()
      document.execCommand('copy')
      document.body.removeChild(el)
      setCopied(true)
      setTimeout(() => setCopied(false), 1800)
    }
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

/* ─── Combined Score + Metadata Panel ───────────────────────────── */

type EvalTab = 'score' | 'metadata'

interface EvalPanelProps {
  score: Record<string, unknown>
  metadata: unknown
  nScore: number
  threshold: number
  scoreStr: string
  metaStr: string
  hasMetadata: boolean
}

function EvalPanel({ score, metadata, nScore, threshold, scoreStr, metaStr, hasMetadata }: EvalPanelProps) {
  const { t } = useLocale()
  const [activeTab, setActiveTab] = useState<EvalTab>('score')
  const [copied, setCopied] = useState(false)

  const copyText = activeTab === 'score' ? scoreStr : metaStr

  const handleCopy = useCallback(async () => {
    try {
      await navigator.clipboard.writeText(copyText)
      setCopied(true)
      setTimeout(() => setCopied(false), 1800)
    } catch {
      const el = document.createElement('textarea')
      el.value = copyText
      document.body.appendChild(el)
      el.select()
      document.execCommand('copy')
      document.body.removeChild(el)
      setCopied(true)
      setTimeout(() => setCopied(false), 1800)
    }
  }, [copyText])

  const tabs: { key: EvalTab; label: string; show: boolean }[] = [
    { key: 'score', label: t('prediction.score'), show: true },
    { key: 'metadata', label: t('prediction.metadata'), show: hasMetadata },
  ]

  return (
    <div
      className="glass-card card-hover rounded-2xl overflow-hidden"
      style={{ borderLeftWidth: 3, borderLeftColor: '#22d3ee', borderLeftStyle: 'solid' }}
    >
      {/* Header */}
      <div
        className="flex items-center gap-2 px-4 py-3"
        style={{ borderBottom: '1px solid var(--color-border-subtle)' }}
      >
        <ClipboardCheck size={14} style={{ color: '#22d3ee' }} />
        <h5
          className="text-xs font-semibold uppercase tracking-wider"
          style={{ color: '#22d3ee' }}
        >
          {t('prediction.evalResult')}
        </h5>

        {/* Tab switcher */}
        <div
          style={{
            display: 'inline-flex',
            marginLeft: '0.5rem',
            borderRadius: '0.5rem',
            border: '1px solid var(--color-border)',
            overflow: 'hidden',
            gap: 0,
          }}
        >
          {tabs.filter(tab => tab.show).map((tab) => (
            <button
              key={tab.key}
              onClick={() => setActiveTab(tab.key)}
              style={{
                padding: '0.2rem 0.6rem',
                fontSize: '0.7rem',
                fontWeight: 500,
                background: activeTab === tab.key ? 'rgba(34,211,238,0.15)' : 'transparent',
                color: activeTab === tab.key ? '#22d3ee' : 'var(--color-ink-muted)',
                border: 'none',
                cursor: 'pointer',
                transition: 'all 0.15s',
              }}
            >
              {tab.label}
            </button>
          ))}
        </div>

        {/* Copy button */}
        <button
          onClick={handleCopy}
          className="flex items-center gap-1 px-2 py-1 rounded-lg text-xs transition-all hover:bg-[var(--color-surface-hover)] ml-auto"
          style={{ color: 'var(--color-ink-muted)' }}
        >
          {copied ? (
            <><Check size={12} /><span>{t('prediction.copySuccess')}</span></>
          ) : (
            <><Copy size={12} /><span>{t('prediction.copyContent')}</span></>
          )}
        </button>
      </div>

      {/* Body */}
      <div className="px-4 py-3">
        {activeTab === 'score' && (
          <div className="flex flex-col gap-3">
            {/* NScore summary row */}
            <div className="flex items-center gap-3">
              <span className="text-xs text-[var(--color-ink-muted)]">NScore</span>
              <ScoreBadge score={nScore} threshold={threshold} />
              <span className="text-xs text-[var(--color-ink-muted)] ml-auto opacity-60">
                thr: {threshold}
              </span>
            </div>
            {/* Full score JSON */}
            {Object.keys(score).length > 0 && (
              <JsonViewer value={score} maxHeight={280} />
            )}
          </div>
        )}
        {activeTab === 'metadata' && hasMetadata && (
          <JsonViewer value={metadata} maxHeight={350} />
        )}
      </div>
    </div>
  )
}

/* ─── main export ─────────────────────────────────────────────── */

export default function FlatView({ prediction, threshold = 0.99 }: Props) {
  const { t } = useLocale()

  const showPred =
    prediction.Pred &&
    prediction.Pred !== '*Same as Generated*' &&
    prediction.Generated &&
    prediction.Pred.trim() !== prediction.Generated.trim()

  const metaStr = formatValue(prediction.Metadata)
  const scoreStr = prettyJson(prediction.Score)
  const hasMetadata = !!(
    prediction.Metadata !== null &&
    prediction.Metadata !== undefined &&
    metaStr &&
    metaStr !== '{}'
  )

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
        {/* Per-sample perf chip */}
        {prediction.PerfMetrics && (
          <PerfChip metrics={prediction.PerfMetrics} variant="green" />
        )}
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

      {/* Combined Score + Metadata panel */}
      <EvalPanel
        score={prediction.Score}
        metadata={prediction.Metadata}
        nScore={prediction.NScore}
        threshold={threshold}
        scoreStr={scoreStr}
        metaStr={metaStr}
        hasMetadata={hasMetadata}
      />
    </div>
  )
}
