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
    <div style={{ marginTop: '0.4rem' }}>
      <button
        onClick={() => setOpen(v => !v)}
        style={{
          display: 'inline-flex',
          alignItems: 'center',
          gap: '0.3rem',
          fontSize: '0.7rem',
          color: 'var(--text-muted)',
          opacity: 0.7,
          background: 'none',
          border: 'none',
          cursor: 'pointer',
          padding: '0.1rem 0',
        }}
      >
        {open ? <ChevronDown size={11} /> : <ChevronRight size={11} />}
        {icon && <span style={{ display: 'inline-flex', alignItems: 'center' }}>{icon}</span>}
        <span className="uppercase tracking-wide font-semibold">{label}</span>
      </button>
      {open && (
        <div
          style={{
            marginTop: '0.3rem',
            borderRadius: '0.4rem',
            overflow: 'hidden',
            border: '1px solid var(--border)',
          }}
        >
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
      style={{
        borderRadius: '0.75rem',
        border: '1px solid var(--border-md)',
        background: 'var(--bg-card2)',
        overflow: 'hidden',
        boxShadow: 'var(--shadow-sm)',
        animation: 'fadeInUp 300ms ease-out 160ms both',
      }}
    >
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: '0.4rem',
          padding: '0.5rem 1rem',
          borderBottom: '1px solid var(--border-md)',
          background: 'var(--bg-deep)',
        }}
      >
        <ClipboardCheck size={13} style={{ color: 'var(--text-muted)', opacity: 0.6 }} />
        <span
          style={{
            fontSize: '0.65rem',
            fontWeight: 700,
            textTransform: 'uppercase',
            letterSpacing: '0.1em',
            color: 'var(--text-muted)',
            opacity: 0.7,
          }}
        >
          {t('prediction.evalResult')}
        </span>
      </div>

      <div
        style={{
          display: 'grid',
          gridTemplateColumns: showPred ? 'minmax(80px,auto) 1fr minmax(100px,auto)' : '1fr minmax(100px,auto)',
          gap: 0,
          padding: '0.75rem 1rem',
        }}
      >
        {showPred && (
          <div style={{ paddingRight: '1rem', borderRight: '1px solid var(--border-md)' }}>
            <div
              style={{
                fontSize: '0.65rem',
                fontWeight: 700,
                textTransform: 'uppercase',
                letterSpacing: '0.08em',
                color: 'var(--purple)',
                opacity: 0.8,
                marginBottom: '0.35rem',
                display: 'flex',
                alignItems: 'center',
                gap: '0.25rem',
              }}
            >
              <Scissors size={11} />
              {t('prediction.extractedAnswer')}
            </div>
            {isSameAsGenerated ? (
              <span
                style={{
                  fontSize: '0.75rem',
                  color: 'var(--text-muted)',
                  opacity: 0.5,
                  fontStyle: 'italic',
                }}
              >
                = Generated
              </span>
            ) : (
              <div style={{ fontSize: '0.875rem' }}>
                <MarkdownRenderer content={pred} />
              </div>
            )}
          </div>
        )}

        <div
          style={{
            padding: showPred ? '0 1rem' : '0 1rem 0 0',
            borderRight: '1px solid var(--border-md)',
          }}
        >
          <div
            style={{
              fontSize: '0.65rem',
              fontWeight: 700,
              textTransform: 'uppercase',
              letterSpacing: '0.08em',
              color: 'var(--yellow)',
              opacity: 0.9,
              marginBottom: '0.35rem',
              display: 'flex',
              alignItems: 'center',
              gap: '0.25rem',
            }}
          >
            <Target size={11} />
            {t('prediction.expectedAnswer')}
          </div>
          <div style={{ fontSize: '0.875rem' }}>
            <MarkdownRenderer content={gold} />
          </div>
        </div>

        <div style={{ paddingLeft: '1rem' }}>
          <div
            style={{
              fontSize: '0.65rem',
              fontWeight: 700,
              textTransform: 'uppercase',
              letterSpacing: '0.08em',
              color: 'var(--cyan)',
              opacity: 0.9,
              marginBottom: '0.35rem',
              display: 'flex',
              alignItems: 'center',
              gap: '0.25rem',
            }}
          >
            <Gauge size={11} />
            {t('prediction.score')}
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.25rem' }}>
            <ScoreBadge score={nScore} threshold={threshold} />
            <span style={{ fontSize: '0.65rem', color: 'var(--text-muted)', opacity: 0.5 }}>
              thr: {threshold}
            </span>
          </div>
        </div>
      </div>

      {(Object.keys(score).length > 0 || hasMetadata) && (
        <div
          style={{
            padding: '0 1rem 0.75rem',
            display: 'flex',
            flexDirection: 'column',
            gap: '0.25rem',
          }}
        >
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
