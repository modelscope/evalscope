import { Plus, X } from 'lucide-react'
import { parseReportRef } from '@/domain/report/reportRef'
import { MAX_COMPARE_SLOTS } from '@/domain/compare/selection'
import Button from '@/components/ui/Button'
import SearchInput from '@/components/ui/SearchInput'
import SelectionCheckbox from '@/components/ui/SelectionCheckbox'
import { MODEL_PALETTE, type Translate } from '@/components/compare/compareSlots'

export default function CompareReportRail({
  activeTab,
  reportNames,
  predictionReportNames,
  displayLabels,
  reportSearch,
  setReportSearch,
  onRemoveReport,
  onTogglePredictionReport,
  addInput,
  setAddInput,
  showAddInput,
  setShowAddInput,
  onAddReport,
  t,
}: {
  activeTab: 'score' | 'prediction'
  reportNames: string[]
  predictionReportNames: string[]
  displayLabels: Record<string, string>
  reportSearch: string
  setReportSearch: (value: string) => void
  onRemoveReport: (name: string) => void
  onTogglePredictionReport: (name: string) => void
  addInput: string
  setAddInput: (value: string) => void
  showAddInput: boolean
  setShowAddInput: (value: boolean) => void
  onAddReport: () => void
  t: Translate
}) {
  const normalizedSearch = reportSearch.trim().toLowerCase()
  const visibleReports = reportNames.filter((name) => {
    if (!normalizedSearch) return true
    return (displayLabels[name] ?? name).toLowerCase().includes(normalizedSearch)
  })
  const isPrediction = activeTab === 'prediction'

  return (
    <aside className="overflow-hidden rounded-[var(--radius)] border border-[var(--border)] bg-[var(--bg-card)] lg:sticky lg:top-20">
      <div className="border-b border-[var(--border)] px-4 py-3">
        <div className="flex items-center justify-between gap-3">
          <h2 className="type-label-xs">{t('compare.selectedReports')}</h2>
          <span className="text-xs font-semibold tabular-nums text-[var(--accent)]">
            {isPrediction
              ? `${predictionReportNames.length} / ${MAX_COMPARE_SLOTS}`
              : t('compare.reportCount', { n: reportNames.length })}
          </span>
        </div>
        <p className="mt-1.5 text-xs leading-5 text-[var(--text-muted)]">
          {isPrediction ? t('compare.predictionSelectionHint') : t('compare.scoreSelectionHint')}
        </p>
      </div>

      <div className="p-3">
        <SearchInput
          value={reportSearch}
          onChange={setReportSearch}
          placeholder={t('compare.searchReports')}
        />
      </div>

      <div className="max-h-[360px] overflow-y-auto border-y border-[var(--border)] lg:max-h-[calc(100vh-360px)] lg:min-h-[280px]">
        {visibleReports.map((name) => {
          const checked = isPrediction ? predictionReportNames.includes(name) : true
          const disabled = isPrediction
            ? !checked && predictionReportNames.length >= MAX_COMPARE_SLOTS
            : reportNames.length <= 2
          const label = displayLabels[name] ?? (parseReportRef(name).modelId || name)
          const reportIndex = reportNames.indexOf(name)
          const slotIndex = isPrediction ? predictionReportNames.indexOf(name) : reportIndex
          const palette = slotIndex >= 0 && slotIndex < MODEL_PALETTE.length
            ? MODEL_PALETTE[slotIndex]
            : null
          return (
            <SelectionCheckbox
              key={name}
              checked={checked}
              disabled={disabled}
              label={`${isPrediction ? t('compare.predictionComparison') : t('compare.scoreComparison')}: ${label}`}
              onClick={() => {
                if (isPrediction) onTogglePredictionReport(name)
                else onRemoveReport(name)
              }}
              className="w-full justify-start gap-2.5 border-b border-[var(--border)] px-3 py-2 text-left last:border-b-0 hover:bg-[var(--bg-card2)]"
            >
              <span className="flex min-w-0 items-start gap-2">
                {palette ? (
                  <span
                    aria-hidden="true"
                    className="mt-1.5 inline-block h-2 w-2 shrink-0 rounded-full"
                    style={{ backgroundColor: palette.dot }}
                  />
                ) : isPrediction ? (
                  <span
                    aria-hidden="true"
                    className="mt-1.5 inline-block h-2 w-2 shrink-0 rounded-full border border-[var(--border-strong)]"
                  />
                ) : (
                  <span className="mt-0.5 min-w-5 shrink-0 rounded-[var(--radius-xs)] bg-[var(--bg-deep)] px-1 py-0.5 text-center text-[10px] tabular-nums text-[var(--text-dim)]">
                    {reportIndex + 1}
                  </span>
                )}
                <span className="min-w-0 break-words text-xs font-medium leading-5 text-[var(--text)]" title={label}>
                  {label}
                </span>
              </span>
            </SelectionCheckbox>
          )
        })}
      </div>

      <div className="p-3">
        {showAddInput ? (
          <div className="flex flex-col gap-2">
            <input
              type="text"
              value={addInput}
              onChange={(event) => setAddInput(event.target.value)}
              onKeyDown={(event) => {
                if (event.key === 'Enter') onAddReport()
                if (event.key === 'Escape') setShowAddInput(false)
              }}
              placeholder={t('compare.reportNamePlaceholder')}
              autoFocus
              className="w-full rounded-[var(--radius-sm)] border border-[var(--border)] bg-[var(--bg-deep)] px-3 py-2 text-sm text-[var(--text)] placeholder:text-[var(--text-dim)] focus:border-[var(--accent)] focus:outline-none"
            />
            <div className="flex items-center gap-2">
              <Button size="sm" onClick={onAddReport}>{t('compare.addModel')}</Button>
              <Button
                size="sm"
                variant="ghost"
                aria-label={t('compare.cancelAdd')}
                onClick={() => setShowAddInput(false)}
              >
                <X size={14} />
              </Button>
            </div>
          </div>
        ) : (
          <Button size="sm" variant="outline" className="w-full" onClick={() => setShowAddInput(true)}>
            <Plus size={14} />
            {t('compare.addModel')}
          </Button>
        )}
      </div>
    </aside>
  )
}
