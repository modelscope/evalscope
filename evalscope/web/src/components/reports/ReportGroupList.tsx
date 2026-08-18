import { ChevronRight } from 'lucide-react'
import { cn } from '@/lib/utils'
import { useLocale } from '@/contexts/LocaleContext'
import { formatTimestamp } from '@/utils/formatUtils'
import SelectionCheckbox from '@/components/ui/SelectionCheckbox'
import Button from '@/components/ui/Button'
import ReportsTable from '@/components/reports/ReportsTable'
import ReportCard from '@/components/reports/ReportCard'
import type { ReportGroup } from '@/api/types'
import { formatReportRef, reportRefFromSummary } from '@/domain/report/reportRef'

interface GroupHeaderProps {
  group: ReportGroup
  expanded: boolean
  onToggleExpand: () => void
  groupSelected: boolean
  onToggleSelectGroup: () => void
  onCompareGroup: () => void
}

/**
 * One model's rollup header: expand/collapse, select-all-children, and a
 * shortcut into Compare for every report under this model. The row itself
 * carries no score - each child keeps its own honestly-attributed score,
 * visible once expanded.
 */
function GroupHeader({
  group,
  expanded,
  onToggleExpand,
  groupSelected,
  onToggleSelectGroup,
  onCompareGroup,
}: GroupHeaderProps) {
  const { t } = useLocale()

  return (
    <div
      className={cn(
        'flex items-center gap-3 px-4 py-3 cursor-pointer transition-colors',
        expanded ? 'bg-[var(--bg-card2)]' : 'hover:bg-[var(--bg-card2)]',
      )}
      onClick={onToggleExpand}
    >
      <SelectionCheckbox
        checked={groupSelected}
        label={`${t('reports.selectAll')}: ${group.model_name}`}
        onClick={(e) => {
          e.stopPropagation()
          onToggleSelectGroup()
        }}
      />

      <ChevronRight
        size={16}
        className={cn('shrink-0 text-[var(--text-dim)] transition-transform', expanded && 'rotate-90')}
      />

      <div className="flex-1 min-w-0 flex flex-wrap items-baseline gap-x-3 gap-y-1">
        <span className="font-bold text-base text-[var(--text)] break-words min-w-0">{group.model_name}</span>
        <span className="text-xs text-[var(--text-muted)]">{t('reports.group.reportCount', { n: group.report_count })}</span>
        <span className="text-xs text-[var(--text-muted)]">{t('reports.group.datasetCount', { n: group.dataset_count })}</span>
        <span className="text-xs text-[var(--text-muted)]">{t('reports.samples')}: {group.num_samples}</span>
      </div>

      <span className="hidden sm:block text-xs text-[var(--text-muted)] font-mono whitespace-nowrap">
        {formatTimestamp(group.timestamp) || '—'}
      </span>

      {group.report_count >= 2 && (
        <Button
          variant="ghost"
          size="sm"
          onClick={(e) => {
            e.stopPropagation()
            onCompareGroup()
          }}
        >
          {t('reports.compareAll', { n: group.report_count })}
        </Button>
      )}
    </div>
  )
}

interface ReportGroupListProps {
  groups: ReportGroup[]
  expandedModels: Set<string>
  onToggleExpand: (modelName: string) => void
  selected: string[]
  onToggleSelect: (ref: string) => void
  onSelectGroup: (refs: string[], select: boolean) => void
  onRowClick: (ref: string) => void
  onCompareGroup: (refs: string[]) => void
  variant: 'table' | 'cards'
}

/** Desktop (table) or narrow (cards) rendering of `group_by=model` rows, expandable per model. */
export default function ReportGroupList({
  groups,
  expandedModels,
  onToggleExpand,
  selected,
  onToggleSelect,
  onSelectGroup,
  onRowClick,
  onCompareGroup,
  variant,
}: ReportGroupListProps) {
  const selectedSet = new Set(selected)

  return (
    <div className="flex flex-col gap-2">
      {groups.map((group) => {
        const expanded = expandedModels.has(group.model_name)
        const groupSelected = group.refs.length > 0 && group.refs.every((ref) => selectedSet.has(ref))
        return (
          <div
            key={group.model_name}
            className="rounded-[var(--radius)] border border-[var(--border)] bg-[var(--bg-card)] overflow-hidden"
          >
            <GroupHeader
              group={group}
              expanded={expanded}
              onToggleExpand={() => onToggleExpand(group.model_name)}
              groupSelected={groupSelected}
              onToggleSelectGroup={() => onSelectGroup(group.refs, !groupSelected)}
              onCompareGroup={() => onCompareGroup(group.refs)}
            />
            {expanded && (
              <div className="border-t border-[var(--border)] p-2">
                {variant === 'table' ? (
                  <ReportsTable
                    reports={group.children}
                    selected={selected}
                    allSelected={groupSelected}
                    onToggleSelectAll={() => onSelectGroup(group.refs, !groupSelected)}
                    onToggleSelect={onToggleSelect}
                    onRowClick={onRowClick}
                  />
                ) : (
                  <div className="flex flex-col gap-2">
                    {group.children.map((child) => {
                      const ref = formatReportRef(reportRefFromSummary(child))
                      return (
                        <ReportCard
                          key={ref}
                          report={child}
                          selected={selectedSet.has(ref)}
                          onSelect={onToggleSelect}
                          onClick={onRowClick}
                        />
                      )
                    })}
                  </div>
                )}
              </div>
            )}
          </div>
        )
      })}
    </div>
  )
}
