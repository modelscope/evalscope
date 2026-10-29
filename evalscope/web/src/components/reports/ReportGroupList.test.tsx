import { cleanup, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { LocaleProvider } from '@/contexts/LocaleContext'
import type { MetricSemantics } from '@/domain/metric'
import type { ReportGroup, ReportSummary } from '@/api/types'
import ReportGroupList from './ReportGroupList'

afterEach(cleanup)

const ACCURACY: MetricSemantics = {
  semantic_id: 'quality.accuracy.ratio',
  metric_name: 'Accuracy',
  kind: 'quality',
  direction: 'higher_is_better',
  value_range: { min: 0, max: 1 },
  display_kind: 'percent',
  display_multiplier: 100,
  display_unit: '%',
  display_precision: 1,
}

function reportOn(runId: string, datasetName: string): ReportSummary {
  return {
    run_id: runId,
    model_id: 'gemma-3-27b-it',
    model_name: 'gemma-3-27b-it',
    dataset_name: datasetName,
    num_samples: 100,
    timestamp: '2026-08-17T22:09:00',
    primary_metrics: [
      { dataset_name: datasetName, identity: { name: 'accuracy', aggregation: 'mean', dimensions: {} }, score: 0.6, semantics: ACCURACY },
    ],
  }
}

function groupOf(children: ReportSummary[]): ReportGroup {
  return {
    model_name: 'gemma-3-27b-it',
    dataset_name: children.map((c) => c.dataset_name).join(', '),
    timestamp: children[0].timestamp,
    report_count: children.length,
    dataset_count: new Set(children.map((c) => c.dataset_name)).size,
    num_samples: children.reduce((sum, c) => sum + c.num_samples, 0),
    refs: children.map((c) => `${c.run_id}/${c.model_id}`),
    children,
  }
}

function renderGroups(groups: ReportGroup[]) {
  return render(
    <LocaleProvider>
      <ReportGroupList
        groups={groups}
        expandedModels={new Set()}
        onToggleExpand={vi.fn()}
        selected={[]}
        onToggleSelect={vi.fn()}
        onSelectGroup={vi.fn()}
        onRowClick={vi.fn()}
        onCompareGroup={vi.fn()}
        variant="table"
      />
    </LocaleProvider>,
  )
}

describe('ReportGroupList "Compare all"', () => {
  it('hides the button when the group\'s reports share no dataset', () => {
    renderGroups([groupOf([reportOn('run-a', 'mmmlu'), reportOn('run-b', 'hellaswag_hi')])])

    expect(screen.queryByRole('button', { name: /compare all/i })).not.toBeInTheDocument()
  })

  it('shows the button when at least two reports share a dataset', () => {
    renderGroups([groupOf([reportOn('run-a', 'mmmlu'), reportOn('run-b', 'mmmlu')])])

    expect(screen.getByRole('button', { name: /compare all/i })).toBeInTheDocument()
  })

  it('hides the button for a single-report group regardless of dataset overlap', () => {
    renderGroups([groupOf([reportOn('run-a', 'mmmlu')])])

    expect(screen.queryByRole('button', { name: /compare all/i })).not.toBeInTheDocument()
  })
})
