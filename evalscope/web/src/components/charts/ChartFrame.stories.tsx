import type { Meta, StoryObj } from '@storybook/react-vite'
import ChartFrame from './ChartFrame'
import DataTableFallback, { type DataTableModel } from '@/components/common/DataTableFallback'
import type { ChartTheme } from '@/domain/chart/chartTheme'

/**
 * Storybook stories for the Chart_Renderer (Req 14.3).
 *
 * `ChartFrame` preflights its chart URL with a real `fetch` before mounting the
 * iframe, so the deterministic, network-independent baselines here focus on:
 * - the failure state (unreachable URL → visible error text + retry + the
 *   authoritative data-table fallback, never a blank iframe — Req 2.6, 2.8), and
 * - the data-table fallback UI on its own.
 *
 * A "ready" story uses a self-contained `data:` URL so the preflight succeeds
 * and the iframe renders without any external dependency. The active theme is
 * driven by the toolbar theme global so both light/dark baselines are captured.
 */

/** Shared authoritative data-table model presented alongside/instead of a chart. */
const fallbackTable: DataTableModel = {
  columns: ['Dataset', 'Accuracy', 'F1'],
  rows: [
    { Dataset: 'gsm8k', Accuracy: 0.921, F1: 0.904 },
    { Dataset: 'math', Accuracy: 0.783, F1: 0.771 },
    { Dataset: 'humaneval', Accuracy: 0.655, F1: 0.642 },
  ],
  scoreColumns: ['Accuracy', 'F1'],
}

/** A self-contained chart-like document used for the successful "ready" baseline. */
const READY_CHART_SRC =
  'data:text/html,' +
  encodeURIComponent(
    '<!doctype html><html><body style="margin:0;font-family:sans-serif">' +
      '<div style="height:100%;display:flex;align-items:center;justify-content:center">' +
      'Rendered chart placeholder</div></body></html>',
  )

const meta = {
  title: 'Chart Renderer/ChartFrame',
  component: ChartFrame,
  parameters: {
    layout: 'padded',
  },
  args: {
    baseSrc: '/charts/unreachable-chart.html',
    theme: 'dark',
    height: 360,
    title: 'Score by dataset',
    fallbackTable,
  },
  // Drive the chart theme from the toolbar theme global so light/dark baselines
  // both render (Req 2.1, 2.2).
  render: (args, { globals }) => (
    <ChartFrame {...args} theme={(globals.theme as ChartTheme) ?? 'dark'} />
  ),
} satisfies Meta<typeof ChartFrame>

export default meta

type Story = StoryObj<typeof meta>

/**
 * Failure baseline: the preflight cannot reach the URL, so the frame shows a
 * visible error message, a retry entry point and the authoritative data-table
 * fallback — never a blank iframe (Req 2.6, 2.8, 2.9).
 */
export const LoadErrorWithFallback: Story = {
  args: { baseSrc: '/charts/unreachable-chart.html' },
}

/**
 * Ready baseline: a self-contained `data:` document passes preflight and the
 * iframe renders, exercising the loading → ready transition (Req 2.7).
 */
export const Ready: Story = {
  args: { baseSrc: READY_CHART_SRC, title: 'Rendered chart' },
}

/**
 * The data-table fallback in isolation — the authoritative representation of the
 * chart's underlying data (Req 2.8).
 */
export const FallbackTableOnly: Story = {
  render: () => (
    <div className="rounded-[var(--radius)] border border-[var(--border)] bg-[var(--bg-card)] p-4">
      <DataTableFallback model={fallbackTable} />
    </div>
  ),
}
