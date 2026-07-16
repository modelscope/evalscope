import type { Meta, StoryObj } from '@storybook/react-vite'
import { MemoryRouter } from 'react-router-dom'
import EmptyStateSystem from './EmptyStateSystem'

/**
 * Storybook stories for the Empty_State_System (Req 14.3).
 *
 * `EmptyStateSystem` renders a reason-specific message plus 1–3 in-product
 * recovery actions, each of which navigates via the router. It therefore needs a
 * Router in scope, provided here through a `MemoryRouter` decorator. The three
 * distinguishable reasons — no-data, load-error and no-match — are each captured
 * as a baseline (Req 6.1, 6.2).
 *
 * `revealDelayMs` is set to 0 so the empty state is visible immediately for the
 * baseline (the component otherwise reveals within a 300ms budget).
 */

const meta = {
  title: 'Empty State System/EmptyStateSystem',
  component: EmptyStateSystem,
  parameters: {
    layout: 'padded',
  },
  decorators: [
    (Story) => (
      <MemoryRouter>
        <div className="w-[520px]">
          <Story />
        </div>
      </MemoryRouter>
    ),
  ],
  args: {
    reason: 'no-data',
    loading: false,
    revealDelayMs: 0,
    context: { view: 'reports' },
  },
} satisfies Meta<typeof EmptyStateSystem>

export default meta

type Story = StoryObj<typeof meta>

/** No records yet: steers the user toward creating work / browsing benchmarks. */
export const NoData: Story = {
  args: { reason: 'no-data', context: { view: 'reports' } },
}

/** Load failed: offers retry plus a safe navigation fallback. */
export const LoadError: Story = {
  args: { reason: 'load-error', context: { view: 'reports' } },
}

/** Filters excluded everything: offers to clear filters or start fresh work. */
export const NoMatch: Story = {
  args: { reason: 'no-match', context: { view: 'reports' } },
}

/**
 * Performance empty state: provides an in-product entry to the task-creation
 * flow rather than a CLI-only action (Req 6.3).
 */
export const PerformanceNoData: Story = {
  args: { reason: 'no-data', context: { view: 'performance' } },
}
