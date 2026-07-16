import { useState } from 'react'
import type { Meta, StoryObj } from '@storybook/react-vite'
import Tabs, { type TabItem } from './Tabs'

/**
 * Storybook stories for the Tabs_Component (Req 14.3).
 *
 * `Tabs` implements the WAI-ARIA tablist pattern: `tablist` / `tab` / `tabpanel`
 * roles, one-to-one `aria-controls` / `aria-labelledby` wiring, a single
 * `aria-selected` tab, roving `tabindex`, arrow-key navigation with wrap-around
 * and Enter/Space activation. When a `panels` map is provided, exactly one panel
 * is visible; a tab whose panel reference is missing is skipped and an error is
 * surfaced (Req 11.1–11.8).
 *
 * The component is controlled, so these stories wrap it in a small stateful
 * container that owns `activeKey`.
 */

const TABS: TabItem[] = [
  { key: 'overview', label: 'Overview', panelId: 'panel-overview' },
  { key: 'details', label: 'Details', panelId: 'panel-details' },
  { key: 'predictions', label: 'Predictions', panelId: 'panel-predictions' },
]

const PANELS: Record<string, React.ReactNode> = {
  'panel-overview': <p className="p-4 text-sm text-[var(--text)]">Overview panel content.</p>,
  'panel-details': <p className="p-4 text-sm text-[var(--text)]">Details panel content.</p>,
  'panel-predictions': <p className="p-4 text-sm text-[var(--text)]">Predictions panel content.</p>,
}

interface ControlledTabsProps {
  tabs: TabItem[]
  panels: Record<string, React.ReactNode>
  initialKey: string
}

/** Stateful wrapper owning `activeKey` for the controlled `Tabs` component. */
function ControlledTabs({ tabs, panels, initialKey }: ControlledTabsProps) {
  const [activeKey, setActiveKey] = useState(initialKey)
  return <Tabs tabs={tabs} activeKey={activeKey} onChange={setActiveKey} panels={panels} />
}

const meta = {
  title: 'Tabs Component/Tabs',
  component: ControlledTabs,
  parameters: {
    layout: 'padded',
  },
  args: {
    tabs: TABS,
    panels: PANELS,
    initialKey: 'overview',
  },
} satisfies Meta<typeof ControlledTabs>

export default meta

type Story = StoryObj<typeof meta>

/** Default: three tabs, each wired to a managed panel; exactly one panel visible. */
export const WithPanels: Story = {}

/**
 * Orphan handling: the "broken" tab references a panel that does not exist, so it
 * is not rendered and an error message is surfaced (Req 11.8).
 */
export const WithOrphanTab: Story = {
  args: {
    tabs: [
      ...TABS,
      { key: 'broken', label: 'Broken', panelId: 'panel-missing' },
    ],
  },
}
