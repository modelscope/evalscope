import { useId, useRef, useState, type KeyboardEvent, type ReactNode } from 'react'

import { useLocale } from '@/contexts/LocaleContext'
import { cn } from '@/lib/utils'
import { moveRovingIndex, type RovingOrientation } from '@/domain/tabs/roving'

/**
 * A single tab descriptor.
 *
 * The component supports two labelling strategies for a smooth migration:
 * - `labelKey`: a translation key resolved through the active locale (preferred,
 *   matches the design contract).
 * - `label`: a pre-localized string (legacy callers). Used only when `labelKey`
 *   is absent.
 */
export interface TabItem {
  /** Stable identity of the tab, also used as the `onChange` payload. */
  key: string
  /** Translation key resolved via the current locale. Preferred over `label`. */
  labelKey?: string
  /** Pre-localized label (legacy API). Prefer `labelKey`. */
  label?: string
  /**
   * The `id` of the corresponding tabpanel. A tab whose `panelId` has no matching
   * entry in the required `panels` map is treated as invalid: it is not
   * rendered and an error is surfaced (Req 11.8).
   */
  panelId?: string
}

export interface TabsProps {
  /** The ordered list of tabs. */
  tabs: TabItem[]
  /** The currently active tab key. */
  activeKey: string
  /** Invoked with the tab key when a tab is activated. */
  onChange: (key: string) => void
  /**
   * Required panel content keyed by `panelId`. The component renders
   * `role="tabpanel"` elements, guarantees exactly one visible panel (Req 11.7),
   * and enforces the one-to-one tab/panel relationship (Req 11.2, 11.8).
   */
  panels: Record<string, ReactNode>
  /** Tablist orientation, controls which arrow keys move focus. */
  orientation?: RovingOrientation
  /** Extra class names applied to the tablist container. */
  className?: string
}

/**
 * Accessible tabs implementing the WAI-ARIA tablist pattern (Req 11).
 *
 * - Exposes `tablist` / `tab` / `tabpanel` roles (Req 11.1).
 * - Wires `aria-controls` (tab -> panel) and `aria-labelledby` (panel -> tab)
 *   with no orphan references (Req 11.2).
 * - Keeps exactly one `aria-selected=true` tab (Req 11.3).
 * - Applies roving `tabindex` (selected 0, others -1) (Req 11.4).
 * - Moves focus with arrow / Home / End keys and wraps around (Req 11.5).
 * - Activates the focused tab on Enter / Space (Req 11.6).
 * - Renders exactly one visible tabpanel when `panels` is supplied (Req 11.7).
 * - Skips tabs whose panel reference is missing and reports an error (Req 11.8).
 */
export default function Tabs({
  tabs,
  activeKey,
  onChange,
  panels,
  orientation = 'horizontal',
  className,
}: TabsProps) {
  const { t } = useLocale()
  // Stable id namespace so tab and panel ids can reference each other.
  const baseId = useId()

  const isValid = (tab: TabItem): boolean =>
    tab.panelId !== undefined && Object.prototype.hasOwnProperty.call(panels, tab.panelId)

  const validTabs = tabs.filter(isValid)
  const invalidTabs = tabs.filter((tab) => !isValid(tab))

  // Guarantee exactly one selected tab: fall back to the first valid tab when
  // `activeKey` does not match any renderable tab (Req 11.3).
  const selectedKey = validTabs.some((tab) => tab.key === activeKey) ? activeKey : validTabs[0]?.key

  // Track which tab currently owns the roving `tabindex=0` / DOM focus. It is
  // decoupled from selection so arrow keys can move focus without activating.
  const [focusKey, setFocusKey] = useState<string | undefined>(selectedKey)
  const effectiveFocusKey = validTabs.some((tab) => tab.key === focusKey) ? focusKey : selectedKey

  const tabRefs = useRef<Record<string, HTMLButtonElement | null>>({})

  const tabDomId = (key: string): string => `${baseId}-tab-${key}`
  const panelDomId = (tab: TabItem): string | undefined => tab.panelId

  const resolveLabel = (tab: TabItem): string => (tab.labelKey ? t(tab.labelKey) : (tab.label ?? tab.key))

  const handleKeyDown = (event: KeyboardEvent<HTMLButtonElement>, index: number, tabKey: string) => {
    if (event.key === 'Enter' || event.key === ' ' || event.key === 'Spacebar') {
      // Manual activation: activate the currently focused tab (Req 11.6).
      event.preventDefault()
      onChange(tabKey)
      return
    }

    // Arrow / Home / End move focus with wrap-around (Req 11.5).
    const nextIndex = moveRovingIndex(index, validTabs.length, event.key, orientation)
    if (nextIndex === null || nextIndex < 0) {
      return
    }
    event.preventDefault()
    const nextKey = validTabs[nextIndex].key
    setFocusKey(nextKey)
    tabRefs.current[nextKey]?.focus()
  }

  return (
    <>
      <div
        role="tablist"
        aria-orientation={orientation}
        className={cn(
          'inline-flex items-center gap-1 p-1 rounded-[var(--radius)] bg-[var(--bg-deep)] border border-[var(--border)]',
          className,
        )}
      >
        {validTabs.map((tab, index) => {
          const isSelected = tab.key === selectedKey
          const panelId = panelDomId(tab)
          return (
            <button
              key={tab.key}
              ref={(el) => {
                tabRefs.current[tab.key] = el
              }}
              id={tabDomId(tab.key)}
              role="tab"
              type="button"
              aria-selected={isSelected}
              aria-controls={panelId}
              tabIndex={tab.key === effectiveFocusKey ? 0 : -1}
              onClick={() => {
                setFocusKey(tab.key)
                onChange(tab.key)
              }}
              onFocus={() => setFocusKey(tab.key)}
              onKeyDown={(event) => handleKeyDown(event, index, tab.key)}
              className={cn(
                'px-4 py-1.5 text-sm font-medium rounded-[var(--radius-sm)] transition-all duration-[var(--transition)] cursor-pointer',
                isSelected
                  ? 'bg-[var(--accent)] text-[var(--text-on-filled)] shadow-[var(--shadow-glow-soft)]'
                  : 'text-[var(--text-muted)] hover:text-[var(--text)] bg-[var(--bg-card)] hover:bg-[var(--bg-card2)]',
              )}
            >
              {resolveLabel(tab)}
            </button>
          )
        })}
      </div>

      {validTabs.map((tab) => {
          const isSelected = tab.key === selectedKey
          return (
            <div
              key={tab.key}
              id={tab.panelId}
              role="tabpanel"
              aria-labelledby={tabDomId(tab.key)}
              hidden={!isSelected}
            >
              {isSelected ? panels[tab.panelId as string] : null}
            </div>
          )
        })}

      {invalidTabs.length > 0 && (
        <div role="alert" className="text-sm text-[var(--danger)]">
          {t('tabs.invalidPanel', { keys: invalidTabs.map((tab) => tab.key).join(', ') })}
        </div>
      )}
    </>
  )
}
