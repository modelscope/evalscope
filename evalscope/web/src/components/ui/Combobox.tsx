import { useCallback, useEffect, useId, useRef, useState } from 'react'
import { ChevronDown } from 'lucide-react'
import { cn } from '@/lib/utils'
import { useLocale } from '@/contexts/LocaleContext'

/** A single selectable option. `labelKey` is resolved through the active locale. */
export interface ComboboxOption {
  value: string
  labelKey: string
}

export interface ComboboxProps {
  /** Selectable options; each label is localized via `labelKey`. */
  options: ComboboxOption[]
  /** Currently selected option value. */
  value: string
  /** Invoked with the newly selected option value. */
  onChange: (value: string) => void
  /** Localized field label (Req 10.11); associated as the accessible name. */
  labelKey: string
  /** Optional stable form control name. */
  name?: string
  /** Optional placeholder shown when no value is selected. */
  placeholderKey?: string
  /** Optional additional class names for the trigger element. */
  className?: string
  /** Optional disabled state. */
  disabled?: boolean
}

/**
 * Dataset_Combobox — an accessible single-select combobox (Req 10.5).
 *
 * Implements the WAI-ARIA APG "combobox with listbox popup" pattern:
 * - The trigger exposes `role="combobox"` with `aria-expanded`, `aria-controls`
 *   (pointing at the listbox) and `aria-activedescendant` (pointing at the
 *   highlighted option).
 * - The popup uses `role="listbox"`; each item uses `role="option"` with a
 *   unique `id` and `aria-selected`.
 * - Keyboard: ArrowDown/ArrowUp move the highlight (clamped, no wrap per the
 *   APG combobox model), Enter selects the highlighted option, Escape closes.
 * - Pointer: clicking an option selects it.
 *
 * The accessible name is provided programmatically via `aria-labelledby`
 * referencing the visible localized label.
 */
export default function Combobox({
  options,
  value,
  onChange,
  labelKey,
  name,
  placeholderKey,
  className,
  disabled,
}: ComboboxProps) {
  const { t } = useLocale()

  // Stable id roots for ARIA relationships (label / trigger / listbox / options).
  const baseId = useId()
  const labelId = `${baseId}-label`
  const triggerId = `${baseId}-trigger`
  const listboxId = `${baseId}-listbox`
  const optionId = useCallback((index: number) => `${baseId}-option-${index}`, [baseId])

  const [open, setOpen] = useState(false)
  // Index of the visually highlighted option while the popup is open.
  const [activeIndex, setActiveIndex] = useState(-1)

  const rootRef = useRef<HTMLDivElement>(null)
  const triggerRef = useRef<HTMLButtonElement>(null)
  const listboxRef = useRef<HTMLUListElement>(null)

  const selectedIndex = options.findIndex((opt) => opt.value === value)
  const selectedOption = selectedIndex >= 0 ? options[selectedIndex] : undefined

  /** Open the popup and highlight the current selection (or the first option). */
  const openList = useCallback(() => {
    if (disabled || options.length === 0) {
      return
    }
    setOpen(true)
    setActiveIndex(selectedIndex >= 0 ? selectedIndex : 0)
  }, [disabled, options.length, selectedIndex])

  /** Close the popup and return focus to the trigger. */
  const closeList = useCallback(() => {
    setOpen(false)
    setActiveIndex(-1)
    triggerRef.current?.focus()
  }, [])

  /** Commit the option at `index` as the new value, then close. */
  const commitIndex = useCallback(
    (index: number) => {
      const opt = options[index]
      if (opt) {
        onChange(opt.value)
      }
      closeList()
    },
    [options, onChange, closeList],
  )

  // Close when a pointer interaction lands outside the component.
  useEffect(() => {
    if (!open) {
      return
    }
    const handlePointerDown = (event: MouseEvent) => {
      if (rootRef.current && !rootRef.current.contains(event.target as Node)) {
        setOpen(false)
        setActiveIndex(-1)
      }
    }
    document.addEventListener('mousedown', handlePointerDown)
    return () => document.removeEventListener('mousedown', handlePointerDown)
  }, [open])

  // Keep the highlighted option scrolled into view within the listbox.
  useEffect(() => {
    if (!open || activeIndex < 0) {
      return
    }
    const el = document.getElementById(optionId(activeIndex))
    el?.scrollIntoView({ block: 'nearest' })
  }, [open, activeIndex, optionId])

  const handleTriggerKeyDown = (event: React.KeyboardEvent<HTMLButtonElement>) => {
    switch (event.key) {
      case 'ArrowDown':
        event.preventDefault()
        if (!open) {
          openList()
        } else {
          // Move down without wrapping (clamped to the last option).
          setActiveIndex((prev) => Math.min(prev + 1, options.length - 1))
        }
        break
      case 'ArrowUp':
        event.preventDefault()
        if (!open) {
          openList()
        } else {
          // Move up without wrapping (clamped to the first option).
          setActiveIndex((prev) => Math.max(prev - 1, 0))
        }
        break
      case 'Enter':
        if (open) {
          event.preventDefault()
          if (activeIndex >= 0) {
            commitIndex(activeIndex)
          }
        }
        break
      case 'Escape':
        if (open) {
          event.preventDefault()
          closeList()
        }
        break
      case ' ':
        // Space toggles the popup open (matches native select ergonomics).
        if (!open) {
          event.preventDefault()
          openList()
        }
        break
      default:
        break
    }
  }

  const triggerLabel = selectedOption
    ? t(selectedOption.labelKey)
    : placeholderKey
      ? t(placeholderKey)
      : ''

  return (
    <div ref={rootRef} className="flex flex-col gap-1.5">
      <label
        id={labelId}
        htmlFor={triggerId}
        className="text-xs font-medium uppercase tracking-wider text-[var(--text-muted)]"
      >
        {t(labelKey)}
      </label>
      <div className="relative">
        <button
          ref={triggerRef}
          id={triggerId}
          type="button"
          role="combobox"
          name={name}
          disabled={disabled}
          aria-haspopup="listbox"
          aria-expanded={open}
          aria-controls={listboxId}
          aria-labelledby={labelId}
          aria-activedescendant={open && activeIndex >= 0 ? optionId(activeIndex) : undefined}
          onClick={() => (open ? closeList() : openList())}
          onKeyDown={handleTriggerKeyDown}
          className={cn(
            'w-full flex items-center justify-between gap-2 px-3 py-2 text-sm text-left rounded-[var(--radius-sm)]',
            'bg-[var(--bg-deep)] border border-[var(--border)] text-[var(--text)]',
            'focus:outline-none focus:border-[var(--accent)] focus:ring-1 focus:ring-[var(--accent-dim)]',
            'transition-all duration-[var(--transition)]',
            'disabled:opacity-50 disabled:cursor-not-allowed',
            // text-dim allowed: combobox placeholder state (DESIGN.md §Text)
            !selectedOption && 'text-[var(--text-dim)]',
            className,
          )}
        >
          <span className="truncate">{triggerLabel}</span>
          {/* text-dim allowed: combobox chevron icon (DESIGN.md §Text) */}
          <ChevronDown size={14} className="shrink-0 text-[var(--text-dim)]" aria-hidden="true" />
        </button>

        {open && (
          <ul
            ref={listboxRef}
            id={listboxId}
            role="listbox"
            aria-labelledby={labelId}
            className={cn(
              'absolute z-50 mt-1 max-h-60 w-full overflow-auto py-1',
              'rounded-[var(--radius-sm)] border border-[var(--border)] bg-[var(--bg-deep)]',
              'shadow-lg',
            )}
          >
            {options.map((opt, index) => {
              const isSelected = opt.value === value
              const isActive = index === activeIndex
              return (
                <li
                  key={opt.value}
                  id={optionId(index)}
                  role="option"
                  aria-selected={isSelected}
                  // Prevent the trigger from losing focus before the click resolves.
                  onMouseDown={(event) => event.preventDefault()}
                  onMouseEnter={() => setActiveIndex(index)}
                  onClick={() => commitIndex(index)}
                  className={cn(
                    'cursor-pointer px-3 py-2 text-sm text-[var(--text)]',
                    isActive && 'bg-[var(--accent-dim)]',
                    isSelected && 'font-medium',
                  )}
                >
                  {t(opt.labelKey)}
                </li>
              )
            })}
          </ul>
        )}
      </div>
    </div>
  )
}
