import type { Meta, StoryObj } from '@storybook/react-vite'
import Field from './Field'
import Combobox from './Combobox'
import { cn } from '@/lib/utils'

/**
 * Storybook stories for the Field_Primitive (Req 14.3).
 *
 * `Field` is control-agnostic: it renders a localized `<label>` and hands the
 * ARIA props (`id`, `name`, `aria-labelledby`, `aria-invalid`,
 * `aria-describedby`, `autoComplete`) to the caller through a render prop, which
 * spreads them onto the actual control. These stories wire that render prop to a
 * plain styled `<input>` to show the default / required / error states, plus the
 * accessible Dataset_Combobox that shares the same accessibility contract.
 */

/** Shared input styling matching the app's form controls. */
const inputClass = cn(
  'w-full px-3 py-2 text-sm rounded-[var(--radius-sm)]',
  'bg-[var(--bg-deep)] border border-[var(--border)] text-[var(--text)]',
  'placeholder:text-[var(--text-dim)]',
  'focus:outline-none focus:border-[var(--accent)] focus:ring-1 focus:ring-[var(--accent-dim)]',
  'transition-all duration-[var(--transition)]',
)

const inputErrorClass = cn(
  inputClass,
  'border-[var(--danger)] focus:border-[var(--danger)] focus:ring-[var(--danger-bg)]',
)

const meta = {
  title: 'Field Primitive/Field',
  component: Field,
  parameters: {
    layout: 'centered',
  },
  decorators: [
    (Story) => (
      <div className="w-[320px]">
        <Story />
      </div>
    ),
  ],
  args: {
    id: 'model-name',
    labelKey: 'eval.modelName',
    name: 'model-name',
    children: (aria) => <input {...aria} className={inputClass} placeholder="Qwen/Qwen2.5-0.5B-Instruct" />,
  },
} satisfies Meta<typeof Field>

export default meta

type Story = StoryObj<typeof meta>

/** Default state: localized label programmatically linked to the control (Req 10.2). */
export const Default: Story = {}

/** Required state: renders the required indicator next to the label. */
export const Required: Story = {
  args: {
    required: true,
  },
}

/**
 * Error state: the control is marked `aria-invalid` and associated with the live
 * error region via `aria-describedby` (Req 10.3). Error text is already localized
 * by the caller (Req 10.10).
 */
export const WithError: Story = {
  args: {
    required: true,
    error: 'This field is required.',
    children: (aria) => (
      <input {...aria} className={inputErrorClass} placeholder="Qwen/Qwen2.5-0.5B-Instruct" />
    ),
  },
}

/**
 * API key field: an accessible `autocomplete` hint is defined without changing
 * backend secret handling (Req 10.12).
 */
export const ApiKeyField: Story = {
  args: {
    id: 'api-key',
    labelKey: 'eval.apiKey',
    name: 'api-key',
    autoComplete: 'off',
    children: (aria) => <input {...aria} type="password" className={inputClass} placeholder="sk-..." />,
  },
}

/**
 * The Dataset_Combobox shares the Field accessibility contract (localized label,
 * programmatic accessible name) and adds full keyboard listbox semantics (Req 10.5).
 */
export const DatasetCombobox: Story = {
  render: () => (
    <Combobox
      labelKey="eval.datasets"
      value="gsm8k"
      onChange={() => {}}
      placeholderKey="eval.datasets"
      options={[
        { value: 'gsm8k', labelKey: 'metrics.accuracy' },
        { value: 'math', labelKey: 'metrics.f1' },
        { value: 'humaneval', labelKey: 'metrics.pass_rate' },
      ]}
    />
  ),
}
