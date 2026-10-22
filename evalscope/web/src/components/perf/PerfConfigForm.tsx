import type { ReactNode } from 'react'
import { useLocale } from '@/contexts/LocaleContext'
import Field from '@/components/ui/Field'
import { FORM_INPUT_CLASS, inputClass } from '@/components/ui/formStyles'
import { buildPerfPayload, validatePerfDraft, type PerfDraft } from '@/components/tasks/taskConfig'
import TaskFormShell from '@/components/tasks/TaskFormShell'
import { useTaskForm } from '@/components/tasks/useTaskForm'

interface Props {
  onSubmit: (config: Record<string, unknown>) => void
  disabled?: boolean
  draft: PerfDraft
  onChange: (draft: PerfDraft) => void
  toolbar?: ReactNode
  onValidateConnection: () => Record<string, string>
}

/** Stable field ids, reused as label/error association targets and focus targets. */
const IDS = {
  parallel: 'perf-parallel',
  number: 'perf-number',
  rate: 'perf-rate',
  maxTokens: 'perf-maxTokens',
  minTokens: 'perf-minTokens',
  dataset: 'perf-dataset',
  maxPromptLen: 'perf-maxPromptLen',
  minPromptLen: 'perf-minPromptLen',
  tokenizerPath: 'perf-tokenizerPath',
} as const

/** DOM order of focusable fields, drives first-invalid focus on submit. */
const DOM_ORDER: string[] = [
  'connection-model',
  'connection-url',
  IDS.parallel,
  IDS.number,
  IDS.rate,
  IDS.maxTokens,
  IDS.minTokens,
  IDS.dataset,
  IDS.maxPromptLen,
  IDS.minPromptLen,
  IDS.tokenizerPath,
]

/** Fields that live inside the collapsible "More Parameters" section. */
const MORE_PARAMS_IDS: string[] = [IDS.tokenizerPath]

export default function PerfConfigForm({ onSubmit, disabled, draft, onChange, toolbar, onValidateConnection }: Props) {
  const { t } = useLocale()
  const { parallel, number, rate, maxTokens, minTokens, dataset, maxPromptLen, minPromptLen, tokenizerPath } = draft
  const changeField = <K extends keyof PerfDraft>(field: K, value: PerfDraft[K]) => {
    onChange({ ...draft, [field]: value })
    clearErr(`perf-${field}`)
  }
  const setParallel = (value: string) => changeField('parallel', value)
  const setNumber = (value: string) => changeField('number', value)
  const setRate = (value: string) => changeField('rate', value)
  const setMaxTokens = (value: string) => changeField('maxTokens', value)
  const setMinTokens = (value: string) => changeField('minTokens', value)
  const setDataset = (value: string) => changeField('dataset', value)
  const setMaxPromptLen = (value: string) => changeField('maxPromptLen', value)
  const setMinPromptLen = (value: string) => changeField('minPromptLen', value)
  const setTokenizerPath = (value: string) => changeField('tokenizerPath', value)

  const {
    errorFor: errMsg,
    clearError: clearErr,
    showMore,
    toggleMore,
    submitHandler,
  } = useTaskForm({
    domOrder: DOM_ORDER,
    moreParamsIds: MORE_PARAMS_IDS,
    values: Object.fromEntries(Object.entries(draft).map(([field, value]) => [`perf-${field}`, value])),
  })

  const validate = () => ({
    ...onValidateConnection(),
    ...Object.fromEntries(Object.entries(validatePerfDraft(draft)).map(([field, error]) => [`perf-${field}`, error])),
  })
  const buildConfig = () => onSubmit(buildPerfPayload(draft))

  return (
    <TaskFormShell
      onSubmit={submitHandler(validate, buildConfig)}
      moreParamsColumns={2}
      moreParamsLabel={t('perf.task.moreParams')}
      showMore={showMore}
      onToggleMore={toggleMore}
      submitLabel={t('perf.task.startPerf')}
      disabled={disabled}
      toolbar={toolbar}
      moreParams={
        <Field
          id={IDS.tokenizerPath}
          name="tokenizer_path"
          labelKey="perf.task.tokenizerPath"
          error={errMsg(IDS.tokenizerPath)}
        >
          {(aria) => (
            <input
              {...aria}
              type="text"
              value={tokenizerPath}
              onChange={(e) => {
                setTokenizerPath(e.target.value)
                clearErr(IDS.tokenizerPath)
              }}
              className={FORM_INPUT_CLASS}
              placeholder="/path/to/tokenizer"
            />
          )}
        </Field>
      }
    >
      <Field id={IDS.parallel} name="parallel" labelKey="perf.task.parallel" error={errMsg(IDS.parallel)}>
        {(aria) => (
          <input
            {...aria}
            type="text"
            inputMode="numeric"
            value={parallel}
            onChange={(e) => {
              setParallel(e.target.value)
              clearErr(IDS.parallel)
            }}
            className={inputClass(errMsg(IDS.parallel))}
            placeholder="1, 4, 8"
          />
        )}
      </Field>

      <Field id={IDS.number} name="number" labelKey="perf.task.number" error={errMsg(IDS.number)}>
        {(aria) => (
          <input
            {...aria}
            type="text"
            inputMode="numeric"
            value={number}
            onChange={(e) => {
              setNumber(e.target.value)
              clearErr(IDS.number)
            }}
            className={inputClass(errMsg(IDS.number))}
            placeholder="10, 100"
          />
        )}
      </Field>

      <Field id={IDS.rate} name="rate" labelKey="perf.task.rate" error={errMsg(IDS.rate)} hintKey="perf.task.rateHint">
        {(aria) => (
          <input
            {...aria}
            type="number"
            step="any"
            value={rate}
            onChange={(e) => {
              setRate(e.target.value)
              clearErr(IDS.rate)
            }}
            className={inputClass(errMsg(IDS.rate))}
          />
        )}
      </Field>

      <Field id={IDS.maxTokens} name="max_tokens" labelKey="perf.task.maxTokens" error={errMsg(IDS.maxTokens)}>
        {(aria) => (
          <input
            {...aria}
            type="number"
            min={1}
            value={maxTokens}
            onChange={(e) => {
              setMaxTokens(e.target.value)
              clearErr(IDS.maxTokens)
            }}
            className={inputClass(errMsg(IDS.maxTokens))}
          />
        )}
      </Field>

      <Field id={IDS.minTokens} name="min_tokens" labelKey="perf.task.minTokens" error={errMsg(IDS.minTokens)}>
        {(aria) => (
          <input
            {...aria}
            type="number"
            min={0}
            value={minTokens}
            onChange={(e) => {
              setMinTokens(e.target.value)
              clearErr(IDS.minTokens)
            }}
            className={inputClass(errMsg(IDS.minTokens))}
          />
        )}
      </Field>

      <Field id={IDS.dataset} name="dataset" labelKey="perf.task.dataset">
        {(aria) => (
          <input
            {...aria}
            type="text"
            value={dataset}
            onChange={(e) => setDataset(e.target.value)}
            className={FORM_INPUT_CLASS}
            placeholder="openqa"
          />
        )}
      </Field>

      <Field
        id={IDS.maxPromptLen}
        name="max_prompt_length"
        labelKey="perf.task.maxPromptLen"
        error={errMsg(IDS.maxPromptLen)}
      >
        {(aria) => (
          <input
            {...aria}
            type="number"
            min={0}
            value={maxPromptLen}
            onChange={(e) => {
              setMaxPromptLen(e.target.value)
              clearErr(IDS.maxPromptLen)
            }}
            className={inputClass(errMsg(IDS.maxPromptLen))}
          />
        )}
      </Field>

      <Field
        id={IDS.minPromptLen}
        name="min_prompt_length"
        labelKey="perf.task.minPromptLen"
        error={errMsg(IDS.minPromptLen)}
      >
        {(aria) => (
          <input
            {...aria}
            type="number"
            min={0}
            value={minPromptLen}
            onChange={(e) => {
              setMinPromptLen(e.target.value)
              clearErr(IDS.minPromptLen)
            }}
            className={inputClass(errMsg(IDS.minPromptLen))}
          />
        )}
      </Field>
    </TaskFormShell>
  )
}
