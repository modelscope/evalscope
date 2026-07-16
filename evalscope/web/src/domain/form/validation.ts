/**
 * Task form validation orchestration (pure logic, no DOM / side effects).
 *
 * These helpers back the `Field_Primitive` form semantics for the Eval and
 * Performance task forms. They compute focus order for submit-time error
 * handling (Req 10.4), validate the free-form `Dataset_Args` JSON input without
 * mutating it (Req 10.6, 10.7) and enforce numeric min/max/step constraints
 * (Req 10.9).
 *
 * All user-facing text is referenced by locale `messageKey` rather than being
 * hard-coded, so the render layer can resolve strings through the locale system
 * (Req 10.10, 10.11). This module never resolves or hard-codes English copy.
 */

/** A validation error associated with a form field. */
export interface FieldError {
  /**
   * Id of the offending field. Numeric validation cannot know the field id, so
   * `validateNumeric` returns an empty string here and the caller associates the
   * error with the concrete field.
   */
  fieldId: string
  /** Locale key describing the error; never a hard-coded, human-readable string. */
  messageKey: string
}

/** Aggregate result of validating a whole form. */
export interface FormValidationResult {
  /** All field errors found during validation. */
  errors: FieldError[]
  /**
   * Id of the first invalid field in DOM order, used to move focus on a failed
   * submit (Req 10.4); `null` when there are no invalid fields.
   */
  firstInvalidId: string | null
}

/** Result of validating the `Dataset_Args` JSON input. */
export type DatasetArgsValidation =
  | { ok: true; value: Record<string, unknown> }
  | { ok: false; messageKey: string }

/**
 * Locale message keys emitted by this module.
 *
 * Centralized so the render layer and locale drift checks share a single source
 * of truth. Values are dot-notation keys, not display strings.
 */
export const FORM_MESSAGE_KEYS = {
  /** A required field is empty (Req 10.2, 10.10). */
  required: 'form.validation.required',
  /** `Dataset_Args` text is not parseable as JSON (Req 10.6). */
  datasetArgsInvalidJson: 'form.validation.datasetArgs.invalidJson',
  /** `Dataset_Args` is valid JSON but not a JSON object (Req 10.7). */
  datasetArgsInvalidStructure: 'form.validation.datasetArgs.invalidStructure',
  /** Numeric value is below the allowed minimum (Req 10.9). */
  numericBelowMin: 'form.validation.numeric.belowMin',
  /** Numeric value is above the allowed maximum (Req 10.9). */
  numericAboveMax: 'form.validation.numeric.aboveMax',
  /** Numeric value is not aligned to the required step (Req 10.9). */
  numericStepMismatch: 'form.validation.numeric.stepMismatch',
  /** Numeric value is not a finite number (Req 10.9). */
  numericNotFinite: 'form.validation.numeric.notFinite',
} as const

/**
 * Determine the first invalid field in DOM order.
 *
 * Scans `order` (the fields as they appear in the DOM) and returns the id of the
 * earliest field that is also present in `invalidIds`. This drives submit-time
 * focus management so the user lands on the first problem field (Req 10.4).
 *
 * The invalid set is looked up in `order`, not the other way around, so the
 * result always reflects DOM ordering regardless of how `invalidIds` was built.
 * Invalid ids that are not part of `order` are ignored.
 *
 * @param order - Field ids in DOM order.
 * @param invalidIds - The set of invalid field ids (accepts a `Set` or array).
 * @returns The first invalid id in DOM order, or `null` when none is invalid.
 */
export function computeFirstInvalid(order: string[], invalidIds: Set<string> | string[]): string | null {
  const invalid = invalidIds instanceof Set ? invalidIds : new Set(invalidIds)
  if (invalid.size === 0) {
    return null
  }
  for (const id of order) {
    if (invalid.has(id)) {
      return id
    }
  }
  return null
}

/**
 * Validate the free-form `Dataset_Args` JSON input.
 *
 * Parsing failures and structurally invalid input both block submission while
 * preserving the raw text (this function is pure and never mutates its input,
 * Req 10.6, 10.7):
 * - Non-JSON text -> `{ ok: false, messageKey: datasetArgsInvalidJson }`.
 * - Valid JSON that is not a JSON object (array, `null` or a primitive) ->
 *   `{ ok: false, messageKey: datasetArgsInvalidStructure }`.
 * - A JSON object -> `{ ok: true, value }` where `value` is the parsed object.
 *
 * @param rawText - The raw `Dataset_Args` text exactly as typed by the user.
 * @returns A discriminated result carrying the parsed object or an error key.
 */
export function validateDatasetArgs(rawText: string): DatasetArgsValidation {
  let parsed: unknown
  try {
    parsed = JSON.parse(rawText)
  } catch {
    return { ok: false, messageKey: FORM_MESSAGE_KEYS.datasetArgsInvalidJson }
  }

  // Expected structure is a JSON object; reject arrays, null and primitives.
  if (parsed === null || typeof parsed !== 'object' || Array.isArray(parsed)) {
    return { ok: false, messageKey: FORM_MESSAGE_KEYS.datasetArgsInvalidStructure }
  }

  return { ok: true, value: parsed as Record<string, unknown> }
}

/**
 * Validate a numeric field value against optional min/max/step constraints.
 *
 * A value outside the `[min, max]` range is a validation failure; a value within
 * the inclusive range is valid (Req 10.9). When a positive `step` is provided the
 * value must also align to the step grid (anchored at `min`, or `0` when `min`
 * is omitted). Non-finite values (`NaN`, `Infinity`) are treated as invalid.
 *
 * The returned `FieldError.fieldId` is empty because this helper is field
 * agnostic; the caller associates the error with the concrete field id.
 *
 * @param value - The numeric value to validate.
 * @param min - Optional inclusive lower bound.
 * @param max - Optional inclusive upper bound.
 * @param step - Optional positive step the value must align to.
 * @returns A `FieldError` when the value violates a constraint, otherwise `null`.
 */
export function validateNumeric(value: number, min?: number, max?: number, step?: number): FieldError | null {
  if (!Number.isFinite(value)) {
    return { fieldId: '', messageKey: FORM_MESSAGE_KEYS.numericNotFinite }
  }
  if (min !== undefined && value < min) {
    return { fieldId: '', messageKey: FORM_MESSAGE_KEYS.numericBelowMin }
  }
  if (max !== undefined && value > max) {
    return { fieldId: '', messageKey: FORM_MESSAGE_KEYS.numericAboveMax }
  }
  if (step !== undefined && step > 0) {
    // Anchor the step grid at `min` when present, otherwise at 0.
    const anchor = min ?? 0
    const offset = value - anchor
    const remainder = offset - Math.round(offset / step) * step
    // Tolerance guards against floating-point drift in the modulo computation.
    const tolerance = 1e-9 * Math.max(1, Math.abs(step))
    if (Math.abs(remainder) > tolerance) {
      return { fieldId: '', messageKey: FORM_MESSAGE_KEYS.numericStepMismatch }
    }
  }
  return null
}
