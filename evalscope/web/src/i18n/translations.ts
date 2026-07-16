/**
 * Barrel entry point for the i18n translation resources.
 *
 * The actual translation data is split by domain under `./translations/*`
 * (see `./translations/index.ts` for the composition and public API). This
 * file is intentionally kept as a re-export so existing import specifiers such
 * as `@/i18n/translations` and `../../src/i18n/translations.ts` continue to
 * resolve without any changes.
 */
export type { Locale, Dict } from './translations/index'
export { localeDictionaries, lookupTranslation } from './translations/index'
