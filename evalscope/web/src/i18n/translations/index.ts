import type { Dict, Locale } from './types'

import * as nav from './nav'
import * as single from './single'
import * as multi from './multi'
import * as evaluation from './eval'
import * as perf from './perf'
import * as benchmarks from './benchmarks'
import * as prediction from './prediction'
import * as common from './common'
import * as markdown from './markdown'
import * as charts from './charts'
import * as trace from './trace'
import * as reports from './reports'
import * as reportDetail from './reportDetail'
import * as metrics from './metrics'
import * as compare from './compare'
import * as dashboard from './dashboard'
import * as performance from './performance'
import * as tasks from './tasks'
import * as tabs from './tabs'
import * as form from './form'
import * as empty from './empty'

export type { Locale, Dict }

const en: Dict = {
  nav: nav.en,
  single: single.en,
  multi: multi.en,
  eval: evaluation.en,
  perf: perf.en,
  benchmarks: benchmarks.en,
  prediction: prediction.en,
  common: common.en,
  markdown: markdown.en,
  charts: charts.en,
  trace: trace.en,
  reports: reports.en,
  reportDetail: reportDetail.en,
  metrics: metrics.en,
  compare: compare.en,
  dashboard: dashboard.en,
  performance: performance.en,
  tasks: tasks.en,
  tabs: tabs.en,
  form: form.en,
  empty: empty.en,
}

const zh: Dict = {
  nav: nav.zh,
  single: single.zh,
  multi: multi.zh,
  eval: evaluation.zh,
  perf: perf.zh,
  benchmarks: benchmarks.zh,
  prediction: prediction.zh,
  common: common.zh,
  markdown: markdown.zh,
  charts: charts.zh,
  trace: trace.zh,
  reports: reports.zh,
  reportDetail: reportDetail.zh,
  metrics: metrics.zh,
  compare: compare.zh,
  dashboard: dashboard.zh,
  performance: performance.zh,
  tasks: tasks.zh,
  tabs: tabs.zh,
  form: form.zh,
  empty: empty.zh,
}

const translations: Record<Locale, Dict> = { en, zh }

/**
 * Raw, nested translation dictionaries keyed by locale.
 *
 * Exposed (in addition to `lookupTranslation`) so tooling such as the locale
 * key drift checker can compare the key sets of different locales. Consumers
 * MUST treat the returned structure as read-only.
 */
export const localeDictionaries: Readonly<Record<Locale, Dict>> = translations

function lookup(locale: Locale, path: string): string | undefined {
  const keys = path.split('.')
  let node: string | Dict = translations[locale]
  for (const key of keys) {
    if (typeof node === 'string') return undefined
    node = (node as Dict)[key]
    if (node === undefined) return undefined
  }
  return typeof node === 'string' ? node : undefined
}

export function lookupTranslation(locale: Locale, path: string): string {
  return lookup(locale, path) ?? lookup('en', path) ?? path
}
