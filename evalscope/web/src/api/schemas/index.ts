/**
 * Barrel re-export for the domain-split API schemas.
 *
 * Re-exporting every domain schema (and its inferred type) from a single entry
 * point keeps existing import paths stable (`from '@/api/schemas'`) even as the
 * schemas are split by domain across `reports`, `perf`, and `eval` files.
 */
export * from './reports.schema'
export * from './perf.schema'
export * from './eval.schema'
export * from './common.schema'
