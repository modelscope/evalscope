// ------------------------------------------------------------------ //
// Shared TypeScript types for API responses                           //
// ------------------------------------------------------------------ //
//
// Barrel re-export for the domain-split API types. Splitting the former
// single `types.ts` into per-domain files (`reports`, `perf`, `eval`,
// `common`) keeps concerns isolated while this barrel preserves the stable
// external import path `@/api/types` — existing consumers need no changes.
// Schema-backed types are derived via `z.infer` from
// `src/api/schemas/*` inside the domain files.

export type * from './common'
export type * from './reports'
export type * from './perf'
export type * from './eval'
