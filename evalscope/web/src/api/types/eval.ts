// ------------------------------------------------------------------ //
// "eval" API domain types                                             //
// ------------------------------------------------------------------ //
//
// All endpoint response types are derived from runtime zod schemas; this file
// preserves the domain import boundary without duplicating contracts.
export type {
  BenchmarkEntry,
  BenchmarksResponse,
  InvokeStatus,
  EvalInvokeResponse,
  ProgressResponse,
} from '@/api/schemas/eval.schema'
