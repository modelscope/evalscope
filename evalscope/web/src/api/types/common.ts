// ------------------------------------------------------------------ //
// Shared, cross-domain API response types                             //
// ------------------------------------------------------------------ //
//
// Generic response shapes reused across more than one API domain are inferred
// from the common runtime schemas so compile-time and runtime contracts cannot
// drift.
export type {
  ConfigResponse,
  DataFrameResponse,
  LogResponse,
  TaskStatusResponse,
} from '@/api/schemas/common.schema'
