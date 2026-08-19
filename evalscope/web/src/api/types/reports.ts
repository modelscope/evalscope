// ------------------------------------------------------------------ //
// "reports" API domain types                                          //
// ------------------------------------------------------------------ //
//
// All endpoint response types are derived from runtime zod schemas. This barrel
// preserves the stable `@/api/types` import path while keeping the schemas as
// the single compile-time/runtime source of truth.

export type {
  // Report score tree
  PercentileStats,
  PerfMetrics,
  // Report payloads
  ReportData,
  LoadReportResponse,
  // Report list / summary
  ReportSummary,
  ListReportsResponse,
  // Prediction rows (chat messages + agent trace)
  ContentBlock,
  ToolCall,
  ChatMessage,
  AgentTraceEvent,
  AgentTrace,
  // LLM judge diagnostics carried on a score
  JudgeAttempt,
  JudgeSummary,
  PredictionScore,
  PredictionRow,
  PredictionsResponse,
  DeleteReportResponse,
  AnalysisResponse,
} from '@/api/schemas/reports.schema'
