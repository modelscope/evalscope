// ------------------------------------------------------------------ //
// "reports" API domain types                                          //
// ------------------------------------------------------------------ //
//
// Public report API types are generated from backend Pydantic response models.
// This barrel preserves the stable `@/api/types` import path.

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
  PredictionToolCall as ToolCall,
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
} from '@/api/generated/contracts'
