// ------------------------------------------------------------------ //
// "reports" API domain types                                          //
// ------------------------------------------------------------------ //
//
// All endpoint response types are derived from runtime zod schemas. This barrel
// preserves the stable `@/api/types` import path while keeping the schemas as
// the single compile-time/runtime source of truth (Req 13.1, 15.3).

export type {
  // Report score tree
  SubsetData,
  CategoryData,
  MetricData,
  // Performance metrics embedded on a report
  PercentileStats,
  PerfMetricsSummary,
  PerfMetrics,
  // Report payloads
  ReportData,
  LoadReportResponse,
  // Report list / summary
  ReportSummary,
  ListReportsResponse,
  // Prediction rows (chat messages + agent trace)
  SamplePerfMetrics,
  ContentBlock,
  ToolCall,
  ToolMessageError,
  ChatMessage,
  AgentTraceEventType,
  AgentTraceEvent,
  AgentTrace,
  PredictionRow,
  PredictionsResponse,
  ScanResponse,
  AnalysisResponse,
} from '@/api/schemas/reports.schema'
