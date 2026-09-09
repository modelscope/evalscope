export type Locale = 'en' | 'zh';

export const pageSlugs = [
  'evaluation',
  'agent',
  'performance',
  'benchmarks',
  'visualization',
  'research',
  'get-started',
] as const;
export type PageSlug = (typeof pageSlugs)[number];
export type ClaimStatus = 'today' | 'exploring' | 'external';

export interface BenchmarkRecord {
  name: string;
  slug: string;
  category: string;
  metrics: string[];
  metricSummary: string;
  taskType: string;
  modalities: string;
  tags: string[];
}
