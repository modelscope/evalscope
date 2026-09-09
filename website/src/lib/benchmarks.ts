import fs from 'node:fs';
import path from 'node:path';

import type { BenchmarkRecord } from '../types';

const knownTags: Record<string, string[]> = {
  reasoning: ['Reasoning'],
  math: ['Math', 'Reasoning'],
  code: ['Coding'],
  agent: ['FunctionCalling'],
  vlm: ['MultiModal'],
  visual: ['MultiModal'],
  image: ['MultiModal'],
  long: ['LongContext'],
};

function text(value: unknown): string {
  return typeof value === 'string' ? value : '';
}

function fieldFromDescription(description: string, field: string): string {
  const match = description.match(new RegExp(`^-\\s*(?:\\*\\*)?${field}(?:\\*\\*)?:\\s*(.+)$`, 'im'));
  return match?.[1]?.trim() ?? '';
}

function metricNames(value: unknown): string[] {
  if (typeof value === 'string') return [value];
  if (Array.isArray(value)) return value.flatMap(metricNames);
  if (value && typeof value === 'object') return Object.keys(value as Record<string, unknown>);
  return [];
}

const metricLabels: Record<string, string> = {
  accuracy: 'Answer accuracy',
  exact_match: 'Exact answer match',
  pass_rate: 'Task pass rate',
  success_rate: 'Run success rate',
  f1: 'F1 score',
  bleu: 'BLEU score',
  rouge_l: 'ROUGE-L score',
  process_acc: 'Process accuracy',
};

function describeMetrics(metrics: string[], tags: string[]): string {
  const readable = metrics
    .map((metric) => metricLabels[metric.toLowerCase()] ?? metric.replaceAll('_', ' '))
    .filter(Boolean);
  if (readable.length) return readable.slice(0, 2).join(' · ');
  if (tags.includes('Math')) return 'Exact answer match';
  if (tags.includes('Coding')) return 'Test pass rate';
  if (tags.includes('FunctionCalling')) return 'Tool-call success';
  if (tags.includes('MultiModal')) return 'Visual answer accuracy';
  return 'Reported evaluation metric';
}

function getCategory(name: string, meta: Record<string, unknown>): string {
  const declared = text(meta.category).toLowerCase();
  if (declared === 'llm') return 'LLM';
  if (declared === 'vlm') return 'VLM';
  if (declared === 'agent') return 'Agent';
  if (declared === 'aigc') return 'AIGC';
  const haystack = `${name} ${text(meta.task_type)} ${text(meta.modalities)} ${text(meta.description)}`.toLowerCase();
  if (haystack.includes('agent') || haystack.includes('tool') || haystack.includes('function')) return 'Agent';
  if (
    haystack.includes('image') ||
    haystack.includes('vision') ||
    haystack.includes('video') ||
    haystack.includes('multimodal')
  )
    return 'VLM';
  if (haystack.includes('generation') || haystack.includes('aigc') || haystack.includes('edit')) return 'AIGC';
  return 'LLM';
}

export function getBenchmarkCatalog(): BenchmarkRecord[] {
  const sourceRoot = process.env.EVALSCOPE_SOURCE_ROOT ?? path.resolve(process.cwd(), '..');
  const metaDir = path.join(sourceRoot, 'evalscope', 'benchmarks', '_meta');
  const names = fs
    .readdirSync(metaDir)
    .filter((file) => file.endsWith('.json'))
    .sort();
  return names.map((file) => {
    const raw = JSON.parse(fs.readFileSync(path.join(metaDir, file), 'utf8')) as { meta?: Record<string, unknown> };
    const meta = raw.meta ?? {};
    const slug = file.replace(/\.json$/, '');
    const name = text(meta.pretty_name) || text(meta.name) || slug;
    const metrics = metricNames(meta.metrics);
    const declaredTags = Array.isArray(meta.tags) ? meta.tags.map(text).filter(Boolean) : [];
    const haystack =
      `${slug} ${name} ${text(meta.task_type)} ${text(meta.modalities)} ${text(meta.description)} ${declaredTags.join(' ')}`.toLowerCase();
    const tags = Object.entries(knownTags).flatMap(([needle, tags]) => (haystack.includes(needle) ? tags : []));
    const allTags = [...new Set([...declaredTags, ...tags])];
    const description = text(meta.description);
    return {
      name,
      slug,
      category: getCategory(name, meta),
      metrics,
      metricSummary: describeMetrics(metrics, allTags),
      taskType: text(meta.task_type) || fieldFromDescription(description, 'Task Type') || 'General evaluation',
      modalities: text(meta.modalities) || 'Text',
      tags: allTags,
    };
  });
}
