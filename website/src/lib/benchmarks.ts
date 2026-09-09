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

function getCategory(name: string, meta: Record<string, unknown>): string {
  const declared = text(meta.category).toLowerCase();
  if (declared === 'llm') return 'LLM';
  if (declared === 'vlm') return 'VLM';
  if (declared === 'agent') return 'Agent';
  if (declared === 'aigc') return 'AIGC';
  const haystack = `${name} ${text(meta.task_type)} ${text(meta.modalities)}`.toLowerCase();
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
    const rawMetrics = meta.metrics;
    const metrics = Array.isArray(rawMetrics)
      ? rawMetrics.map(text).filter(Boolean)
      : [text(rawMetrics)].filter(Boolean);
    const declaredTags = Array.isArray(meta.tags) ? meta.tags.map(text).filter(Boolean) : [];
    const haystack =
      `${slug} ${name} ${text(meta.task_type)} ${text(meta.modalities)} ${declaredTags.join(' ')}`.toLowerCase();
    const tags = Object.entries(knownTags).flatMap(([needle, tags]) => (haystack.includes(needle) ? tags : []));
    return {
      name,
      slug,
      category: getCategory(name, meta),
      metrics,
      taskType: text(meta.task_type) || 'Evaluation',
      modalities: text(meta.modalities) || 'Text',
      tags: [...new Set([...declaredTags, ...tags])],
    };
  });
}
