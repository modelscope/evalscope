import type { Locale, PageSlug } from '../types';

export const productLinks: { slug: PageSlug; en: string; zh: string }[] = [
  { slug: 'evaluation', en: 'Evaluation', zh: '模型评测' },
  { slug: 'agent', en: 'Agent & Harness', zh: 'Agent 与 Harness' },
  { slug: 'performance', en: 'Performance', zh: '性能压测' },
  { slug: 'visualization', en: 'Visualization', zh: '可视化' },
];

export const copy = {
  en: {
    product: 'Product',
    benchmarks: 'Benchmarks',
    research: 'Research',
    docs: 'Docs',
    github: 'GitHub',
    ai: 'Use with AI',
    home: 'Home',
    getStarted: 'Get started',
    allRights: 'Open source evaluation for the AI systems you ship.',
  },
  zh: {
    product: '产品',
    benchmarks: '评测集',
    research: '研究',
    docs: '文档',
    github: 'GitHub',
    ai: '让 AI 帮你用',
    home: '首页',
    getStarted: '开始使用',
    allRights: '为你交付的 AI 系统提供开源评测。',
  },
} satisfies Record<Locale, Record<string, string>>;

export function localizePath(path: '/' | `/${PageSlug}`, locale: Locale): string {
  const localized = locale === 'en' ? path : path === '/' ? '/zh/' : `/zh${path}`;
  const base = import.meta.env.BASE_URL.replace(/\/$/, '');
  return localized === '/' ? `${base}/` : `${base}${localized}`;
}

export function otherLocalePath(path: '/' | `/${PageSlug}`, locale: Locale): string {
  return localizePath(path, locale === 'en' ? 'zh' : 'en');
}

export function docsUrl(locale: Locale): string {
  return locale === 'zh'
    ? 'https://evalscope.readthedocs.io/zh-cn/latest/'
    : 'https://evalscope.readthedocs.io/en/latest/';
}
