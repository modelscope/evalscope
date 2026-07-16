import { test, expect } from './fixtures'

const REPORTS = [
  {
    name: '20260701_090000@@test-model-a::gsm8k',
    model_name: 'test-model-a',
    dataset_name: 'gsm8k',
    score: 0.92,
    dataset_scores: { gsm8k: 0.92 },
    num_samples: 200,
    timestamp: '2026-07-01T09:00:00Z',
  },
  {
    name: '20260701_080000@@test-model-b::gsm8k',
    model_name: 'test-model-b',
    dataset_name: 'gsm8k',
    score: 0.88,
    dataset_scores: { gsm8k: 0.88 },
    num_samples: 200,
    timestamp: '2026-07-01T08:00:00Z',
  },
]

function reportDetail(model: string, score: number) {
  return {
    report_list: [
      {
        name: `${model}_gsm8k`,
        dataset_name: 'gsm8k',
        model_name: model,
        score,
        analysis: `${model} deterministic fixture`,
        metrics: [
          {
            name: 'AverageAccuracy',
            num: 200,
            score,
            categories: [
              {
                name: ['default'],
                num: 200,
                score,
                subsets: [{ name: 'main', score, num: 200 }],
              },
            ],
          },
        ],
      },
    ],
    datasets: ['gsm8k'],
    task_config: { model, datasets: ['gsm8k'], api_key: 'sk-***REDACTED***' },
  }
}

const PERF_RUN = {
  path: 'perf/test-model-a/20260701_090000',
  model: 'test-model-a',
  api_type: 'openai_api',
  dataset: 'openqa',
  num_runs: 1,
  total_requests: 100,
  success_rate: 100,
  best_rps: 5.43,
  best_latency: 1.842,
  is_embedding: false,
  has_html: true,
  timestamp: '2026-07-01T09:00:00Z',
  api_host: 'dashscope.aliyuncs.com',
  concurrency: [10],
}

const PERF_DETAIL = {
  path: PERF_RUN.path,
  model: PERF_RUN.model,
  api_type: PERF_RUN.api_type,
  dataset: PERF_RUN.dataset,
  generated_at: PERF_RUN.timestamp,
  basic_info: {
    Provider: 'Custom',
    Protocol: 'OpenAI-compatible',
    'Total requests': '100',
  },
  summary_columns: ['Metric', 'Value'],
  summary_rows: [
    ['Concurrency', 10],
    ['Number of requests', 100],
    ['Success rate', 1],
    ['Average latency (s)', 1.842],
    ['Request throughput (req/s)', 5.43],
  ],
  best_config: { Concurrency: '10', 'Request rate': 'closed-loop' },
  recommendations: ['Deterministic single-run fixture.'],
  num_runs: 1,
  is_embedding: false,
  has_html: true,
}

const PERF_RUN_DETAILS = {
  runs: [
    {
      dir_name: 'parallel_10',
      name: 'Concurrency 10',
      parallel: 10,
      number: 100,
      rate: null,
      total_requests: 100,
      succeed_requests: 100,
      success_rate: 1,
      num_requests: 100,
      has_requests: true,
      percentile_columns: ['Percentile', 'Latency (s)'],
      percentile_rows: [['50%', 1.842], ['99%', 3.401]],
    },
  ],
  total: 1,
}

const PERF_WIDE_COLUMNS = [
  'Conc.', 'Rate', 'RPS', 'Avg Lat.(s)', 'P99 Lat.(s)', 'Avg TTFT(ms)',
  'P99 TTFT(ms)', 'Avg TPOT(ms)', 'P99 TPOT(ms)', 'Gen. tok/s', 'Success Rate',
]

function perfWideDetail(
  path: string,
  generatedAt: string,
  requests: number,
  row: (string | number)[],
) {
  return {
    path,
    model: 'qwen-plus',
    api_type: 'openai',
    dataset: 'openqa',
    generated_at: generatedAt,
    basic_info: { 'Total Requests': String(requests), 'API Host': 'dashscope.aliyuncs.com' },
    summary_columns: PERF_WIDE_COLUMNS,
    summary_rows: [row],
    best_config: {},
    recommendations: [],
    num_runs: 1,
    is_embedding: false,
    has_html: true,
  }
}

const PERF_P1 = perfWideDetail(
  'run-p1', '2026-07-15T15:37:19', 4,
  ['1', 'INF', '1.0326', '0.968', '1.080', '467.79', '597.30', '16.13', '17.10', '33.04', '100.0%'],
)
const PERF_P2 = perfWideDetail(
  'run-p2', '2026-07-15T15:39:04', 6,
  ['2', 'INF', '1.9256', '1.027', '1.210', '493.17', '656.15', '17.21', '17.77', '61.62', '100.0%'],
)

async function mockCommonApi(page: import('@playwright/test').Page) {
  await page.route('**/api/v1/config', (route) => route.fulfill({ json: { outputs_root: './outputs' } }))
}

test('dashboard avoids duplicate quick links while retaining KPI navigation', async ({ page }) => {
  await mockCommonApi(page)
  await page.route('**/api/v1/reports/list**', (route) =>
    route.fulfill({
      json: {
        reports: REPORTS,
        total: REPORTS.length,
        page: 1,
        page_size: 1000,
        filters: {
          available_models: REPORTS.map((report) => report.model_name),
          available_datasets: ['gsm8k'],
        },
      },
    }),
  )
  await page.route('**/api/v1/perf/list**', (route) => route.fulfill({ json: { runs: [PERF_RUN], total: 1 } }))

  await page.goto('/dashboard')

  await expect(page.getByText('Total Evaluations')).toBeVisible()
  await expect(page.getByText('Recent Runs')).toBeVisible()
  await expect(page.getByText('Run a new model evaluation task')).toHaveCount(0)
  await expect(page.getByText('Benchmark inference throughput & latency')).toHaveCount(0)
  await expect(page.getByText('Explore loaded evaluation results')).toHaveCount(0)
})

test('evaluation history exposes complete metadata and responsive touch targets', async ({ page }, testInfo) => {
  await mockCommonApi(page)
  await page.route('**/api/v1/reports/list**', (route) =>
    route.fulfill({
      json: {
        reports: REPORTS,
        total: REPORTS.length,
        page: 1,
        page_size: 20,
        filters: {
          available_models: REPORTS.map((report) => report.model_name),
          available_datasets: ['gsm8k'],
        },
      },
    }),
  )

  await page.goto('/reports')

  if (testInfo.project.name === 'mobile-390') {
    const reportButton = page.getByRole('button', { name: /test-model-a/ }).first()
    await expect(reportButton).toBeVisible()
    await expect(reportButton).toContainText('gsm8k')
    await expect(reportButton).toContainText('200')
    await expect(reportButton).toContainText('92.0%')
    const widths = await page.evaluate(() => ({
      scroll: document.documentElement.scrollWidth,
      client: document.documentElement.clientWidth,
    }))
    expect(widths.scroll).toBeLessThanOrEqual(widths.client)
    const detailButton = page.getByRole('button', { name: 'View report detail' }).first()
    const box = await detailButton.boundingBox()
    expect(box?.width ?? 0).toBeGreaterThanOrEqual(44)
    expect(box?.height ?? 0).toBeGreaterThanOrEqual(44)
  } else {
    const row = page.getByRole('row').filter({ has: page.getByRole('cell', { name: 'test-model-a' }) })
    await expect(row).toBeVisible()
    await expect(row).toContainText('gsm8k')
    await expect(row).toContainText('200')
    await expect(row).toContainText('92.0%')
    await expect(page.getByRole('columnheader', { name: 'Model' })).toBeVisible()
    await expect(page.getByRole('columnheader', { name: 'Status' })).toBeVisible()
  }
})

test('report prediction detail keeps structured messages in the chat view', async ({ page }) => {
  await mockCommonApi(page)
  await page.route('**/api/v1/reports/load**', (route) =>
    route.fulfill({ json: reportDetail('test-model-a', 0.92) }),
  )
  await page.route('**/api/v1/reports/chart**', (route) =>
    route.fulfill({ status: 503, body: 'fixture unavailable' }),
  )
  await page.route('**/api/v1/reports/dataframe**', (route) =>
    route.fulfill({
      json: {
        columns: ['Cat.', 'Subset'],
        data: [{ 'Cat.': 'default', Subset: 'main' }],
      },
    }),
  )
  await page.route('**/api/v1/reports/predictions**', (route) =>
    route.fulfill({
      json: {
        predictions: [
          {
            Index: '0',
            Input: 'Sort the inbox.',
            Metadata: {},
            Generated: 'I sorted the inbox.',
            Gold: 'I sorted the inbox.',
            Pred: 'I sorted the inbox.',
            Score: { AverageAccuracy: 1 },
            NScore: 1,
            Messages: [
              { id: 'user-1', role: 'user', content: 'Sort the inbox.' },
              { id: 'assistant-1', role: 'assistant', content: 'I sorted the inbox.', model: 'test-model-a' },
            ],
          },
        ],
      },
    }),
  )

  await page.goto(`/reports/${encodeURIComponent(REPORTS[0].name)}?root_path=${encodeURIComponent('./outputs')}`)
  await expect(page.getByRole('img', { name: 'Single metric value' })).toBeVisible()
  await expect(page.locator('iframe')).toHaveCount(0)
  await page.getByRole('tab', { name: 'Predictions' }).click()

  await expect(page.getByText('Sort the inbox.')).toBeVisible()
  await expect(page.getByText('I sorted the inbox.').first()).toBeVisible()
  await expect(page.getByText('User', { exact: true })).toBeVisible()
  await expect(page.getByText('Assistant', { exact: true })).toBeVisible()
  await expect(page.getByText('Summary', { exact: true })).toHaveCount(0)
})

test('compare loads two runs and keeps an authoritative score table when charts fail', async ({ page }) => {
  await mockCommonApi(page)
  await page.route('**/api/v1/reports/load**', (route) => {
    const reportName = new URL(route.request().url()).searchParams.get('report_name') ?? ''
    const isModelB = reportName.includes('test-model-b')
    return route.fulfill({ json: reportDetail(isModelB ? 'test-model-b' : 'test-model-a', isModelB ? 0.88 : 0.92) })
  })
  await page.route('**/api/v1/reports/chart**', (route) => route.fulfill({ status: 503, body: 'fixture unavailable' }))
  await page.route('**/api/v1/reports/predictions**', (route) => {
    const reportName = new URL(route.request().url()).searchParams.get('report_name') ?? ''
    const score = reportName.includes('test-model-b') ? 0.88 : 0.92
    return route.fulfill({
      json: {
        predictions: [
          {
            Index: '0',
            Input: 'What is 1 + 1?',
            Metadata: {},
            Generated: '2',
            Gold: '2',
            Pred: '2',
            Score: { AverageAccuracy: score },
            NScore: score,
          },
        ],
      },
    })
  })

  const reports = REPORTS.map((report) => report.name).join(';')
  await page.goto(`/compare?reports=${encodeURIComponent(reports)}&root_path=${encodeURIComponent('./outputs')}`)

  await expect(page.getByRole('tab', { name: 'Score Comparison' })).toHaveAttribute('aria-selected', 'true')
  await expect(page.getByText('test-model-a').first()).toBeVisible()
  await expect(page.getByText('test-model-b').first()).toBeVisible()
  await expect(page.getByRole('table').first()).toContainText('gsm8k')

  await page.getByRole('tab', { name: 'Prediction Comparison' }).click()
  await expect(page.getByRole('spinbutton', { name: 'Score Threshold' })).toHaveAttribute(
    'name',
    'compare-score-threshold',
  )
  await expect(page.getByRole('button', { name: 'All Above' })).toBeVisible()
  await expect(page.getByRole('button', { name: 'All Below' })).toBeVisible()
  await expect(page.getByRole('button', { name: 'Previous sample' })).toBeVisible()
  await expect(page.getByRole('button', { name: 'Next sample' })).toBeVisible()
  await expect(page.getByText('compare.allAbove')).toHaveCount(0)
  await expect(page.getByText('compare.allBelow')).toHaveCount(0)
})

test('compare surfaces invalid report responses and offers a retry', async ({ page }) => {
  await mockCommonApi(page)
  await page.route('**/api/v1/reports/load**', (route) =>
    route.fulfill({ json: { report_list: null, datasets: [], task_config: {} } }),
  )

  const reports = REPORTS.map((report) => report.name).join(';')
  await page.goto(`/compare?reports=${encodeURIComponent(reports)}&root_path=${encodeURIComponent('./outputs')}`)

  await expect(page.getByRole('alert').filter({ hasText: 'Response validation failed' })).toBeVisible()
  await expect(page.getByRole('button', { name: 'Retry' }).first()).toBeVisible()
})

test('performance archive opens a real run detail with provider and workload context', async ({ page }) => {
  await mockCommonApi(page)
  await page.route('**/api/v1/perf/list**', (route) => route.fulfill({ json: { runs: [PERF_RUN], total: 1 } }))
  await page.route('**/api/v1/perf/detail**', (route) => route.fulfill({ json: PERF_DETAIL }))
  await page.route('**/api/v1/perf/runs**', (route) => route.fulfill({ json: PERF_RUN_DETAILS }))
  await page.route('**/api/v1/perf/requests**', (route) =>
    route.fulfill({ json: { columns: [], rows: [], total: 0, page: 1, page_size: 50, has_db: true } }),
  )
  await page.route('**/api/v1/perf/chart**', (route) => route.fulfill({ status: 503, body: 'fixture unavailable' }))

  await page.goto('/performance')
  await page.getByText('test-model-a').first().click()
  await expect(page).toHaveURL(/\/perf-report\?/)
  await expect(page.getByText('Provider')).toBeVisible()
  await expect(page.getByText('Custom')).toBeVisible()
  await expect(page.getByText('OpenAI-compatible')).toBeVisible()
  await expect(page.getByText('Concurrency').first()).toBeVisible()
  await expect(page.getByText('closed-loop')).toBeVisible()
})

test('performance compare reconciles the real archive wide-table response', async ({ page }) => {
  await mockCommonApi(page)
  await page.route('**/api/v1/perf/detail**', (route) => {
    const path = new URL(route.request().url()).searchParams.get('path')
    return route.fulfill({ json: path === PERF_P1.path ? PERF_P1 : PERF_P2 })
  })
  await page.route('**/api/v1/perf/compare/chart**', (route) =>
    route.fulfill({ status: 503, body: 'fixture unavailable' }),
  )

  await page.goto(`/perf-compare?paths=${encodeURIComponent('run-p2;run-p1')}&root_path=${encodeURIComponent('./outputs')}`)

  await expect(page.getByText('Samples: 4')).toBeVisible()
  await expect(page.getByText('Samples: 6')).toBeVisible()
  await expect(page.getByTestId('delta-row-rps')).toContainText('1.03 req/s')
  await expect(page.getByTestId('delta-row-rps')).toContainText('1.93 req/s')
  await expect(page.getByTestId('delta-row-rps')).toContainText('86.48%')
  await expect(page.getByTestId('delta-row-rps')).toContainText('Improvement')
  await expect(page.getByTestId('delta-row-latency')).toContainText('Regression')
  await expect(page.getByTestId('missing-perf-data')).toHaveCount(0)
})

test('evaluation task form validates and submits a redacted deterministic payload', async ({ page }) => {
  await mockCommonApi(page)
  await page.route('**/api/v1/eval/benchmarks**', (route) =>
    route.fulfill({
      json: {
        text: [
          {
            name: 'gsm8k',
            pretty_name: 'GSM8K',
            tags: ['reasoning'],
            category: 'llm',
            subset_list: ['main'],
            total_samples: 100,
            few_shot_num: 0,
            dataset_id: 'fixture/gsm8k',
            paper_url: null,
            metrics: ['accuracy'],
            meta: {},
            description: { en: { full: 'Fixture benchmark', sections: {} } },
          },
        ],
      },
    }),
  )
  let submitted: Record<string, unknown> | null = null
  await page.route('**/api/v1/eval/invoke', async (route) => {
    submitted = route.request().postDataJSON() as Record<string, unknown>
    await route.fulfill({ json: { status: 'ok', task_id: 'eval_fixture' } })
  })
  await page.route('**/api/v1/eval/progress**', (route) =>
    route.fulfill({ json: { percent: 100, current_step: 'complete' } }),
  )
  await page.route('**/api/v1/eval/log**', (route) =>
    route.fulfill({ json: { text: '', head_line: 0, tail_line: 0, total_lines: 0 } }),
  )

  await page.goto('/tasks?tab=eval')
  await page.getByLabel('Model Name').fill('test-model-a')
  await page.getByLabel('Datasets').fill('gsm')
  await page.getByRole('option', { name: 'gsm8k' }).click()
  await page.getByRole('button', { name: 'Start Evaluation' }).click()

  await expect(page.getByText('Completed')).toBeVisible()
  expect(submitted).toMatchObject({ model: 'test-model-a', datasets: ['gsm8k'] })
  expect(JSON.stringify(submitted)).not.toContain('sk-')
})
