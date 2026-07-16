import { test, expect } from './fixtures'
import { captureThemeBaselines } from './visual/capture'

const REPORT = {
  name: '20260701_090000@@test-model-a::gsm8k',
  model_name: 'test-model-a',
  dataset_name: 'gsm8k',
  score: 0.92,
  dataset_scores: { gsm8k: 0.92 },
  num_samples: 200,
  timestamp: '2026-07-01T09:00:00Z',
}

test('evaluation history remains visually stable in both themes', async ({ page }, testInfo) => {
  await page.route('**/api/v1/config', (route) => route.fulfill({ json: { outputs_root: './outputs' } }))
  await page.route('**/api/v1/reports/list**', (route) =>
    route.fulfill({
      json: {
        reports: [REPORT],
        total: 1,
        page: 1,
        page_size: 20,
        filters: { available_models: ['test-model-a'], available_datasets: ['gsm8k'] },
      },
    }),
  )

  await page.goto('/reports')
  if (testInfo.project.name === 'mobile-390') {
    await expect(page.getByRole('button', { name: /test-model-a/ }).first()).toBeVisible()
  } else {
    await expect(page.getByRole('cell', { name: 'test-model-a' }).first()).toBeVisible()
  }
  await captureThemeBaselines(page, 'evaluation-history', { fullPage: false })
})

test('task form primitives and tabs remain visually stable in both themes', async ({ page }) => {
  await page.route('**/api/v1/config', (route) => route.fulfill({ json: { outputs_root: './outputs' } }))
  await page.route('**/api/v1/eval/benchmarks**', (route) => route.fulfill({ json: { text: [] } }))

  await page.goto('/tasks?tab=eval')
  await expect(page.getByRole('button', { name: 'Start Evaluation' })).toBeVisible()
  await captureThemeBaselines(page, 'task-form', { fullPage: false })
})
