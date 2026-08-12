// Benchmark catalog page.
//
// It loads the benchmark list once, normalises each entry's optional fields, and
// filters the grid by a debounced search. `listBenchmarks` is mocked so the page
// resolves a fixed catalog; these pin the load path, the search filter and the
// failure surface (a rejected load must not read as "no benchmarks match").

import { afterEach, describe, expect, it, vi } from 'vitest'
import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'

import { LocaleProvider } from '@/contexts/LocaleContext'
import type { BenchmarkEntry } from '@/api/types'

vi.mock('@/api/eval', async (importOriginal) => ({
  ...(await importOriginal<typeof import('@/api/eval')>()),
  listBenchmarks: vi.fn(),
}))

import * as evalApi from '@/api/eval'
import BenchmarksPage from './BenchmarksPage'

function entry(name: string, prettyName: string): BenchmarkEntry {
  return {
    name,
    pretty_name: prettyName,
    tags: [],
    category: 'llm',
    subset_list: [],
    total_samples: 10,
    few_shot_num: 0,
    dataset_id: name,
    paper_url: null,
    metrics: ['accuracy'],
    meta: {},
    description: {},
  }
}

const CATALOG = { text: [entry('gsm8k', 'GSM8K'), entry('arc', 'ARC Challenge')] }

async function settle(): Promise<void> {
  for (let i = 0; i < 8; i++) {
    await act(async () => { await Promise.resolve() })
  }
}

async function renderBenchmarks(): Promise<void> {
  render(
    <LocaleProvider>
      <MemoryRouter initialEntries={['/benchmarks']}>
        <BenchmarksPage />
      </MemoryRouter>
    </LocaleProvider>,
  )
  await settle()
}

afterEach(() => {
  cleanup()
  vi.clearAllMocks()
})

describe('BenchmarksPage', () => {
  it('lists every benchmark the catalog returns', async () => {
    vi.mocked(evalApi.listBenchmarks).mockResolvedValue(CATALOG)
    await renderBenchmarks()

    expect(screen.getByText('GSM8K')).toBeInTheDocument()
    expect(screen.getByText('ARC Challenge')).toBeInTheDocument()
  })

  it('filters the grid by the debounced search', async () => {
    vi.mocked(evalApi.listBenchmarks).mockResolvedValue(CATALOG)
    await renderBenchmarks()

    // The search is debounced by 300 ms (timers are faked globally by setup.ts).
    fireEvent.change(screen.getByPlaceholderText(/search/i), { target: { value: 'gsm' } })
    await act(async () => { await vi.advanceTimersByTimeAsync(350) })

    expect(screen.getByText('GSM8K')).toBeInTheDocument()
    expect(screen.queryByText('ARC Challenge')).not.toBeInTheDocument()
  })

  it('surfaces a load failure rather than an empty catalog', async () => {
    vi.mocked(evalApi.listBenchmarks).mockRejectedValue(new Error('catalog down'))
    await renderBenchmarks()

    expect(screen.getByText(/catalog down/i)).toBeInTheDocument()
  })
})
