import { afterEach, describe, expect, it, vi } from 'vitest'
import { act, cleanup, render, screen } from '@testing-library/react'
import { useEffect } from 'react'

import type { LoadReportResponse, ReportData } from '@/api/types'

vi.mock('@/api/client', () => ({
  apiValidated: vi.fn().mockResolvedValue({ outputs_root: '' }),
}))

vi.mock('@/api/reports', async (importOriginal) => ({
  ...(await importOriginal<typeof import('@/api/reports')>()),
  loadReport: vi.fn(),
}))

import * as reportsApi from '@/api/reports'
import { ReportsProvider, useReportCache, useScan } from './ReportsContext'

interface Deferred<T> {
  promise: Promise<T>
  resolve: (value: T) => void
}

function deferred<T>(): Deferred<T> {
  let resolve!: (value: T) => void
  const promise = new Promise<T>((done) => { resolve = done })
  return { promise, resolve }
}

function response(): LoadReportResponse {
  return { report_list: [], datasets: [], task_config: {} }
}

let controls: {
  triggerScan: (path?: string) => void
  loadMultiReports: (names: string[], signal?: AbortSignal) => Promise<ReportData[]>
}

function Probe() {
  const scan = useScan()
  const cache = useReportCache()
  useEffect(() => {
    controls = { triggerScan: scan.triggerScan, loadMultiReports: cache.loadMultiReports }
  }, [cache.loadMultiReports, scan.triggerScan])
  return (
    <div>
      <span data-testid="root">{scan.rootPath}</span>
      <span data-testid="cache">{Object.keys(cache.reportCache).join(',')}</span>
    </div>
  )
}

async function flush(): Promise<void> {
  await act(async () => { await Promise.resolve() })
}

afterEach(() => {
  cleanup()
  vi.clearAllMocks()
})

describe('ReportsProvider cache scope', () => {
  it('ignores an old-scope response that resolves after the new scope', async () => {
    const oldResponse = deferred<LoadReportResponse>()
    const newResponse = deferred<LoadReportResponse>()
    vi.mocked(reportsApi.loadReport).mockImplementation((rootPath) => {
      return rootPath === 'old-root' ? oldResponse.promise : newResponse.promise
    })

    render(<ReportsProvider><Probe /></ReportsProvider>)
    await flush()

    act(() => controls.triggerScan('old-root'))
    await flush()
    let oldRequest!: Promise<ReportData[]>
    act(() => { oldRequest = controls.loadMultiReports(['old-report']) })

    act(() => controls.triggerScan('new-root'))
    await flush()
    let newRequest!: Promise<ReportData[]>
    act(() => { newRequest = controls.loadMultiReports(['new-report']) })

    await act(async () => {
      newResponse.resolve(response())
      await newRequest
    })
    expect(screen.getByTestId('root')).toHaveTextContent('new-root')
    expect(screen.getByTestId('cache')).toHaveTextContent('new-report')

    await act(async () => {
      oldResponse.resolve(response())
      await oldRequest
    })
    expect(screen.getByTestId('cache')).toHaveTextContent('new-report')
  })
})
