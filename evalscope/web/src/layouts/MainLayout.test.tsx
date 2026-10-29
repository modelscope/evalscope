import { afterEach, describe, expect, it, vi } from 'vitest'
import { act, cleanup, render, screen } from '@testing-library/react'
import { MemoryRouter, Route, Routes } from 'react-router-dom'

vi.mock('@/api/client', () => ({
  apiValidated: vi.fn().mockResolvedValue({ outputs_root: '', version: '1.2.3' }),
}))

import { LocaleProvider } from '@/contexts/LocaleContext'
import { ReportsProvider } from '@/contexts/ReportsContext'
import MainLayout from './MainLayout'

afterEach(() => {
  cleanup()
  vi.clearAllMocks()
})

describe('MainLayout', () => {
  it('shows the backend EvalScope version', async () => {
    vi.stubGlobal('scrollTo', vi.fn())

    render(
      <LocaleProvider>
        <ReportsProvider>
          <MemoryRouter initialEntries={['/dashboard']}>
            <Routes>
              <Route element={<MainLayout />}>
                <Route path="/dashboard" element={null} />
              </Route>
            </Routes>
          </MemoryRouter>
        </ReportsProvider>
      </LocaleProvider>,
    )

    await act(async () => { await Promise.resolve() })

    expect(screen.getByTitle('EvalScope version 1.2.3')).toHaveTextContent('v1.2.3')
  })
})
