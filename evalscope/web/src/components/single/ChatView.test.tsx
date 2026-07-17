import { cleanup, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'

import type { PredictionRow, PredictionsResponse } from '@/api/types'
import { LocaleProvider } from '@/contexts/LocaleContext'
import { loadFixture } from '@/test/loadFixture'
import ChatView from './ChatView'

const tracedPrediction = loadFixture<PredictionsResponse>('predictions-tool-trace').predictions[0]

function renderChatView(prediction: PredictionRow) {
  return render(
    <LocaleProvider>
      <ChatView prediction={prediction} />
    </LocaleProvider>,
  )
}

afterEach(cleanup)

describe('ChatView rendering modes', () => {
  it('keeps agent traces in the step-by-step chat timeline', () => {
    renderChatView(tracedPrediction)

    expect(screen.getAllByText('Let me calculate the total cost and the change.').length).toBeGreaterThan(0)
    expect(screen.getAllByRole('button', { name: /calculator/ }).length).toBeGreaterThan(0)
    expect(screen.getByText('The customer receives 29 dollars in change.')).toBeInTheDocument()
  })

  it('renders structured messages as distinct chat rows without an agent trace', () => {
    renderChatView({ ...tracedPrediction, AgentTrace: null })

    expect(screen.getByText('You are a helpful math assistant. Use the calculator tool when arithmetic is required.')).toBeInTheDocument()
    expect(screen.getByText('A store sells notebooks at 3 dollars each. If a customer buys 7 notebooks and pays with a 50 dollar bill, how much change do they receive?')).toBeInTheDocument()
    expect(screen.getByText('The customer receives 29 dollars in change.')).toBeInTheDocument()
  })

  it('retains the legacy input/generated fallback when messages are absent', () => {
    const legacyPrediction: PredictionRow = {
      ...tracedPrediction,
      Input: 'Legacy user prompt',
      Generated: 'Legacy model response',
      Messages: null,
      AgentTrace: null,
    }

    renderChatView(legacyPrediction)

    expect(screen.getByText('Legacy user prompt')).toBeInTheDocument()
    expect(screen.getByText('Legacy model response')).toBeInTheDocument()
  })
})
