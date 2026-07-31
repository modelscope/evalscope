import { cleanup, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'

import type { PredictionRow, PredictionsResponse } from '@/api/types'
import { LocaleProvider } from '@/contexts/LocaleContext'
import { loadFixture } from '@/test/loadFixture'
import ChatView from './ChatView'

const tracedPrediction = loadFixture<PredictionsResponse>('predictions-tool-trace').predictions[0]
const miniwobPrediction: PredictionRow = {
  ...tracedPrediction,
  Messages: [
    {
      id: 'browser-observation-0',
      role: 'user',
      content: [
        { type: 'text', text: 'Goal: Click OK\nStep: 0\nAccessibility tree:\n[1] button "OK"' },
        { type: 'image', image: '/tmp/miniwob/step-000.png' },
      ],
    },
    {
      id: 'model-response-0',
      role: 'assistant',
      content: 'I will click the OK button.',
      tool_calls: [
        {
          id: 'browser-call-0',
          function: 'browser_action',
          arguments: { action: 'click("1")' },
        },
      ],
    },
    {
      id: 'browser-tool-result-0',
      role: 'tool',
      content: 'Step: 1\nReward: 1.0\nDone: True',
      tool_call_id: 'browser-call-0',
      function: 'browser_action',
    },
    {
      id: 'browser-observation-1',
      role: 'user',
      content: [
        { type: 'image', image: '/tmp/miniwob/step-001.png' },
      ],
      tool_call_id: ['browser-call-0'],
    },
  ],
  AgentTrace: {
    strategy: 'function_calling',
    environment: 'browsergym',
    max_steps: 10,
    events: [
      {
        step: 0,
        timestamp: 1_700_000_000,
        type: 'env_reset',
        message_id: 'browser-observation-0',
        latency_ms: 120,
        payload: {
          backend: 'browsergym',
          reward: 0,
          done: false,
          screenshot_path: '/tmp/miniwob/step-000.png',
        },
      },
      {
        step: 0,
        timestamp: 1_700_000_001,
        type: 'model_generate',
        message_id: 'model-response-0',
        latency_ms: 500,
        payload: {},
      },
      {
        step: 0,
        timestamp: 1_700_000_002,
        type: 'tool_call',
        message_id: 'model-response-0',
        payload: {
          name: 'browser_action',
          arguments: { action: 'click("1")' },
          id: 'browser-call-0',
        },
      },
      {
        step: 0,
        timestamp: 1_700_000_003,
        type: 'tool_result',
        message_id: 'browser-tool-result-0',
        payload: {
          name: 'browser_action',
          id: 'browser-call-0',
          attachments: ['/tmp/miniwob/step-001.png'],
        },
      },
    ],
  },
}

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

  it('renders BrowserGym reset observations and screenshots in the traced timeline', () => {
    const { container } = renderChatView(miniwobPrediction)

    expect(screen.getByText('Environment Reset')).toBeInTheDocument()
    const observation = screen.getByText(/Goal: Click OK/)
    const modelResponse = screen.getByText('I will click the OK button.')
    expect(observation.compareDocumentPosition(modelResponse) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy()
    expect(screen.getAllByText('Environment Observation')).toHaveLength(2)
    expect(screen.queryByText('User')).not.toBeInTheDocument()
    const images = container.querySelectorAll('img')
    expect(images).toHaveLength(2)
    expect(images[0]).toHaveAttribute(
      'src',
      '/api/v1/reports/media/file?path=%2Ftmp%2Fminiwob%2Fstep-000.png',
    )
    expect(images[1]).toHaveAttribute(
      'src',
      '/api/v1/reports/media/file?path=%2Ftmp%2Fminiwob%2Fstep-001.png',
    )
    for (const label of screen.getAllByText('Environment Observation')) {
      expect(label.closest('[style*="--bubble-environment-bg"]')).not.toBeNull()
    }
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
