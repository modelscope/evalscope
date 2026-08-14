import { afterEach, describe, expect, it } from 'vitest'
import { cleanup, fireEvent, render, screen } from '@testing-library/react'

import Tooltip from './Tooltip'

afterEach(cleanup)

function renderTooltip() {
  return render(
    <Tooltip content="Filters rows above / below the threshold." label="Threshold help">
      <span>icon</span>
    </Tooltip>,
  )
}

describe('Tooltip', () => {
  it('is hidden until the trigger is interacted with', () => {
    renderTooltip()
    expect(screen.queryByRole('tooltip')).not.toBeInTheDocument()
  })

  it('shows on hover and hides on leave, with no artificial delay', () => {
    renderTooltip()
    const trigger = screen.getByLabelText('Threshold help')

    fireEvent.mouseEnter(trigger)
    expect(screen.getByRole('tooltip')).toHaveTextContent('Filters rows above / below the threshold.')

    fireEvent.mouseLeave(trigger)
    expect(screen.queryByRole('tooltip')).not.toBeInTheDocument()
  })

  it('shows on keyboard focus and hides on blur', () => {
    renderTooltip()
    const trigger = screen.getByLabelText('Threshold help')

    fireEvent.focus(trigger)
    expect(screen.getByRole('tooltip')).toBeInTheDocument()

    fireEvent.blur(trigger)
    expect(screen.queryByRole('tooltip')).not.toBeInTheDocument()
  })

  it('dismisses on Escape while focused', () => {
    renderTooltip()
    const trigger = screen.getByLabelText('Threshold help')

    fireEvent.focus(trigger)
    expect(screen.getByRole('tooltip')).toBeInTheDocument()

    fireEvent.keyDown(trigger, { key: 'Escape' })
    expect(screen.queryByRole('tooltip')).not.toBeInTheDocument()
  })

  it('links the trigger to the bubble via aria-describedby only while open', () => {
    renderTooltip()
    const trigger = screen.getByLabelText('Threshold help')
    expect(trigger).not.toHaveAttribute('aria-describedby')

    fireEvent.focus(trigger)
    const tooltip = screen.getByRole('tooltip')
    expect(trigger.getAttribute('aria-describedby')).toBe(tooltip.id)
  })

  it('keeps two instances independent', () => {
    render(
      <>
        <Tooltip content="first" label="first-help"><span>a</span></Tooltip>
        <Tooltip content="second" label="second-help"><span>b</span></Tooltip>
      </>,
    )

    fireEvent.mouseEnter(screen.getByLabelText('first-help'))
    // Only the hovered trigger's bubble exists; the other stays closed.
    expect(screen.getAllByRole('tooltip')).toHaveLength(1)
    expect(screen.getByRole('tooltip')).toHaveTextContent('first')
  })
})
