// Direct tests for the message-text helpers.
//
// These were previously reachable only by rendering ChatView, which left the
// system-prompt parsing branches (the three delimiter dialects) uncovered even
// though they are plain string functions. Moving them into the domain layer makes
// them table-testable, so each dialect is asserted on its own.

import { describe, expect, it } from 'vitest'

import { argsPreview, contentToText, hasSystemPrompt, parseSystemUser } from './messageText'
import type { ContentBlock } from '@/api/types'

describe('hasSystemPrompt', () => {
  it.each([
    ['<|system|> be terse', true],
    ['[system] be terse', true],
    ['system: be terse', true],
    ['SYSTEM: be terse', true],
    ['<|system|>a<|user|>b', true],
    ['```\nsystem role\n```', true],
    ['  <|system|> leading space', true],
    ['what is 2 + 2?', false],
    ['the system under test', false],
    ['', false],
  ])('detects %j as %s', (input, expected) => {
    expect(hasSystemPrompt(input)).toBe(expected)
  })
})

describe('parseSystemUser', () => {
  it('splits the pipe-delimited dialect', () => {
    expect(parseSystemUser('<|system|>be terse<|user|>2 + 2?')).toEqual({
      system: 'be terse',
      user: '2 + 2?',
    })
  })

  // A system prompt with no user turn: the whole input is the system prompt and
  // the user part is empty, rather than the raw input (delimiter included) being
  // echoed back as the user turn.
  it('returns an empty user part when there is no user turn', () => {
    expect(parseSystemUser('<|system|>be terse')).toEqual({ system: 'be terse', user: '' })
    expect(parseSystemUser('[system]be terse')).toEqual({ system: 'be terse', user: '' })
    expect(parseSystemUser('system: be terse')).toEqual({ system: 'be terse', user: '' })
  })

  it('stops the user part at an assistant turn', () => {
    expect(parseSystemUser('<|system|>s<|user|>u<|assistant|>a')).toEqual({ system: 's', user: 'u' })
  })

  it('splits the bracket-delimited dialect', () => {
    expect(parseSystemUser('[system]be terse[user]2 + 2?')).toEqual({
      system: 'be terse',
      user: '2 + 2?',
    })
  })

  it('splits the colon-delimited dialect', () => {
    expect(parseSystemUser('system: be terse\nuser: 2 + 2?')).toEqual({
      system: 'be terse',
      user: '2 + 2?',
    })
  })

  it('treats input with no recognised delimiter as a bare user turn', () => {
    expect(parseSystemUser('2 + 2?')).toEqual({ system: '', user: '2 + 2?' })
  })
})

describe('contentToText', () => {
  it('returns a string payload unchanged', () => {
    expect(contentToText('plain')).toBe('plain')
  })

  it('joins text and reasoning blocks and names the media ones', () => {
    const blocks: ContentBlock[] = [
      { type: 'text', text: 'answer' },
      { type: 'reasoning', reasoning: 'because' },
      { type: 'image', image: 'a.png' },
      { type: 'audio', audio: 'a.mp3' },
      { type: 'video', video: 'a.webm' },
    ]
    expect(contentToText(blocks)).toBe('answer\n\nbecause\n\n[image]\n\n[audio]\n\n[video]')
  })

  it('drops blocks whose payload is absent', () => {
    expect(contentToText([{ type: 'text' }, { type: 'text', text: 'kept' }])).toBe('kept')
  })
})

describe('argsPreview', () => {
  it('renders nothing for an absent argument object', () => {
    expect(argsPreview(null)).toBe('')
    expect(argsPreview(undefined)).toBe('')
  })

  it('collapses whitespace in serialized arguments', () => {
    expect(argsPreview({ a: 1 })).toBe('{"a":1}')
    expect(argsPreview('a   \n  b')).toBe('a b')
  })

  it('truncates past the limit with an ellipsis', () => {
    expect(argsPreview('x'.repeat(20), 10)).toBe(`${'x'.repeat(10)}…`)
  })

  it('falls back to String() when the value cannot be serialized', () => {
    const cyclic: Record<string, unknown> = {}
    cyclic.self = cyclic
    expect(argsPreview(cyclic)).toBe('[object Object]')
  })
})
