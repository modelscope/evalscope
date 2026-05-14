import type { ContentBlock } from '@/api/types'

/** Detect whether the input string contains an embedded system prompt. */
export function hasSystemPrompt(input: string): boolean {
  const lower = input.trim().toLowerCase()
  return (
    lower.startsWith('<|system|>') ||
    lower.startsWith('[system]') ||
    lower.startsWith('system:') ||
    /^```[\s\S]*?system/i.test(input.trim()) ||
    (input.includes('<|system|>') && input.includes('<|user|>'))
  )
}

/** Split a mixed input string into separate system and user parts. */
export function parseSystemUser(input: string): { system: string; user: string } {
  const sysMatch = input.match(/<\|system\|>([\s\S]*?)(?:<\|user\|>|$)/i)
  const userMatch = input.match(/<\|user\|>([\s\S]*?)(?:<\|assistant\|>|$)/i)
  if (sysMatch) {
    return {
      system: sysMatch[1].trim(),
      user: userMatch ? userMatch[1].trim() : input.replace(/<\|system\|>[\s\S]*?<\|user\|>/i, '').trim(),
    }
  }
  const bracketMatch = input.match(/^\[system\]([\s\S]*?)(?:\[user\]|$)/i)
  if (bracketMatch) {
    return {
      system: bracketMatch[1].trim(),
      user: input.replace(/^\[system\][\s\S]*?(?:\[user\])/i, '').trim(),
    }
  }
  const colonMatch = input.match(/^system:\s*([\s\S]*?)(?:\nuser:|$)/i)
  if (colonMatch) {
    return {
      system: colonMatch[1].trim(),
      user: input.replace(/^system:\s*[\s\S]*?\nuser:\s*/i, '').trim(),
    }
  }
  return { system: '', user: input }
}

/** Extract plain text from string or ContentBlock[] for clipboard copy. */
export function contentToText(content: string | ContentBlock[]): string {
  if (typeof content === 'string') return content
  return content
    .map(b => {
      if (b.type === 'text') return b.text ?? ''
      if (b.type === 'reasoning') return b.reasoning ?? ''
      if (b.type === 'image') return '[image]'
      if (b.type === 'audio') return '[audio]'
      if (b.type === 'video') return '[video]'
      return ''
    })
    .join('\n\n')
    .trim()
}

/** One-line preview for arguments JSON (truncate). */
export function argsPreview(args: unknown, max = 100): string {
  if (args == null) return ''
  let s: string
  try {
    s = typeof args === 'string' ? args : JSON.stringify(args)
  } catch {
    s = String(args)
  }
  s = s.replace(/\s+/g, ' ').trim()
  return s.length > max ? s.slice(0, max) + '…' : s
}
