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

/**
 * Split a mixed input string into separate system and user parts.
 *
 * Three delimiter dialects are recognised (`<|system|>`/`<|user|>`, `[system]`/
 * `[user]`, `system:`/`user:`). When a system prompt is present but there is no
 * user turn, the whole input is the system prompt and the user part is empty --
 * the earlier fallbacks returned the raw input (delimiter included) as the user
 * part, which made the system text appear twice.
 */
export function parseSystemUser(input: string): { system: string; user: string } {
  const sysMatch = input.match(/<\|system\|>([\s\S]*?)(?:<\|user\|>|$)/i)
  if (sysMatch) {
    const userMatch = input.match(/<\|user\|>([\s\S]*?)(?:<\|assistant\|>|$)/i)
    return {
      system: sysMatch[1].trim(),
      user: userMatch ? userMatch[1].trim() : '',
    }
  }
  const bracketMatch = input.match(/^\[system\]([\s\S]*?)(?:\[user\]|$)/i)
  if (bracketMatch) {
    const userMatch = input.match(/\[user\]([\s\S]*?)$/i)
    return {
      system: bracketMatch[1].trim(),
      user: userMatch ? userMatch[1].trim() : '',
    }
  }
  const colonMatch = input.match(/^system:\s*([\s\S]*?)(?:\nuser:|$)/i)
  if (colonMatch) {
    const userMatch = input.match(/\nuser:\s*([\s\S]*?)$/i)
    return {
      system: colonMatch[1].trim(),
      user: userMatch ? userMatch[1].trim() : '',
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
