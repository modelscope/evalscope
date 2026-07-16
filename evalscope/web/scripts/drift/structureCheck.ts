import { readFileSync } from 'node:fs'
import { dirname, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

export const REQUIRED_SECTIONS = ['principles', 'tokens', 'components', 'metrics', 'decisions'] as const

export interface StructureCheckResult {
  ok: boolean
  missing: string[]
}

export function checkStructure(markdown: string): StructureCheckResult {
  const anchors = new Set<string>()
  for (const match of markdown.matchAll(/\{#([\w-]+)\}|<(?:a)\s+(?:id|name)=["']([\w-]+)["']/gi)) {
    anchors.add(match[1] ?? match[2])
  }
  for (const match of markdown.matchAll(/^#{1,6}\s+(.+?)(?:\s+\{#[^}]+\})?\s*$/gm)) {
    anchors.add(match[1].toLowerCase().replace(/[^a-z0-9\s-]/g, '').trim().replace(/\s+/g, '-'))
  }
  const missing = REQUIRED_SECTIONS.filter((section) => !anchors.has(section))
  return { ok: missing.length === 0, missing }
}

export function runStructureCheck(): StructureCheckResult {
  const webRoot = resolve(dirname(fileURLToPath(import.meta.url)), '..', '..')
  return checkStructure(readFileSync(resolve(webRoot, '..', '..', 'DESIGN.md'), 'utf8'))
}

export function formatStructureReport(result: StructureCheckResult): string {
  return result.ok
    ? 'Design structure check passed.'
    : `Design structure check failed: missing ${result.missing.join(', ')}.`
}

if (import.meta.url === `file://${process.argv[1]}`) {
  const result = runStructureCheck()
  process.stdout.write(`${formatStructureReport(result)}\n`)
  if (!result.ok) process.exitCode = 1
}
