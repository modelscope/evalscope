import { describe, expect, it } from 'vitest'

import {
  checkStructure,
  formatStructureReport,
  REQUIRED_SECTIONS,
  runStructureCheck,
} from '../../../scripts/drift/structureCheck.ts'

// Req 17.1 — DESIGN.md must expose five separately addressable sections
// (principles, tokens, components, metrics, decisions). If any is missing the
// structure check must fail.

/** Build a minimal Markdown doc that exposes the given anchor ids as headings. */
function docWithSections(ids: readonly string[]): string {
  return ['# Design System', '', ...ids.map((id) => `## ${id} {#${id}}\n\nbody\n`)].join('\n')
}

describe('checkStructure — required sections', () => {
  it('passes when every required section is addressable via explicit anchors', () => {
    const md = docWithSections(REQUIRED_SECTIONS.map((s) => s.id))
    const result = checkStructure(md)
    expect(result.ok).toBe(true)
    expect(result.missing).toEqual([])
    expect(result.found).toEqual(['principles', 'tokens', 'components', 'metrics', 'decisions'])
  })

  it('fails and reports the id when a single section is missing', () => {
    // Drop `metrics`.
    const md = docWithSections(['principles', 'tokens', 'components', 'decisions'])
    const result = checkStructure(md)
    expect(result.ok).toBe(false)
    expect(result.missing).toEqual(['metrics'])
    expect(result.found).toEqual(['principles', 'tokens', 'components', 'decisions'])
  })

  it('fails when the document has no sections at all', () => {
    const result = checkStructure('# Design System\n\njust prose, no anchors\n')
    expect(result.ok).toBe(false)
    expect(result.missing).toEqual(['principles', 'tokens', 'components', 'metrics', 'decisions'])
    expect(result.found).toEqual([])
  })

  it('reports every missing section, preserving canonical order', () => {
    const md = docWithSections(['tokens', 'components'])
    const result = checkStructure(md)
    expect(result.ok).toBe(false)
    expect(result.missing).toEqual(['principles', 'metrics', 'decisions'])
  })
})

describe('checkStructure — addressability mechanisms', () => {
  it('detects an explicit heading anchor: "## Design Tokens {#tokens}"', () => {
    const md = '# D\n\n## Design Tokens {#tokens}\n'
    expect(checkStructure(md, [{ id: 'tokens', label: 'Tokens' }]).ok).toBe(true)
  })

  it('detects an inline HTML anchor: <a id="metrics">', () => {
    const md = '# D\n\n<a id="metrics"></a>\n### Metric Display Contract\n'
    expect(checkStructure(md, [{ id: 'metrics', label: 'Metrics' }]).ok).toBe(true)
  })

  it('detects an <a name="..."> anchor', () => {
    const md = '# D\n\n<a name="decisions"></a>\n'
    expect(checkStructure(md, [{ id: 'decisions', label: 'Decisions' }]).ok).toBe(true)
  })

  it('detects a GitHub-style slug of a bare heading: "## Components"', () => {
    const md = '# D\n\n## Components\n\nbody\n'
    expect(checkStructure(md, [{ id: 'components', label: 'Components' }]).ok).toBe(true)
  })

  it('does not match a section whose slug differs and has no explicit anchor', () => {
    // "Metric Display Contract" slugs to "metric-display-contract", not "metrics".
    const md = '# D\n\n## Metric Display Contract\n\nbody\n'
    expect(checkStructure(md, [{ id: 'metrics', label: 'Metrics' }]).ok).toBe(false)
  })

  it('is case-insensitive on anchor ids', () => {
    const md = '# D\n\n## Principles {#PRINCIPLES}\n'
    expect(checkStructure(md, [{ id: 'principles', label: 'Principles' }]).ok).toBe(true)
  })
})

describe('formatStructureReport', () => {
  it('renders a pass line when ok', () => {
    const report = formatStructureReport({
      mdPath: '/x/DESIGN.md',
      ok: true,
      found: ['principles', 'tokens'],
      missing: [],
    })
    expect(report).toContain('Structure check passed')
  })

  it('lists missing anchors when failing', () => {
    const report = formatStructureReport({
      mdPath: '/x/DESIGN.md',
      ok: false,
      found: ['principles'],
      missing: ['metrics', 'decisions'],
    })
    expect(report).toContain('FAILED')
    expect(report).toContain('#metrics')
    expect(report).toContain('#decisions')
  })
})

describe('runStructureCheck — against the real DESIGN.md', () => {
  it('finds all five addressable sections in the repository DESIGN.md', () => {
    const result = runStructureCheck()
    expect(result.missing).toEqual([])
    expect(result.ok).toBe(true)
  })
})
