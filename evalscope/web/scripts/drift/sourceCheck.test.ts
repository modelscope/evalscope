import { mkdtempSync, mkdirSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { describe, expect, it } from 'vitest'

import { checkSource } from './sourceCheck'

/**
 * Builds a throwaway `src` tree so each check can be driven from a known state
 * rather than from the real source, which is expected to always pass.
 */
function fixture(files: Record<string, string>): string {
  const root = join(mkdtempSync(join(tmpdir(), 'source-check-')), 'src')
  for (const [rel, content] of Object.entries(files)) {
    const full = join(root, rel)
    mkdirSync(join(full, '..'), { recursive: true })
    writeFileSync(full, content)
  }
  return root
}

const CSS = `@utility {
  .type-body-sm { @apply text-sm; }
  .type-title-md { @apply text-base font-bold; }
}`

/** A tree that satisfies every rule, used as the base for each negative case. */
const CLEAN: Record<string, string> = {
  'index.css': CSS,
  'App.tsx': "import Home from '@/pages/Home'\nexport default function App() { return <Home /> }\n",
  'pages/Home.tsx': "import Button from '@/components/ui/Button'\nexport default function Home() { return <Button /> }\n",
  'components/ui/Button.tsx': 'export default function Button() { return <button className="type-body-sm" /> }\n',
}

describe('checkSource', () => {
  it('passes a tree where every rule holds', () => {
    const result = checkSource(fixture(CLEAN))
    expect(result).toEqual({
      ok: true,
      undefinedTypeClasses: [],
      unusedComponents: [],
      unroutedPages: [],
    })
  })

  it('reports a type-* class that index.css never declares', () => {
    const result = checkSource(fixture({
      ...CLEAN,
      'components/ui/Button.tsx': 'export default function Button() { return <button className="type-title-sm" /> }\n',
    }))

    expect(result.ok).toBe(false)
    expect(result.undefinedTypeClasses).toEqual(['type-title-sm'])
  })

  it('reports a design-system component that nothing imports', () => {
    const result = checkSource(fixture({
      ...CLEAN,
      'components/ui/Orphan.tsx': 'export default function Orphan() { return null }\n',
    }))

    expect(result.ok).toBe(false)
    expect(result.unusedComponents).toEqual(['components/ui/Orphan.tsx'])
  })

  it('reports a pages/ module the router never renders', () => {
    const result = checkSource(fixture({
      ...CLEAN,
      'pages/StrayTab.tsx': 'export default function StrayTab() { return null }\n',
    }))

    expect(result.ok).toBe(false)
    expect(result.unroutedPages).toEqual(['pages/StrayTab.tsx'])
  })

  it('ignores test files when deciding whether a component is used', () => {
    const result = checkSource(fixture({
      ...CLEAN,
      'components/ui/Orphan.tsx': 'export default function Orphan() { return null }\n',
      'components/ui/Orphan.test.tsx': "import Orphan from './Orphan'\nit('renders', () => { Orphan() })\n",
    }))

    // Tests do not make a component reachable from the production application.
    expect(result.unusedComponents).toEqual(['components/ui/Orphan.tsx'])
  })
})
