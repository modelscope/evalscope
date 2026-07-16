import { describe, expect, it } from 'vitest'

import { loadFixture } from './loadFixture'

interface ExampleFixture {
  name: string
  value: number
  nested: { items: number[] }
}

describe('loadFixture', () => {
  it('loads a JSON fixture by name without the extension', () => {
    const fixture = loadFixture<ExampleFixture>('example')
    expect(fixture.name).toBe('example')
    expect(fixture.value).toBe(42)
    expect(fixture.nested.items).toEqual([1, 2, 3])
  })

  it('loads a JSON fixture when the .json extension is provided', () => {
    const fixture = loadFixture<ExampleFixture>('example.json')
    expect(fixture.value).toBe(42)
  })

  it('is deterministic across repeated calls', () => {
    const first = loadFixture<ExampleFixture>('example')
    const second = loadFixture<ExampleFixture>('example')
    expect(first).toEqual(second)
  })

  it('rejects names that escape the fixtures directory', () => {
    expect(() => loadFixture('../loadFixture')).toThrow(/outside the fixtures directory/)
  })

  it('throws for a missing fixture', () => {
    expect(() => loadFixture('does-not-exist')).toThrow()
  })
})
