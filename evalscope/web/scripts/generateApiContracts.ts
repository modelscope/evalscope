import { execFileSync } from 'node:child_process'
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { dirname, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

import { compileFromFile } from 'json-schema-to-typescript'

const scriptDir = dirname(fileURLToPath(import.meta.url))
const webRoot = resolve(scriptDir, '..')
const repoRoot = resolve(webRoot, '../..')
const generatedDir = resolve(webRoot, 'src/api/generated')
const committedSchema = resolve(generatedDir, 'contracts.schema.json')
const committedTypes = resolve(generatedDir, 'contracts.ts')
const check = process.argv.includes('--check')
const tempRoot = check ? mkdtempSync(resolve(tmpdir(), 'evalscope-api-contracts-')) : generatedDir
const schemaPath = resolve(tempRoot, 'contracts.schema.json')
const typesPath = resolve(tempRoot, 'contracts.ts')

function generateSchema(): void {
  execFileSync(process.env.PYTHON || 'python', [
    resolve(repoRoot, 'scripts/generate_web_api_contracts.py'),
    '--output',
    schemaPath,
  ], { cwd: repoRoot, stdio: 'inherit' })
}

async function generateTypes(): Promise<void> {
  const source = await compileFromFile(schemaPath, {
    bannerComment: '/* eslint-disable */\n/** Generated from Pydantic Web API response models. Do not edit. */',
    format: true,
    unknownAny: false,
  })
  writeFileSync(typesPath, source, 'utf8')
}

function assertCurrent(generated: string, committed: string): void {
  const expected = readFileSync(committed, 'utf8')
  const actual = readFileSync(generated, 'utf8')
  if (actual !== expected) {
    throw new Error(`Generated API contract is stale: ${committed}`)
  }
}

try {
  generateSchema()
  await generateTypes()
  if (check) {
    assertCurrent(schemaPath, committedSchema)
    assertCurrent(typesPath, committedTypes)
  }
} finally {
  if (check) rmSync(tempRoot, { recursive: true, force: true })
}
