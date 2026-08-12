import { formatStructureReport, runStructureCheck } from './structureCheck'
import { formatDriftReport, runTokenDriftCheck } from './tokenDrift'
import { runLocaleKeyCheck } from './localeKeyCheck'

const tokenResult = runTokenDriftCheck()
const structureResult = runStructureCheck()

process.stdout.write(`${formatDriftReport(tokenResult)}\n`)
process.stdout.write(`${formatStructureReport(structureResult)}\n`)

const localeOk = runLocaleKeyCheck()
if (!tokenResult.ok || !structureResult.ok || !localeOk) {
  process.exitCode = 1
}
