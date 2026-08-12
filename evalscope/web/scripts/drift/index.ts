import { formatStructureReport, runStructureCheck } from './structureCheck'
import { formatSourceReport, runSourceCheck } from './sourceCheck'
import { formatDriftReport, runTokenDriftCheck } from './tokenDrift'
import { runLocaleKeyCheck } from './localeKeyCheck'

const tokenResult = runTokenDriftCheck()
const structureResult = runStructureCheck()
const sourceResult = runSourceCheck()

process.stdout.write(`${formatDriftReport(tokenResult)}\n`)
process.stdout.write(`${formatStructureReport(structureResult)}\n`)
process.stdout.write(`${formatSourceReport(sourceResult)}\n`)

const localeOk = runLocaleKeyCheck()
if (!tokenResult.ok || !structureResult.ok || !sourceResult.ok || !localeOk) {
  process.exitCode = 1
}
