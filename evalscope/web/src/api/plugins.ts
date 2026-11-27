import { apiValidated } from './client'
import type { PluginsResponse } from './types'

/** Plugins the operator enabled on this service, with the names each one registered. */
export async function listPlugins(signal?: AbortSignal): Promise<PluginsResponse> {
  return apiValidated<PluginsResponse>('/api/v1/plugins', { signal })
}
