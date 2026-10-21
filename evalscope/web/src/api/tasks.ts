import { apiValidated } from './client'
import type { ListActiveTasksResponse } from './types'

/**
 * List tasks currently running on the service, regardless of which browser
 * session launched them. Backed by the service's in-process task registry.
 */
export async function listActiveTasks(signal?: AbortSignal): Promise<ListActiveTasksResponse> {
  return apiValidated<ListActiveTasksResponse>('/api/v1/tasks/active', { signal })
}
