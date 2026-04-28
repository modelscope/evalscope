async function handleResponse<T>(res: Response): Promise<T> {
  if (!res.ok) {
    const body = await res.json().catch(() => ({ error: res.statusText }))
    throw new Error(body.error || `HTTP ${res.status}`)
  }
  return res.json() as Promise<T>
}

export async function api<T = unknown>(path: string, params?: Record<string, unknown>): Promise<T> {
  const url = new URL(path, window.location.origin)
  if (params) {
    for (const [k, v] of Object.entries(params)) {
      if (v !== undefined && v !== '') url.searchParams.set(k, String(v))
    }
  }
  const res = await fetch(url.toString())
  return handleResponse<T>(res)
}

export async function apiPost<T = unknown>(path: string, body: unknown, headers?: Record<string, string>): Promise<T> {
  const res = await fetch(path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', ...headers },
    body: JSON.stringify(body),
  })
  return handleResponse<T>(res)
}
