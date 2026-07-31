// Unified Enterprise API Client for Gateway & Replica Communication
import { getAuthHeaders, getGatewayUrl } from './authStore'
import { extractAttributionFromHeaders } from './attribution'

export async function fetchEnterpriseJson(pathOrUrl, options = {}) {
  const gatewayBase = getGatewayUrl()
  const url = pathOrUrl.startsWith('http')
    ? pathOrUrl
    : `${gatewayBase}${pathOrUrl.startsWith('/') ? '' : '/'}${pathOrUrl}`

  const headers = {
    ...(options.body ? { 'Content-Type': 'application/json' } : {}),
    ...getAuthHeaders(),
    ...(options.headers || {}),
  }

  const response = await fetch(url, {
    ...options,
    headers,
  })

  const attribution = extractAttributionFromHeaders(response.headers)
  const text = await response.text()
  let body = null
  if (text) {
    try {
      body = JSON.parse(text)
    } catch {
      body = text
    }
  }

  if (response.status === 401) {
    const errorEvent = new CustomEvent('camelid:auth-error', {
      detail: { status: 401, body, message: 'Authentication required or token expired' },
    })
    window.dispatchEvent(errorEvent)
  }

  if (!response.ok) {
    const message = typeof body === 'string'
      ? body
      : body?.error?.message || body?.message || response.statusText
    const error = new Error(message)
    error.status = response.status
    error.body = body
    error.attribution = attribution
    throw error
  }

  if (body && typeof body === 'object' && !Array.isArray(body)) {
    body._attribution = attribution
  }

  return body
}
