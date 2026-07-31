// Enterprise Replica Attribution Parser & Collector

export function extractAttributionFromHeaders(headers) {
  if (!headers) return null
  const getHeader = (name) => {
    if (typeof headers.get === 'function') return headers.get(name) || headers.get(name.toLowerCase())
    return headers[name] || headers[name.toLowerCase()] || null
  }

  const lane = getHeader('x-camelid-lane')
  const configSha256 = getHeader('x-camelid-config-sha256')
  const admissionSha256 = getHeader('x-camelid-admission-sha256')
  const modelSha256 = getHeader('x-camelid-model-sha256')
  const host = getHeader('x-camelid-host')
  const workerThreads = getHeader('x-camelid-worker-threads')
  const requestId = getHeader('x-camelid-request-id')

  if (!lane && !configSha256 && !modelSha256 && !requestId) return null

  return {
    lane: lane || 'deterministic',
    configSha256: configSha256 || null,
    admissionSha256: admissionSha256 || null,
    modelSha256: modelSha256 || null,
    host: host || null,
    workerThreads: workerThreads ? Number.parseInt(workerThreads, 10) : null,
    requestId: requestId || null,
    timestamp: new Date().toISOString(),
  }
}

export function extractAttributionFromBody(jsonBody) {
  if (!jsonBody || typeof jsonBody !== 'object') return null
  const lane = jsonBody.camelid_lane
  const configSha256 = jsonBody.camelid_config_sha256
  const admissionSha256 = jsonBody.camelid_admission_sha256
  const modelSha256 = jsonBody.camelid_model_sha256
  const host = jsonBody.camelid_host
  const workerThreads = jsonBody.camelid_worker_threads

  if (!lane && !configSha256 && !modelSha256) return null

  return {
    lane: lane || 'deterministic',
    configSha256: configSha256 || null,
    admissionSha256: admissionSha256 || null,
    modelSha256: modelSha256 || null,
    host: host || null,
    workerThreads: workerThreads ? Number(workerThreads) : null,
    requestId: jsonBody.request_id || null,
    timestamp: new Date().toISOString(),
  }
}

export function formatShaShort(sha) {
  if (!sha) return '—'
  return sha.length > 12 ? sha.slice(0, 12) : sha
}
