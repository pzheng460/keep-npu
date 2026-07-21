export const AUTO_REFRESH_INTERVAL_MS = 10000
const PUBLIC_VRAM_MAX_BYTES = 2 ** 50

export function canRunAutoRefresh(autoRefresh, visibilityState = "visible") {
  return autoRefresh && visibilityState !== "hidden"
}

export function canReuseInFlightRefresh(inFlightRefresh, afterMutation = false) {
  return Boolean(inFlightRefresh) && !afterMutation
}

export function formatRefreshWarningMessage(error) {
  const message =
    error instanceof Error ? error.message : String(error ?? "unknown error")
  return `Refresh warning: ${message || "unknown error"}`
}

const SESSION_STATES = new Set([
  "active",
  "starting",
  "stopping",
  "runtime_failed",
  "stop_failed"
])

function isObjectRecord(value) {
  return value !== null && typeof value === "object" && !Array.isArray(value)
}

function isNonNegativeInteger(value) {
  return Number.isInteger(value) && value >= 0
}

function isNullableMemory(value) {
  return value === null || isNonNegativeInteger(value)
}

function isNullableUtilization(value) {
  return (
    value === null ||
    (typeof value === "number" &&
      Number.isFinite(value) &&
      value >= 0 &&
      value <= 100)
  )
}

function isNpuIds(value) {
  return value === null || (Array.isArray(value) && value.every(isNonNegativeInteger))
}

function isVramValue(value) {
  return (
    typeof value === "string" ||
    (isNonNegativeInteger(value) && value >= 4 && value <= PUBLIC_VRAM_MAX_BYTES)
  )
}

function isValidNpuRecord(npu) {
  return (
    isObjectRecord(npu) &&
    isNonNegativeInteger(npu.id) &&
    isNonNegativeInteger(npu.visible_id) &&
    npu.id === npu.visible_id &&
    typeof npu.platform === "string" &&
    typeof npu.name === "string" &&
    isNullableMemory(npu.memory_total) &&
    isNullableMemory(npu.memory_used) &&
    (npu.memory_total === null ||
      npu.memory_used === null ||
      npu.memory_used <= npu.memory_total) &&
    isNullableUtilization(npu.utilization)
  )
}

function isValidSessionRecord(session) {
  if (
    !isObjectRecord(session) ||
    typeof session.job_id !== "string" ||
    !session.job_id ||
    !isObjectRecord(session.params) ||
    !SESSION_STATES.has(session.state) ||
    (session.last_error !== null && typeof session.last_error !== "string")
  ) {
    return false
  }

  const params = session.params
  return (
    isNpuIds(params.npu_ids) &&
    isVramValue(params.vram) &&
    typeof params.interval === "number" &&
    Number.isFinite(params.interval) &&
    params.interval > 0 &&
    Number.isInteger(params.busy_threshold) &&
    (params.busy_threshold === -1 ||
      (params.busy_threshold >= 0 && params.busy_threshold <= 100))
  )
}

function readRefreshList(result, fieldName, malformedMessage, isValidItem) {
  if (result.status === "rejected") {
    return {
      items: null,
      warning: formatRefreshWarningMessage(result.reason)
    }
  }
  const items = result.value?.[fieldName]
  if (!Array.isArray(items)) {
    return {
      items: null,
      warning: formatRefreshWarningMessage(new Error(malformedMessage))
    }
  }
  if (!items.every(isValidItem)) {
    return {
      items: null,
      warning: formatRefreshWarningMessage(new Error(malformedMessage))
    }
  }
  return { items, warning: null }
}

export async function fetchDashboardPayloads(requestJson) {
  const [npuResult, sessionResult] = await Promise.allSettled([
    requestJson("GET", "/api/npus"),
    requestJson("GET", "/api/sessions")
  ])
  const npuPayload = readRefreshList(
    npuResult,
    "npus",
    "malformed NPU list response",
    isValidNpuRecord
  )
  const sessionPayload = readRefreshList(
    sessionResult,
    "active_jobs",
    "malformed session list response",
    isValidSessionRecord
  )

  return {
    npus: npuPayload.items,
    sessions: sessionPayload.items,
    warning: npuPayload.warning ?? sessionPayload.warning
  }
}

export function nextRefreshMessage({
  afterMutation = false,
  previousMessage = null,
  userInitiated = false,
  warning = null
} = {}) {
  if (warning) {
    return afterMutation ? previousMessage : warning
  }
  return userInitiated ? "Dashboard refreshed." : previousMessage
}

export function formatRefreshMode(autoRefresh, visibilityState = "visible") {
  if (!autoRefresh) {
    return "manual refresh"
  }
  if (visibilityState === "hidden") {
    return "auto paused"
  }
  return `auto ${AUTO_REFRESH_INTERVAL_MS / 1000}s`
}
