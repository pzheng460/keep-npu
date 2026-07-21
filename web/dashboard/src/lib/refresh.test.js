import { describe, expect, it } from "vitest"

import {
  AUTO_REFRESH_INTERVAL_MS,
  canRunAutoRefresh,
  canReuseInFlightRefresh,
  fetchDashboardPayloads,
  formatRefreshWarningMessage,
  formatRefreshMode,
  nextRefreshMessage
} from "./refresh"

const validNpuRecord = {
  id: 0,
  visible_id: 0,
  platform: "cuda",
  name: "Test NPU",
  memory_total: 1024,
  memory_used: 512,
  utilization: 12
}

const validSessionRecord = {
  job_id: "job-a",
  params: {
    npu_ids: [0],
    vram: "1GiB",
    interval: 300,
    busy_threshold: 25
  },
  state: "active",
  last_error: null
}

describe("dashboard refresh helpers", () => {
  it("keeps auto refresh opt-in and pauses while hidden", () => {
    expect(AUTO_REFRESH_INTERVAL_MS).toBeGreaterThanOrEqual(10000)
    expect(canRunAutoRefresh(false, "visible")).toBe(false)
    expect(canRunAutoRefresh(true, "hidden")).toBe(false)
    expect(canRunAutoRefresh(true, "visible")).toBe(true)
  })

  it("renders concise refresh mode labels", () => {
    expect(formatRefreshMode(false, "visible")).toBe("manual refresh")
    expect(formatRefreshMode(true, "hidden")).toBe("auto paused")
    expect(formatRefreshMode(true, "visible")).toBe("auto 10s")
  })

  it("does not reuse stale in-flight refreshes after mutations", () => {
    const inFlightRefresh = Promise.resolve()

    expect(canReuseInFlightRefresh(null, false)).toBe(false)
    expect(canReuseInFlightRefresh(inFlightRefresh, false)).toBe(true)
    expect(canReuseInFlightRefresh(inFlightRefresh, true)).toBe(false)
  })

  it("formats refresh failures from non-Error rejections", () => {
    expect(formatRefreshWarningMessage(new Error("service unavailable"))).toBe(
      "Refresh warning: service unavailable"
    )
    expect(formatRefreshWarningMessage("network offline")).toBe(
      "Refresh warning: network offline"
    )
    expect(formatRefreshWarningMessage(null)).toBe("Refresh warning: unknown error")
  })

  it("keeps session payloads when telemetry refresh fails", async () => {
    const calls = []
    const requestJson = async (method, path) => {
      calls.push([method, path])
      if (path === "/api/npus") {
        throw new Error("telemetry unavailable")
      }
      return { active_jobs: [validSessionRecord] }
    }

    await expect(fetchDashboardPayloads(requestJson)).resolves.toEqual({
      npus: null,
      sessions: [validSessionRecord],
      warning: "Refresh warning: telemetry unavailable"
    })
    expect(calls).toEqual([
      ["GET", "/api/npus"],
      ["GET", "/api/sessions"]
    ])
  })

  it("accepts integer-byte VRAM values in session refresh payloads", async () => {
    const sessionWithByteVram = {
      ...validSessionRecord,
      params: {
        ...validSessionRecord.params,
        vram: 1024
      }
    }
    const requestJson = async (_method, path) => {
      if (path === "/api/sessions") {
        return { active_jobs: [sessionWithByteVram] }
      }
      return { npus: [validNpuRecord] }
    }

    await expect(fetchDashboardPayloads(requestJson)).resolves.toEqual({
      npus: [validNpuRecord],
      sessions: [sessionWithByteVram],
      warning: null
    })
  })

  it("keeps telemetry payloads when session refresh fails", async () => {
    const requestJson = async (_method, path) => {
      if (path === "/api/sessions") {
        throw new Error("sessions unavailable")
      }
      return { npus: [validNpuRecord] }
    }

    await expect(fetchDashboardPayloads(requestJson)).resolves.toEqual({
      npus: [validNpuRecord],
      sessions: null,
      warning: "Refresh warning: sessions unavailable"
    })
  })

  it("warns without replacing telemetry when the NPU payload is malformed", async () => {
    const requestJson = async (_method, path) => {
      if (path === "/api/npus") {
        return {}
      }
      return { active_jobs: [validSessionRecord] }
    }

    await expect(fetchDashboardPayloads(requestJson)).resolves.toEqual({
      npus: null,
      sessions: [validSessionRecord],
      warning: "Refresh warning: malformed NPU list response"
    })
  })

  it("warns without replacing sessions when the session payload is malformed", async () => {
    const requestJson = async (_method, path) => {
      if (path === "/api/sessions") {
        return { active_jobs: {} }
      }
      return { npus: [validNpuRecord] }
    }

    await expect(fetchDashboardPayloads(requestJson)).resolves.toEqual({
      npus: [validNpuRecord],
      sessions: null,
      warning: "Refresh warning: malformed session list response"
    })
  })

  it("warns without replacing sessions when a session record is malformed", async () => {
    const requestJson = async (_method, path) => {
      if (path === "/api/sessions") {
        return { active_jobs: [{ job_id: "bad-session" }] }
      }
      return { npus: [validNpuRecord] }
    }

    await expect(fetchDashboardPayloads(requestJson)).resolves.toEqual({
      npus: [validNpuRecord],
      sessions: null,
      warning: "Refresh warning: malformed session list response"
    })
  })

  it("warns without replacing telemetry when a NPU record is malformed", async () => {
    const requestJson = async (_method, path) => {
      if (path === "/api/npus") {
        return { npus: [{ id: 0, name: "missing render fields" }] }
      }
      return { active_jobs: [validSessionRecord] }
    }

    await expect(fetchDashboardPayloads(requestJson)).resolves.toEqual({
      npus: null,
      sessions: [validSessionRecord],
      warning: "Refresh warning: malformed NPU list response"
    })
  })

  it("preserves mutation result messages when follow-up refresh warns", () => {
    expect(nextRefreshMessage({ userInitiated: true })).toBe("Dashboard refreshed.")
    expect(nextRefreshMessage({ warning: "Refresh warning: telemetry unavailable" })).toBe(
      "Refresh warning: telemetry unavailable"
    )
    expect(
      nextRefreshMessage({
        afterMutation: true,
        previousMessage: "Released session: job-a.",
        warning: "Refresh warning: telemetry unavailable"
      })
    ).toBe("Released session: job-a.")
  })
})
