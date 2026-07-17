import type { Permission, StreamChunk, ThreadHistoryResponse } from './types'

/** Same-origin base for CRUD (proxied by Next.js rewrites). */
const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL ?? ''

/**
 * SSE must hit FastAPI directly. Next.js rewrites buffer `text/event-stream`
 * until the upstream finishes, so the UI only sees a burst at the end.
 */
const STREAM_API_BASE =
  process.env.NEXT_PUBLIC_STREAM_API_BASE_URL ??
  process.env.NEXT_PUBLIC_BACKEND_API_BASE_URL ??
  'http://127.0.0.1:8000'

export type StreamRequest = {
  agentId: number
  threadId: string
  message?: string
  permissions?: Permission[]
  signal?: AbortSignal
  onChunk: (chunk: StreamChunk) => void
}

function emitSseFrames(buffer: string, onChunk: (chunk: StreamChunk) => void): string {
  // SSE events are delimited by a blank line.
  const parts = buffer.split('\n\n')
  const rest = parts.pop() ?? ''
  for (const frame of parts) {
    for (const line of frame.split('\n')) {
      const trimmed = line.trim()
      if (!trimmed.startsWith('data:')) continue
      const payload = trimmed.slice(5).trim()
      if (!payload || payload === '[DONE]') continue
      onChunk(JSON.parse(payload) as StreamChunk)
    }
  }
  return rest
}

export async function streamAgent({
  agentId,
  threadId,
  message,
  permissions,
  signal,
  onChunk,
}: StreamRequest): Promise<void> {
  const body: Record<string, unknown> = {
    agent_id: agentId,
    thread_id: threadId,
  }
  if (permissions) {
    body.permissions = permissions
  } else if (message !== undefined) {
    body.message = message
  }

  const response = await fetch(`${STREAM_API_BASE}/chats/stream`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', Accept: 'text/event-stream' },
    body: JSON.stringify(body),
    signal,
    cache: 'no-store',
  })

  if (!response.ok) {
    const detail = await response.text()
    throw new Error(detail || `Stream failed (${response.status})`)
  }
  if (!response.body) {
    throw new Error('Stream response had no body')
  }

  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''

  while (true) {
    const { done, value } = await reader.read()
    if (done) break
    buffer = emitSseFrames(buffer + decoder.decode(value, { stream: true }), onChunk)
  }

  buffer = emitSseFrames(buffer + decoder.decode(), onChunk)
  if (buffer.trim()) {
    // Final partial frame (no trailing blank line).
    emitSseFrames(`${buffer}\n\n`, onChunk)
  }
}

export async function cancelStream(threadId: string): Promise<void> {
  const response = await fetch(
    `${STREAM_API_BASE}/chats/cancel-stream/${encodeURIComponent(threadId)}`,
    { method: 'POST', cache: 'no-store' },
  )
  if (!response.ok && response.status !== 404) {
    const detail = await response.text()
    throw new Error(detail || `Cancel failed (${response.status})`)
  }
}

export type ClearFromLastUserResult = {
  thread_id: string
  message: string
  removed_count: number
  remaining_count: number
}

export async function clearFromLastUser(
  threadId: string,
  agentId?: number,
): Promise<ClearFromLastUserResult> {
  const params = new URLSearchParams()
  if (agentId != null) {
    params.set('agent_id', String(agentId))
  }
  const query = params.toString()
  const response = await fetch(
    `${API_BASE}/chats/threads/${encodeURIComponent(threadId)}/clear-from-last-user${
      query ? `?${query}` : ''
    }`,
    { method: 'POST' },
  )
  if (!response.ok) {
    const detail = await response.text()
    throw new Error(detail || `Clear failed (${response.status})`)
  }
  return (await response.json()) as ClearFromLastUserResult
}

export class ThreadNotFoundError extends Error {
  constructor(public threadId: string) {
    super(`No checkpoint history for thread ${threadId}`)
    this.name = 'ThreadNotFoundError'
  }
}

export async function getHistory(
  threadId: string,
  agentId?: number,
): Promise<ThreadHistoryResponse> {
  const params = new URLSearchParams()
  if (agentId != null) {
    params.set('agent_id', String(agentId))
  }
  const query = params.toString()
  const response = await fetch(
    `${API_BASE}/chats/get-history/${encodeURIComponent(threadId)}${
      query ? `?${query}` : ''
    }`,
  )
  if (response.status === 404) {
    throw new ThreadNotFoundError(threadId)
  }
  if (!response.ok) {
    const detail = await response.text()
    throw new Error(detail || `History failed (${response.status})`)
  }
  return (await response.json()) as ThreadHistoryResponse
}

export type TranscriptionResult = {
  text: string
  utterances?: Array<{
    speaker: string
    text: string
    start?: number | null
    end?: number | null
  }>
  language_code?: string | null
  audio_duration?: number | null
}

export async function speechToText(audio: Blob, filename: string): Promise<TranscriptionResult> {
  const form = new FormData()
  form.append('audio', audio, filename)

  const response = await fetch(`${API_BASE}/chats/speech-to-text`, {
    method: 'POST',
    body: form,
  })

  if (!response.ok) {
    const detail = await response.text()
    throw new Error(detail || `Speech-to-text failed (${response.status})`)
  }

  return (await response.json()) as TranscriptionResult
}

/* ---------- Generic CRUD helpers ---------- */

async function parseError(response: Response, fallback: string): Promise<never> {
  const detail = await response.text()
  throw new Error(detail || `${fallback} (${response.status})`)
}

export async function crudGetMany<T>(endpoint: string): Promise<T[]> {
  const response = await fetch(`${API_BASE}${endpoint}`)
  if (!response.ok) await parseError(response, 'Failed to load')
  return (await response.json()) as T[]
}

export async function crudGetOne<T>(endpoint: string, id: number): Promise<T> {
  const response = await fetch(`${API_BASE}${endpoint}/${id}`)
  if (!response.ok) await parseError(response, 'Failed to load')
  return (await response.json()) as T
}

export async function crudCreate<T>(
  endpoint: string,
  payload: Record<string, unknown>,
): Promise<T> {
  const response = await fetch(`${API_BASE}${endpoint}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  })
  if (!response.ok) await parseError(response, 'Failed to create')
  return (await response.json()) as T
}

export async function crudUpdate<T>(
  endpoint: string,
  id: number,
  payload: Record<string, unknown>,
): Promise<T> {
  const response = await fetch(`${API_BASE}${endpoint}/${id}`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  })
  if (!response.ok) await parseError(response, 'Failed to update')
  return (await response.json()) as T
}

export async function crudDelete(endpoint: string, id: number): Promise<void> {
  const response = await fetch(`${API_BASE}${endpoint}/${id}`, { method: 'DELETE' })
  if (!response.ok && response.status !== 404) {
    await parseError(response, 'Failed to delete')
  }
}
