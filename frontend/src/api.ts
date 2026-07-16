import type { Permission, StreamChunk } from './types'

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? ''

export type StreamRequest = {
  agentId: number
  threadId: string
  message?: string
  permissions?: Permission[]
  signal?: AbortSignal
  onChunk: (chunk: StreamChunk) => void
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

  const response = await fetch(`${API_BASE}/stream`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', Accept: 'text/event-stream' },
    body: JSON.stringify(body),
    signal,
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
    buffer += decoder.decode(value, { stream: true })
    const lines = buffer.split('\n')
    buffer = lines.pop() ?? ''

    for (const line of lines) {
      const trimmed = line.trim()
      if (!trimmed.startsWith('data:')) continue
      const payload = trimmed.slice(5).trim()
      if (!payload || payload === '[DONE]') continue
      onChunk(JSON.parse(payload) as StreamChunk)
    }
  }

  const leftover = buffer.trim()
  if (leftover.startsWith('data:')) {
    const payload = leftover.slice(5).trim()
    if (payload && payload !== '[DONE]') {
      onChunk(JSON.parse(payload) as StreamChunk)
    }
  }
}

export async function cancelStream(threadId: string): Promise<void> {
  const response = await fetch(
    `${API_BASE}/cancel-stream/${encodeURIComponent(threadId)}`,
    { method: 'POST' },
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
    `${API_BASE}/threads/${encodeURIComponent(threadId)}/clear-from-last-user${
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

  const response = await fetch(`${API_BASE}/speech-to-text`, {
    method: 'POST',
    body: form,
  })

  if (!response.ok) {
    const detail = await response.text()
    throw new Error(detail || `Speech-to-text failed (${response.status})`)
  }

  return (await response.json()) as TranscriptionResult
}
