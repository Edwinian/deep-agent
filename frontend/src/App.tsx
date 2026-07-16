import { useCallback, useEffect, useRef, useState } from 'react'
import { cancelStream, speechToText, streamAgent } from './api'
import type {
  ActionRequest,
  ChatMessage,
  DecisionType,
  PendingHitl,
  Permission,
  StreamChunk,
  ToolEvent,
} from './types'
import { GENERAL_AGENT_ID } from './types'
import './App.css'

type MicState = 'idle' | 'recording' | 'transcribing'

function pickRecorderMimeType(): string {
  const candidates = [
    'audio/webm;codecs=opus',
    'audio/webm',
    'audio/mp4',
    'audio/ogg',
  ]
  for (const type of candidates) {
    if (
      typeof MediaRecorder !== 'undefined' &&
      MediaRecorder.isTypeSupported(type)
    ) {
      return type
    }
  }
  return ''
}

function extensionForMime(mimeType: string): string {
  if (mimeType.includes('mp4')) return 'mp4'
  if (mimeType.includes('ogg')) return 'ogg'
  return 'webm'
}

function newId(): string {
  return crypto.randomUUID()
}

function formatArgs(args: Record<string, unknown>): string {
  try {
    return JSON.stringify(args, null, 2)
  } catch {
    return String(args)
  }
}

/** Backend surfaces HITL pauses as tool_call_finished.error = "(Interrupt(...),)". */
function isInterruptPayload(value: unknown): boolean {
  if (value == null) return false
  const text = typeof value === 'string' ? value : String(value)
  return /\bInterrupt\s*\(/.test(text)
}

function toolFinishStatus(
  error: unknown,
): 'done' | 'error' | 'interrupt' {
  if (error == null) return 'done'
  if (isInterruptPayload(error)) return 'interrupt'
  return 'error'
}

function dedupeActionRequests(actions: ActionRequest[]): ActionRequest[] {
  const seen = new Set<string>()
  const deduped: ActionRequest[] = []
  for (const action of actions) {
    const key = `${action.name}:${JSON.stringify(action.args ?? {})}`
    if (seen.has(key)) continue
    seen.add(key)
    deduped.push(action)
  }
  return deduped
}

/** Split model content-block payloads into visible text vs thinking. */
function splitReplyBlocks(raw: string): { text: string; reasoning: string } {
  const trimmed = raw.trim()
  if (!trimmed.startsWith('[')) {
    return { text: raw, reasoning: '' }
  }

  try {
    const normalized = trimmed
      .replace(/'/g, '"')
      .replace(/\bNone\b/g, 'null')
      .replace(/\bTrue\b/g, 'true')
      .replace(/\bFalse\b/g, 'false')
    const blocks = JSON.parse(normalized) as Array<Record<string, unknown>>
    if (!Array.isArray(blocks)) {
      return { text: raw, reasoning: '' }
    }

    const textParts: string[] = []
    const reasoningParts: string[] = []
    for (const block of blocks) {
      const type = String(block.type || '')
      if (type === 'reasoning') {
        const value = block.reasoning ?? block.text ?? ''
        if (value) reasoningParts.push(String(value))
      } else if (type === 'text') {
        const value = block.text ?? block.content ?? ''
        if (value) textParts.push(String(value))
      }
    }
    if (textParts.length || reasoningParts.length) {
      return {
        text: textParts.join('\n\n'),
        reasoning: reasoningParts.join('\n\n'),
      }
    }
  } catch {
    // Fall through — treat as plain text.
  }

  return { text: raw, reasoning: '' }
}

function ThinkingBlock({
  reasoning,
  streaming,
}: {
  reasoning: string
  streaming?: boolean
}) {
  return (
    <details className="collapsible" open={streaming || undefined}>
      <summary>
        Thinking
        {streaming ? <span className="collapsible-live">live</span> : null}
      </summary>
      <pre>
        {reasoning}
        {streaming ? <span className="cursor" aria-hidden="true" /> : null}
      </pre>
    </details>
  )
}

function ToolEventBlock({ tool }: { tool: ToolEvent }) {
  const running = tool.status === 'running'
  return (
    <details
      className={`collapsible tool-event tool-${tool.status}`}
      open={running || undefined}
    >
      <summary>
        <span className="tool-summary-name">{tool.name}</span>
        <span className={`tool-status tool-status-${tool.status}`}>
          {tool.status}
        </span>
        {running ? <span className="collapsible-live">live</span> : null}
      </summary>
      {tool.subagentName ? (
        <div className="tool-subagent">via {tool.subagentName}</div>
      ) : null}
      <pre>{formatArgs(tool.args)}</pre>
      {tool.output ? <pre className="tool-output">{tool.output}</pre> : null}
    </details>
  )
}

function HitlPanel({
  pending,
  busy,
  onSubmit,
}: {
  pending: PendingHitl
  busy: boolean
  onSubmit: (permissions: Permission[]) => void
}) {
  const [decisions, setDecisions] = useState<
    Record<number, { decision: DecisionType; note: string }>
  >(() =>
    Object.fromEntries(
      pending.actionRequests.map((_, index) => [
        index,
        { decision: 'approve' as DecisionType, note: '' },
      ]),
    ),
  )

  const setDecision = (index: number, decision: DecisionType) => {
    setDecisions((prev) => ({
      ...prev,
      [index]: { ...prev[index], decision },
    }))
  }

  const setNote = (index: number, note: string) => {
    setDecisions((prev) => ({
      ...prev,
      [index]: { ...prev[index], note },
    }))
  }

  const submit = () => {
    // Backend keys permissions by tool name; one decision covers all
    // pending action_requests with that name.
    const byName = new Map<string, Permission>()
    pending.actionRequests.forEach((action, index) => {
      const choice = decisions[index] ?? { decision: 'approve' as DecisionType, note: '' }
      const permission: Permission = {
        name: action.name,
        decision: choice.decision,
      }
      if (choice.decision === 'edit') {
        permission.edit_instruction = choice.note || null
      } else if (choice.decision === 'respond') {
        permission.respond_instruction = choice.note || null
      } else if (choice.decision === 'reject') {
        permission.reject_reason = choice.note || null
      }
      byName.set(action.name, permission)
    })
    onSubmit([...byName.values()])
  }

  return (
    <section className="hitl-panel" aria-label="Tool approval">
      <header className="hitl-header">
        <h2>Tool permission needed</h2>
        <p>Approve, edit, reject, or respond before the agent continues.</p>
      </header>
      <ul className="hitl-list">
        {pending.actionRequests.map((action, index) => (
          <li key={`${action.name}-${index}`} className="hitl-item">
            <div className="hitl-item-top">
              <span className="tool-pill">{action.name}</span>
              <select
                value={decisions[index]?.decision ?? 'approve'}
                disabled={busy}
                onChange={(event) =>
                  setDecision(index, event.target.value as DecisionType)
                }
              >
                <option value="approve">Approve</option>
                <option value="edit">Edit</option>
                <option value="reject">Reject</option>
                <option value="respond">Respond</option>
              </select>
            </div>
            {action.description ? (
              <p className="hitl-description">{action.description}</p>
            ) : null}
            <pre className="hitl-args">{formatArgs(action.args ?? {})}</pre>
            {decisions[index]?.decision !== 'approve' ? (
              <textarea
                className="hitl-note"
                placeholder={
                  decisions[index]?.decision === 'edit'
                    ? 'Edit instruction…'
                    : decisions[index]?.decision === 'respond'
                      ? 'Response instruction…'
                      : 'Reject reason…'
                }
                value={decisions[index]?.note ?? ''}
                disabled={busy}
                onChange={(event) => setNote(index, event.target.value)}
                rows={2}
              />
            ) : null}
          </li>
        ))}
      </ul>
      <button type="button" className="primary-btn" disabled={busy} onClick={submit}>
        Continue with decisions
      </button>
    </section>
  )
}

function MessageCard({ message }: { message: ChatMessage }) {
  return (
    <article className={`message message-${message.role}`}>
      <div className="message-meta">
        {message.role === 'user'
          ? 'You'
          : message.role === 'assistant'
            ? 'Agent'
            : 'System'}
      </div>
      {message.reasoning ? (
        <ThinkingBlock
          reasoning={message.reasoning}
          streaming={message.reasoningStreaming}
        />
      ) : null}
      {message.tools?.length ? (
        <ul className="tool-list">
          {message.tools.map((tool) => (
            <li key={tool.id}>
              <ToolEventBlock tool={tool} />
            </li>
          ))}
        </ul>
      ) : null}
      {message.content ? (
        <div className="message-body">
          {message.content}
          {message.streaming ? <span className="cursor" aria-hidden="true" /> : null}
        </div>
      ) : message.streaming ? (
        <div className="message-body muted">
          Streaming
          <span className="cursor" aria-hidden="true" />
        </div>
      ) : null}
    </article>
  )
}

export default function App() {
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [input, setInput] = useState('')
  const [threadId, setThreadId] = useState<string | null>(null)
  const [streaming, setStreaming] = useState(false)
  const [pendingHitl, setPendingHitl] = useState<PendingHitl | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [micState, setMicState] = useState<MicState>('idle')
  const abortRef = useRef<AbortController | null>(null)
  const bottomRef = useRef<HTMLDivElement | null>(null)
  const assistantIdRef = useRef<string | null>(null)
  const mediaRecorderRef = useRef<MediaRecorder | null>(null)
  const mediaStreamRef = useRef<MediaStream | null>(null)
  const audioChunksRef = useRef<Blob[]>([])

  const stopMediaTracks = useCallback(() => {
    mediaStreamRef.current?.getTracks().forEach((track) => track.stop())
    mediaStreamRef.current = null
    mediaRecorderRef.current = null
    audioChunksRef.current = []
  }, [])

  useEffect(() => {
    return () => {
      mediaRecorderRef.current?.stop()
      stopMediaTracks()
    }
  }, [stopMediaTracks])

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, pendingHitl, streaming])

  const updateAssistant = useCallback((updater: (message: ChatMessage) => ChatMessage) => {
    const id = assistantIdRef.current
    if (!id) return
    setMessages((prev) =>
      prev.map((message) => (message.id === id ? updater(message) : message)),
    )
  }, [])

  const handleChunk = useCallback(
    (chunk: StreamChunk) => {
      if (chunk.thread_id) {
        setThreadId(chunk.thread_id)
      }

      if (chunk.kind === 'text' && chunk.delta) {
        updateAssistant((message) => ({
          ...message,
          content: message.content + chunk.delta!,
          streaming: true,
          reasoningStreaming: false,
        }))
        return
      }

      if (chunk.kind === 'reasoning' && chunk.delta) {
        updateAssistant((message) => ({
          ...message,
          reasoning: (message.reasoning ?? '') + chunk.delta!,
          reasoningStreaming: true,
        }))
        return
      }

      if (chunk.kind === 'tool_call_started') {
        const tool: ToolEvent = {
          id: chunk.tool_call_id || newId(),
          name: chunk.tool_name || 'tool',
          args: chunk.input ?? {},
          status: 'running',
          source: chunk.source,
          subagentName: chunk.subagent_name,
        }
        updateAssistant((message) => ({
          ...message,
          tools: [...(message.tools ?? []), tool],
        }))
        return
      }

      if (chunk.kind === 'tool_call_finished') {
        updateAssistant((message) => ({
          ...message,
          tools: (message.tools ?? []).map((tool) => {
            if (chunk.tool_call_id && tool.id !== chunk.tool_call_id) {
              if (tool.name !== chunk.tool_name || tool.status !== 'running') {
                return tool
              }
            } else if (!chunk.tool_call_id && tool.name !== chunk.tool_name) {
              return tool
            }
            return {
              ...tool,
              status: toolFinishStatus(chunk.error),
              output:
                chunk.error != null
                  ? String(chunk.error)
                  : chunk.output != null
                    ? typeof chunk.output === 'string'
                      ? chunk.output
                      : JSON.stringify(chunk.output, null, 2)
                    : tool.output,
            }
          }),
        }))
        return
      }

      if (chunk.kind === 'subagent_started' && chunk.subagent_name) {
        updateAssistant((message) => ({
          ...message,
          content:
            message.content +
            (message.content ? '\n\n' : '') +
            `[${chunk.subagent_name}] `,
        }))
        return
      }

      if (chunk.kind === 'interrupt') {
        const actionRequests = dedupeActionRequests(
          (chunk.action_requests ?? []) as ActionRequest[],
        )
        setPendingHitl({
          actionRequests,
          interruptIds: chunk.interrupt_ids ?? [],
        })
        return
      }

      if (chunk.kind === 'message_finished') {
        updateAssistant((message) => {
          const fromContent = chunk.content
            ? splitReplyBlocks(chunk.content)
            : { text: '', reasoning: '' }
          const reasoning =
            message.reasoning ||
            chunk.reasoning_content ||
            fromContent.reasoning ||
            ''
          const content =
            message.content ||
            fromContent.text ||
            (chunk.content && !fromContent.reasoning ? chunk.content : '') ||
            ''
          return {
            ...message,
            content,
            reasoning: reasoning || undefined,
            reasoningStreaming: false,
          }
        })
        return
      }

      if (chunk.kind === 'run_finished') {
        if (chunk.reply) {
          updateAssistant((message) => {
            const split = splitReplyBlocks(chunk.reply || '')
            return {
              ...message,
              content: message.content || split.text || '',
              reasoning:
                message.reasoning ||
                chunk.reasoning_content ||
                split.reasoning ||
                undefined,
              reasoningStreaming: false,
            }
          })
        } else {
          updateAssistant((message) => ({
            ...message,
            reasoningStreaming: false,
          }))
        }
        if (chunk.status === 'awaiting_tool_permission') {
          setPendingHitl({
            actionRequests: dedupeActionRequests(
              (chunk.action_requests ?? []) as ActionRequest[],
            ),
            interruptIds: chunk.interrupt_ids ?? [],
          })
        } else if (chunk.status === 'cancelled') {
          setMessages((prev) => [
            ...prev,
            {
              id: newId(),
              role: 'system',
              content: 'Stream cancelled.',
            },
          ])
        }
        updateAssistant((message) => ({
          ...message,
          streaming: false,
          reasoningStreaming: false,
        }))
      }
    },
    [updateAssistant],
  )

  const runStream = useCallback(
    async ({
      message,
      permissions,
      existingThreadId,
    }: {
      message?: string
      permissions?: Permission[]
      existingThreadId?: string | null
    }) => {
      setError(null)
      setPendingHitl(null)
      setStreaming(true)

      const activeThreadId = existingThreadId ?? threadId ?? newId()
      if (!threadId) {
        setThreadId(activeThreadId)
      }

      const assistantId = newId()
      assistantIdRef.current = assistantId
      setMessages((prev) => [
        ...prev,
        {
          id: assistantId,
          role: 'assistant',
          content: '',
          streaming: true,
          tools: [],
        },
      ])

      const controller = new AbortController()
      abortRef.current = controller

      try {
        await streamAgent({
          agentId: GENERAL_AGENT_ID,
          threadId: activeThreadId,
          message,
          permissions,
          signal: controller.signal,
          onChunk: handleChunk,
        })
      } catch (err) {
        if ((err as Error).name === 'AbortError') {
          setMessages((prev) => [
            ...prev,
            { id: newId(), role: 'system', content: 'Stopped.' },
          ])
        } else {
          setError((err as Error).message || 'Stream failed')
          updateAssistant((message) => ({
            ...message,
            streaming: false,
            content: message.content || 'Something went wrong while streaming.',
          }))
        }
      } finally {
        abortRef.current = null
        setStreaming(false)
        updateAssistant((message) => ({ ...message, streaming: false }))
      }
    },
    [handleChunk, threadId, updateAssistant],
  )

  const sendText = useCallback(
    async (text: string) => {
      const trimmed = text.trim()
      if (!trimmed || streaming) return
      setInput('')
      setMessages((prev) => [
        ...prev,
        { id: newId(), role: 'user', content: trimmed },
      ])
      await runStream({ message: trimmed })
    },
    [runStream, streaming],
  )

  const onSend = async () => {
    await sendText(input)
  }

  const startRecording = async () => {
    if (streaming || pendingHitl || micState !== 'idle') return
    setError(null)

    if (
      typeof navigator === 'undefined' ||
      !navigator.mediaDevices?.getUserMedia ||
      typeof MediaRecorder === 'undefined'
    ) {
      setError('Microphone recording is not supported in this browser.')
      return
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      mediaStreamRef.current = stream
      audioChunksRef.current = []

      const mimeType = pickRecorderMimeType()
      const recorder = mimeType
        ? new MediaRecorder(stream, { mimeType })
        : new MediaRecorder(stream)

      recorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data)
        }
      }

      recorder.onerror = () => {
        setError('Recording failed.')
        setMicState('idle')
        stopMediaTracks()
      }

      mediaRecorderRef.current = recorder
      recorder.start(250)
      setMicState('recording')
    } catch {
      setError('Microphone permission denied or unavailable.')
      stopMediaTracks()
      setMicState('idle')
    }
  }

  const stopRecordingAndSend = async () => {
    const recorder = mediaRecorderRef.current
    if (!recorder || micState !== 'recording') return

    setMicState('transcribing')

    const blob = await new Promise<Blob>((resolve, reject) => {
      recorder.onstop = () => {
        const type = recorder.mimeType || 'audio/webm'
        resolve(new Blob(audioChunksRef.current, { type }))
      }
      recorder.onerror = () => reject(new Error('Recording failed.'))
      try {
        recorder.stop()
      } catch (err) {
        reject(err instanceof Error ? err : new Error('Recording failed.'))
      }
    }).catch((err: Error) => {
      setError(err.message)
      setMicState('idle')
      stopMediaTracks()
      return null
    })

    stopMediaTracks()
    if (!blob || blob.size === 0) {
      if (blob) setError('Recording was empty.')
      setMicState('idle')
      return
    }

    try {
      const filename = `recording.${extensionForMime(blob.type)}`
      const result = await speechToText(blob, filename)
      const text = result.text?.trim()
      setMicState('idle')
      if (!text) {
        setError('No speech detected. Try again.')
        return
      }
      await sendText(text)
    } catch (err) {
      setError((err as Error).message || 'Speech-to-text failed')
      setMicState('idle')
    }
  }

  const onMicClick = () => {
    if (micState === 'recording') {
      void stopRecordingAndSend()
      return
    }
    void startRecording()
  }

  const onStop = async () => {
    const active = threadId
    abortRef.current?.abort()
    if (active) {
      try {
        await cancelStream(active)
      } catch (err) {
        setError((err as Error).message)
      }
    }
  }

  const onNewChat = () => {
    abortRef.current?.abort()
    if (mediaRecorderRef.current?.state === 'recording') {
      mediaRecorderRef.current.stop()
    }
    stopMediaTracks()
    setMicState('idle')
    setMessages([])
    setThreadId(null)
    setPendingHitl(null)
    setError(null)
    setInput('')
    assistantIdRef.current = null
  }

  return (
    <div className="app-shell">
      <div className="atmosphere" aria-hidden="true" />
      <header className="topbar">
        <div className="brand-block">
          <p className="brand">Deep Agents</p>
          <p className="brand-sub">Streamed chat with human-in-the-loop tools</p>
        </div>
        <div className="topbar-meta">
          <span className="thread-chip">
            {threadId ? `thread ${threadId.slice(0, 8)}…` : 'new thread'}
          </span>
          <button type="button" className="ghost-btn" onClick={onNewChat}>
            New chat
          </button>
        </div>
      </header>

      <main className="chat-main">
        {messages.length === 0 ? (
          <section className="empty-state">
            <h1>Ask anything</h1>
            <p>
              Messages stream from <code>/stream</code>. When a tool needs approval,
              decide here and the same thread continues.
            </p>
          </section>
        ) : (
          <div className="message-list">
            {messages.map((message) => (
              <MessageCard key={message.id} message={message} />
            ))}
            <div ref={bottomRef} />
          </div>
        )}

        {pendingHitl ? (
          <HitlPanel
            pending={pendingHitl}
            busy={streaming}
            onSubmit={(permissions) =>
              runStream({ permissions, existingThreadId: threadId })
            }
          />
        ) : null}

        {error ? <p className="error-banner">{error}</p> : null}
      </main>

      <footer className="composer">
        <textarea
          value={input}
          onChange={(event) => setInput(event.target.value)}
          rows={2}
          disabled={
            streaming || Boolean(pendingHitl) || micState !== 'idle'
          }
          placeholder={
            micState === 'recording'
              ? 'Listening… click the mic to stop'
              : micState === 'transcribing'
                ? 'Transcribing…'
                : pendingHitl
                  ? 'Resolve tool permission above, or start a new chat…'
                  : 'Message the agent…'
          }
          onKeyDown={(event) => {
            if (event.key === 'Enter' && !event.shiftKey) {
              event.preventDefault()
              void onSend()
            }
          }}
        />
        <div className="composer-actions">
          <button
            type="button"
            className={`mic-btn${micState === 'recording' ? ' mic-recording' : ''}`}
            aria-label={
              micState === 'recording'
                ? 'Stop recording'
                : micState === 'transcribing'
                  ? 'Transcribing'
                  : 'Start voice message'
            }
            aria-pressed={micState === 'recording'}
            disabled={
              streaming ||
              Boolean(pendingHitl) ||
              micState === 'transcribing'
            }
            onClick={onMicClick}
          >
            {micState === 'transcribing' ? (
              '…'
            ) : (
              <svg
                viewBox="0 0 24 24"
                width="18"
                height="18"
                aria-hidden="true"
                fill="currentColor"
              >
                <path d="M12 14a3 3 0 0 0 3-3V6a3 3 0 1 0-6 0v5a3 3 0 0 0 3 3Zm5-3a5 5 0 0 1-10 0H5a7 7 0 0 0 6 6.92V21h2v-3.08A7 7 0 0 0 19 11h-2Z" />
              </svg>
            )}
          </button>
          {streaming ? (
            <button type="button" className="stop-btn" onClick={() => void onStop()}>
              Stop
            </button>
          ) : (
            <button
              type="button"
              className="primary-btn"
              disabled={
                !input.trim() ||
                Boolean(pendingHitl) ||
                micState !== 'idle'
              }
              onClick={() => void onSend()}
            >
              Send
            </button>
          )}
        </div>
      </footer>
    </div>
  )
}
