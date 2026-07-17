'use client'

import { useCallback, useEffect, useRef, useState, type CSSProperties } from 'react'
import { cancelStream, clearFromLastUser, getHistory, speechToText, streamAgent, ThreadNotFoundError } from '@/api'
import type {
  ActionRequest,
  ChatMessage,
  DecisionType,
  PendingHitl,
  Permission,
  Source,
  StreamChunk,
  ThreadHistoryResponse,
  ToolEvent,
} from '@/types'
import { GENERAL_AGENT_ID } from '@/types'

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

function threadIdFromLocation(): string | null {
  if (typeof window === 'undefined') return null
  const value = new URLSearchParams(window.location.search).get('threadId')?.trim()
  return value || null
}

function syncThreadIdInLocation(threadId: string | null) {
  if (typeof window === 'undefined') return
  const url = new URL(window.location.href)
  if (threadId) {
    url.searchParams.set('threadId', threadId)
  } else {
    url.searchParams.delete('threadId')
  }
  const next = `${url.pathname}${url.search}${url.hash}`
  const current = `${window.location.pathname}${window.location.search}${window.location.hash}`
  if (next !== current) {
    window.history.replaceState(null, '', next)
  }
}

function historyToChatMessages(history: ThreadHistoryResponse): ChatMessage[] {
  return history.messages.map((message) => ({
    id: message.id,
    role: message.role,
    content: message.content,
    reasoning: message.reasoning || undefined,
    sources: message.sources?.length ? message.sources : undefined,
    tools: message.tools?.map((tool) => ({
      id: tool.id,
      name: tool.name,
      args: tool.args ?? {},
      status:
        tool.status === 'interrupt' || isInterruptPayload(tool.output)
          ? 'interrupt'
          : tool.status === 'error'
            ? 'error'
            : 'done',
      output: tool.output || undefined,
      subagentName: tool.subagent_name,
    })),
  }))
}

function formatArgs(args: Record<string, unknown>): string {
  try {
    return JSON.stringify(args, null, 2)
  } catch {
    return String(args)
  }
}

function stripInlineSources(text: string): string {
  return text
    .replace(/\s*\(\s*sources?\s*:[^)]+\)/gi, '')
    .replace(/\n{0,2}\*{0,2}Sources?\*{0,2}\s*(\([^)]*\)\s*)?:[\s\S]*$/i, '')
    .replace(/[ \t]+\n/g, '\n')
    .replace(/\n{3,}/g, '\n\n')
    .trimEnd()
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

function mergeSources(
  existing: Source[] | undefined,
  incoming: Source[] | null | undefined,
): Source[] | undefined {
  if (!incoming?.length) return existing
  const byUrl = new Map<string, Source>()
  for (const source of existing ?? []) {
    if (source.url) byUrl.set(source.url, source)
  }
  for (const source of incoming) {
    if (!source.url) continue
    byUrl.set(source.url, source)
  }
  const merged = [...byUrl.values()]
  return merged.length ? merged : existing
}

/** Sources from all assistant messages after the latest user message. */
function sourcesSinceLastUser(messages: ChatMessage[]): Source[] {
  let start = 0
  for (let i = messages.length - 1; i >= 0; i -= 1) {
    if (messages[i]?.role === 'user') {
      start = i + 1
      break
    }
  }
  let merged: Source[] | undefined
  for (let i = start; i < messages.length; i += 1) {
    const message = messages[i]
    if (message?.role === 'assistant') {
      merged = mergeSources(merged, message.sources)
    }
  }
  return merged ?? []
}

function sourcesForTurn(messages: ChatMessage[], messageId: string): Source[] {
  const index = messages.findIndex((message) => message.id === messageId)
  if (index < 0) return []
  return sourcesSinceLastUser(messages.slice(0, index + 1))
}

/** Favicon via Google's domain service — avoids cross-site fetches to publisher URLs. */
function faviconForSource(source: Source): string {
  try {
    const host = new URL(source.url).hostname
    return `https://www.google.com/s2/favicons?domain=${encodeURIComponent(host)}&sz=64`
  } catch {
    return `https://www.google.com/s2/favicons?domain=example.com&sz=64`
  }
}

function SourceFavicon({
  source,
  width,
  height,
  className,
  style,
}: {
  source: Source
  width: number
  height: number
  className?: string
  style?: CSSProperties
}) {
  const [failed, setFailed] = useState(false)
  if (failed) return null
  return (
    <img
      className={className}
      src={faviconForSource(source)}
      alt=""
      width={width}
      height={height}
      loading="lazy"
      referrerPolicy="no-referrer"
      style={style}
      onError={() => setFailed(true)}
    />
  )
}

function sourceHostname(source: Source): string {
  try {
    return new URL(source.url).hostname.replace(/^www\./, '')
  } catch {
    return 'source'
  }
}

function formatPublishedDate(value: string | null | undefined): string | null {
  if (!value) return null
  const parsed = new Date(value)
  if (Number.isNaN(parsed.getTime())) {
    const match = value.match(/(\d{4})[/-](\d{1,2})[/-](\d{1,2})/)
    if (match) {
      return `${match[1]}/${match[2].padStart(2, '0')}/${match[3].padStart(2, '0')}`
    }
    return value
  }
  const y = parsed.getUTCFullYear()
  const m = String(parsed.getUTCMonth() + 1).padStart(2, '0')
  const d = String(parsed.getUTCDate()).padStart(2, '0')
  return `${y}/${m}/${d}`
}

function SourcesPill({
  sources,
  onOpen,
}: {
  sources: Source[]
  onOpen: (sources: Source[]) => void
}) {
  const preview = sources.slice(0, 3)
  const label = `${sources.length} source${sources.length === 1 ? '' : 's'}`

  return (
    <button
      type="button"
      className="sources-pill"
      onClick={() => onOpen(sources)}
      title="View search results"
    >
      <span className="sources-icons" aria-hidden="true">
        {preview.map((source, index) => (
          <SourceFavicon
            key={`${source.url}-${index}`}
            className="sources-icon"
            source={source}
            width={20}
            height={20}
            style={{ zIndex: preview.length - index }}
          />
        ))}
      </span>
      <span className="sources-label">{label}</span>
    </button>
  )
}

function SourcesDrawer({
  sources,
  onClose,
}: {
  sources: Source[]
  onClose: () => void
}) {
  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (event.key === 'Escape') onClose()
    }
    window.addEventListener('keydown', onKey)
    const previous = document.body.style.overflow
    document.body.style.overflow = 'hidden'
    return () => {
      window.removeEventListener('keydown', onKey)
      document.body.style.overflow = previous
    }
  }, [onClose])

  return (
    <div className="sources-drawer-root">
      <button
        type="button"
        className="sources-drawer-backdrop"
        aria-label="Close search results"
        onClick={onClose}
      />
      <aside className="sources-drawer" role="dialog" aria-label="Search results">
        <header className="sources-drawer-header">
          <h2>Search results</h2>
          <button
            type="button"
            className="sources-drawer-close"
            aria-label="Close"
            onClick={onClose}
          >
            ×
          </button>
        </header>
        <ul className="sources-drawer-list">
          {sources.map((source, index) => {
            const published = formatPublishedDate(source.published_date)
            const snippet = (source.content || '').trim()
            return (
              <li key={`${source.url}-${index}`}>
                <a
                  className="sources-drawer-item"
                  href={source.url}
                  target="_blank"
                  rel="noreferrer"
                >
                  <div className="sources-drawer-meta">
                    <span className="sources-drawer-site">
                      <SourceFavicon source={source} width={16} height={16} />
                      <span>{sourceHostname(source)}</span>
                      {published ? (
                        <>
                          <span className="sources-drawer-sep" aria-hidden="true">
                            |
                          </span>
                          <span>{published}</span>
                        </>
                      ) : null}
                    </span>
                    <span className="sources-drawer-index">{index + 1}</span>
                  </div>
                  <div className="sources-drawer-title">{source.title}</div>
                  {snippet ? (
                    <p className="sources-drawer-snippet">{snippet}</p>
                  ) : null}
                </a>
              </li>
            )
          })}
        </ul>
      </aside>
    </div>
  )
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

/** Convert Python-ish list/dict literals to JSON (handles nested quotes). */
function pythonLiteralToJson(input: string): string {
  let result = ''
  let i = 0
  while (i < input.length) {
    const char = input[i]

    if (char === "'" || char === '"') {
      const quote = char
      let out = '"'
      i += 1
      while (i < input.length) {
        const ch = input[i]
        if (ch === '\\' && i + 1 < input.length) {
          const next = input[i + 1]
          if (next === 'n') out += '\\n'
          else if (next === 't') out += '\\t'
          else if (next === 'r') out += '\\r'
          else if (next === '\\') out += '\\\\'
          else if (next === quote) out += quote === '"' ? '\\"' : "'"
          else if (next === '"') out += '\\"'
          else out += next
          i += 2
          continue
        }
        if (ch === quote) {
          out += '"'
          i += 1
          break
        }
        if (ch === '"') {
          out += '\\"'
          i += 1
          continue
        }
        if (ch === '\n') {
          out += '\\n'
          i += 1
          continue
        }
        out += ch
        i += 1
      }
      result += out
      continue
    }

    if (/\d/.test(char) || char === '-') {
      const start = i
      i += 1
      while (i < input.length && /[\d.]/.test(input[i]!)) i += 1
      result += input.slice(start, i)
      continue
    }

    if (/[A-Za-z_]/.test(char)) {
      const start = i
      i += 1
      while (i < input.length && /[A-Za-z0-9_]/.test(input[i]!)) i += 1
      const word = input.slice(start, i)
      if (word === 'None') result += 'null'
      else if (word === 'True') result += 'true'
      else if (word === 'False') result += 'false'
      else result += `"${word}"`
      continue
    }

    result += char
    i += 1
  }
  return result
}

function parseContentBlocks(
  raw: string,
): Array<Record<string, unknown>> | null {
  const trimmed = raw.trim()
  if (!trimmed.startsWith('[')) return null

  const candidates = [trimmed, pythonLiteralToJson(trimmed)]
  for (const candidate of candidates) {
    try {
      const parsed = JSON.parse(candidate) as unknown
      if (Array.isArray(parsed)) {
        return parsed as Array<Record<string, unknown>>
      }
    } catch {
      // try next candidate
    }
  }
  return null
}

/** Split model content-block payloads into visible text vs thinking. */
function splitReplyBlocks(raw: string): { text: string; reasoning: string } {
  if (!raw.trim()) {
    return { text: raw, reasoning: '' }
  }

  const blocks = parseContentBlocks(raw)
  if (!blocks) {
    return { text: raw, reasoning: '' }
  }

  const textParts: string[] = []
  const reasoningParts: string[] = []
  let sawKnown = false
  for (const block of blocks) {
    const type = String(block.type || '')
    if (type === 'reasoning') {
      sawKnown = true
      const value = block.reasoning ?? block.text ?? ''
      if (value) reasoningParts.push(String(value))
    } else if (type === 'text') {
      sawKnown = true
      const value = block.text ?? block.content ?? ''
      if (value) textParts.push(String(value))
    } else if (type === 'tool_call') {
      sawKnown = true
    }
  }

  if (!sawKnown) {
    return { text: raw, reasoning: '' }
  }

  return {
    text: textParts.join('\n\n'),
    reasoning: reasoningParts.join('\n\n'),
  }
}

/** True when an assistant bubble has something to show (or is still streaming). */
function hasRenderableContent(message: ChatMessage): boolean {
  if (message.role !== 'assistant') return true
  if (message.streaming || message.reasoningStreaming) return true

  const split = splitReplyBlocks(message.content)
  const parsedAsBlocks = parseContentBlocks(message.content) != null
  const reasoning = (message.reasoning || split.reasoning || '').trim()
  const bodyText = (parsedAsBlocks ? split.text : message.content).trim()

  if (bodyText) return true
  if (reasoning) return true
  if (message.tools?.length) return true
  if (message.statusLines?.length) return true
  return false
}

function appendStreamStatus(
  message: ChatMessage,
  line: string,
): ChatMessage {
  if (!message.streaming) {
    return message
  }
  const statusLines = message.statusLines ?? []
  if (statusLines.includes(line)) {
    return message
  }
  return { ...message, statusLines: [...statusLines, line] }
}

function StreamStatusBlock({ lines }: { lines: string[] }) {
  return (
    <ul className="stream-status-list" aria-label="Agent activity">
      {lines.map((line) => (
        <li key={line} className="stream-status-item">
          <span className="stream-status-label">System</span>
          <span className="stream-status-text">{line}</span>
        </li>
      ))}
    </ul>
  )
}

function formatToolLabel(name: string): string {
  return name
    .split(/[_-]+/)
    .filter(Boolean)
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(' ')
}

function ThinkingBlock({
  reasoning,
  streaming,
}: {
  reasoning: string
  streaming?: boolean
}) {
  return (
    <details className="collapsible" open={Boolean(streaming)}>
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
      key={running ? `${tool.id}-live` : tool.id}
      className={`collapsible tool-event tool-${tool.status}`}
      open={running}
    >
      <summary>
        <span className="tool-summary-name">{formatToolLabel(tool.name)}</span>
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

function MessageActions({
  copyText,
  onRegenerate,
  canRegenerate,
}: {
  copyText: string
  onRegenerate?: () => void
  canRegenerate?: boolean
}) {
  const [copied, setCopied] = useState(false)

  const onCopy = async () => {
    const text = copyText.trim()
    if (!text) return
    try {
      await navigator.clipboard.writeText(text)
      setCopied(true)
      window.setTimeout(() => setCopied(false), 1500)
    } catch {
      setCopied(false)
    }
  }

  return (
    <div className="message-actions" role="group" aria-label="Message actions">
      <button
        type="button"
        className="message-action-btn"
        onClick={() => void onCopy()}
        disabled={!copyText.trim()}
        title={copied ? 'Copied' : 'Copy'}
        aria-label={copied ? 'Copied' : 'Copy reply'}
      >
        {copied ? (
          <svg viewBox="0 0 24 24" width="16" height="16" aria-hidden="true" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M5 13l4 4L19 7" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
        ) : (
          <svg viewBox="0 0 24 24" width="16" height="16" aria-hidden="true" fill="none" stroke="currentColor" strokeWidth="2">
            <rect x="9" y="9" width="11" height="11" rx="2" />
            <path d="M5 15V5a2 2 0 0 1 2-2h10" strokeLinecap="round" />
          </svg>
        )}
      </button>
      {canRegenerate && onRegenerate ? (
        <button
          type="button"
          className="message-action-btn"
          onClick={onRegenerate}
          title="Regenerate"
          aria-label="Regenerate reply"
        >
          <svg viewBox="0 0 24 24" width="16" height="16" aria-hidden="true" fill="none" stroke="currentColor" strokeWidth="2">
            <path
              d="M4 12a8 8 0 0 1 13.66-5.66M20 4v5h-5M20 12a8 8 0 0 1-13.66 5.66M4 20v-5h5"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          </svg>
        </button>
      ) : null}
    </div>
  )
}

function MessageCard({
  message,
  turnSources,
  onOpenSources,
  onRegenerate,
  showActions,
}: {
  message: ChatMessage
  turnSources?: Source[]
  onOpenSources: (sources: Source[]) => void
  onRegenerate?: () => void
  showActions?: boolean
}) {
  const sources = turnSources?.length ? turnSources : message.sources
  const split = splitReplyBlocks(message.content)
  const parsedAsBlocks = parseContentBlocks(message.content) != null
  const reasoning = message.reasoning || split.reasoning || undefined
  const bodyText = parsedAsBlocks ? split.text : message.content
  const copyText = stripInlineSources(bodyText || reasoning || '')
  const showFooter =
    !message.streaming && message.role === 'assistant' && Boolean(showActions)

  return (
    <article className={`message message-${message.role}`}>
      <div className="message-meta">
        {message.role === 'user'
          ? 'You'
          : message.role === 'assistant'
            ? 'Agent'
            : 'System'}
      </div>
      {message.statusLines?.length &&
      (message.streaming || !bodyText.trim()) ? (
        <StreamStatusBlock lines={message.statusLines} />
      ) : null}
      {reasoning ? (
        <ThinkingBlock
          reasoning={reasoning}
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
      {bodyText ? (
        <div className="message-body">
          {stripInlineSources(bodyText)}
          {message.streaming ? <span className="cursor" aria-hidden="true" /> : null}
        </div>
      ) : message.streaming && !message.statusLines?.length ? (
        <div className="message-body muted">
          Streaming
          <span className="cursor" aria-hidden="true" />
        </div>
      ) : null}
      {showFooter ? (
        <div className="message-footer">
          <MessageActions
            copyText={copyText}
            onRegenerate={onRegenerate}
            canRegenerate={Boolean(onRegenerate)}
          />
          {sources?.length ? (
            <SourcesPill sources={sources} onOpen={onOpenSources} />
          ) : null}
        </div>
      ) : null}
    </article>
  )
}

export default function ChatClient() {
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [input, setInput] = useState('')
  const [threadId, setThreadId] = useState<string | null>(() =>
    threadIdFromLocation(),
  )
  const [streaming, setStreaming] = useState(false)
  const [loadingHistory, setLoadingHistory] = useState(false)
  const [pendingHitl, setPendingHitl] = useState<PendingHitl | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [micState, setMicState] = useState<MicState>('idle')
  const [drawerSources, setDrawerSources] = useState<Source[] | null>(null)
  const abortRef = useRef<AbortController | null>(null)
  const bottomRef = useRef<HTMLDivElement | null>(null)
  const assistantIdRef = useRef<string | null>(null)
  const mediaRecorderRef = useRef<MediaRecorder | null>(null)
  const mediaStreamRef = useRef<MediaStream | null>(null)
  const audioChunksRef = useRef<Blob[]>([])
  const historyLoadedRef = useRef<string | null>(null)

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

  useEffect(() => {
    syncThreadIdInLocation(threadId)
  }, [threadId])

  useEffect(() => {
    const initialThreadId = threadIdFromLocation()
    if (!initialThreadId) {
      setLoadingHistory(false)
      return
    }
    if (historyLoadedRef.current === initialThreadId) return

    let cancelled = false
    setLoadingHistory(true)
    setError(null)

    void getHistory(initialThreadId, GENERAL_AGENT_ID)
      .then((history) => {
        if (cancelled) return
        historyLoadedRef.current = initialThreadId
        setThreadId(history.thread_id)
        setMessages(historyToChatMessages(history))
        if (
          history.status === 'awaiting_tool_permission' &&
          history.action_requests?.length
        ) {
          setPendingHitl({
            actionRequests: history.action_requests,
            interruptIds: history.interrupt_ids ?? [],
          })
        } else {
          setPendingHitl(null)
        }
      })
      .catch((err: Error) => {
        if (cancelled) return
        if (err instanceof ThreadNotFoundError) {
          historyLoadedRef.current = initialThreadId
          setThreadId(initialThreadId)
          setMessages([])
          setPendingHitl(null)
          return
        }
        setError(err.message || 'Failed to load thread history')
        setMessages([])
        setPendingHitl(null)
      })
      .finally(() => {
        if (!cancelled) setLoadingHistory(false)
      })

    return () => {
      cancelled = true
    }
  }, [])

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
        historyLoadedRef.current = chunk.thread_id
      }

      if (chunk.kind === 'system' && chunk.content) {
        updateAssistant((message) =>
          appendStreamStatus(message, String(chunk.content)),
        )
        return
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

      if (chunk.kind === 'tool_call_output_delta' && chunk.delta) {
        const status = String(chunk.delta).trim()
        if (!status) return
        updateAssistant((message) => appendStreamStatus(message, status))
        return
      }

      if (chunk.kind === 'tool_call_finished') {
        updateAssistant((message) => ({
          ...message,
          sources: mergeSources(message.sources, chunk.sources),
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
              sources: mergeSources(message.sources, chunk.sources),
              reasoningStreaming: false,
            }
          })
        } else {
          updateAssistant((message) => ({
            ...message,
            sources: mergeSources(message.sources, chunk.sources),
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
          sources: mergeSources(message.sources, chunk.sources),
          streaming: false,
          reasoningStreaming: false,
          statusLines: undefined,
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
      historyLoadedRef.current = activeThreadId

      const assistantId = newId()
      assistantIdRef.current = assistantId
      setMessages((prev) => {
        // HITL resume creates a new assistant bubble — keep sources from this turn.
        const seededSources = permissions ? sourcesSinceLastUser(prev) : undefined
        return [
          ...prev,
          {
            id: assistantId,
            role: 'assistant',
            content: '',
            streaming: true,
            tools: [],
            sources: seededSources?.length ? seededSources : undefined,
          },
        ]
      })

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
        const assistantId = assistantIdRef.current
        setMessages((prev) =>
          prev
            .map((message) =>
              message.id === assistantId
                ? {
                    ...message,
                    streaming: false,
                    reasoningStreaming: false,
                  }
                : message,
            )
            .filter(hasRenderableContent),
        )
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

  const onRegenerate = useCallback(async () => {
    if (streaming || micState !== 'idle') return

    let lastUserIndex = -1
    for (let i = messages.length - 1; i >= 0; i -= 1) {
      if (messages[i]?.role === 'user') {
        lastUserIndex = i
        break
      }
    }
    if (lastUserIndex < 0) return

    const prompt = messages[lastUserIndex]?.content?.trim()
    if (!prompt) return

    setError(null)
    setPendingHitl(null)
    setDrawerSources(null)
    setMessages((prev) => prev.slice(0, lastUserIndex + 1))

    const activeThreadId = threadId
    if (activeThreadId) {
      try {
        if (abortRef.current) {
          abortRef.current.abort()
          abortRef.current = null
          await cancelStream(activeThreadId).catch(() => undefined)
        }
        await clearFromLastUser(activeThreadId, GENERAL_AGENT_ID)
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to prepare regenerate')
        return
      }
    }

    await runStream({
      message: prompt,
      existingThreadId: activeThreadId,
    })
  }, [messages, micState, runStream, streaming, threadId])

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
    historyLoadedRef.current = null
    setPendingHitl(null)
    setError(null)
    setInput('')
    setLoadingHistory(false)
    assistantIdRef.current = null
    syncThreadIdInLocation(null)
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
        {loadingHistory ? (
          <section className="empty-state">
            <h1>Loading thread…</h1>
            <p>Restoring conversation from checkpoint history.</p>
          </section>
        ) : messages.length === 0 ? (
          <section className="empty-state">
            <h1>Ask anything</h1>
            <p>
              Messages stream from <code>/chats/stream</code>. When a tool needs approval,
              decide here and the same thread continues.
            </p>
          </section>
        ) : (
          <div className="message-list">
            {messages.filter(hasRenderableContent).map((message, index, visible) => {
              const isLastAssistant =
                message.role === 'assistant' &&
                visible.slice(index + 1).every((item) => item.role !== 'assistant')
              return (
                <MessageCard
                  key={message.id}
                  message={message}
                  onOpenSources={setDrawerSources}
                  showActions={isLastAssistant && !message.streaming}
                  onRegenerate={
                    isLastAssistant && !message.streaming
                      ? () => void onRegenerate()
                      : undefined
                  }
                  turnSources={
                    isLastAssistant && !message.streaming
                      ? sourcesForTurn(messages, message.id)
                      : undefined
                  }
                />
              )
            })}
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
            loadingHistory ||
            streaming ||
            Boolean(pendingHitl) ||
            micState !== 'idle'
          }
          placeholder={
            loadingHistory
              ? 'Loading thread history…'
              : micState === 'recording'
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
              loadingHistory ||
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
                loadingHistory ||
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

      {drawerSources?.length ? (
        <SourcesDrawer
          sources={drawerSources}
          onClose={() => setDrawerSources(null)}
        />
      ) : null}
    </div>
  )
}
