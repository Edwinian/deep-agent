/** Shared types for Deep Agents stream + HITL client. */

export const GENERAL_AGENT_ID = 1002

export type DecisionType = 'approve' | 'edit' | 'reject' | 'respond'

export type Permission = {
  name: string
  decision: DecisionType
  edit_instruction?: string | null
  respond_instruction?: string | null
  reject_reason?: string | null
}

export type ActionRequest = {
  name: string
  args: Record<string, unknown>
  description?: string
}

export type StreamMessageKind =
  | 'text'
  | 'reasoning'
  | 'message_tool_call_chunk'
  | 'message_tool_calls_finalized'
  | 'tool_call_started'
  | 'tool_call_output_delta'
  | 'tool_call_finished'
  | 'subagent_started'
  | 'subagent_finished'
  | 'message_finished'
  | 'run_finished'
  | 'interrupt'

export type StreamChunk = {
  thread_id: string
  agent_id: number
  kind: StreamMessageKind
  source?: 'agent' | 'subagent'
  subagent_name?: string | null
  delta?: string | null
  content?: string | null
  reasoning_content?: string | null
  reply?: string | null
  status?: 'completed' | 'awaiting_tool_permission' | 'cancelled' | string | null
  tool_name?: string | null
  tool_call_id?: string | null
  input?: Record<string, unknown> | null
  output?: unknown
  error?: string | null
  action_requests?: ActionRequest[] | null
  interrupt_ids?: string[] | null
  tool_calls?: unknown[] | null
}

export type ToolEvent = {
  id: string
  name: string
  args: Record<string, unknown>
  status: 'running' | 'done' | 'error'
  output?: string
  source?: string
  subagentName?: string | null
}

export type ChatMessage = {
  id: string
  role: 'user' | 'assistant' | 'system'
  content: string
  streaming?: boolean
  reasoningStreaming?: boolean
  tools?: ToolEvent[]
  reasoning?: string
}

export type PendingHitl = {
  actionRequests: ActionRequest[]
  interruptIds: string[]
}
