'use client'

import { createCrudPage, type FieldDef } from '@/components/CrudPage'
import type { AgentRow } from '@/types'

const fields: readonly FieldDef<AgentRow>[] = [
  { name: 'id', label: 'ID', kind: 'number', hideInTable: true },
  { name: 'name', label: 'Name', kind: 'text' },
  { name: 'description', label: 'Description', kind: 'textarea', truncateInTable: true },
  { name: 'system_prompt_id', label: 'System prompt ID', kind: 'number' },
  { name: 'model', label: 'Model', kind: 'text' },
  { name: 'subagent_ids', label: 'Subagent IDs', kind: 'list' },
  { name: 'tool_ids', label: 'Tool IDs', kind: 'list' },
  { name: 'skill_ids', label: 'Skill IDs', kind: 'list' },
]

// Lock CrudPage's generic to AgentRow once, so we can use it as a JSX tag.
const AgentsCrudPage = createCrudPage<AgentRow>()

export default function AgentsPage() {
  return (
    <AgentsCrudPage
      title="Agents"
      subtitle="Manage agent specs, models, tools, and subagents."
      endpoint="/api/agents"
      rowName={(row) => row.name}
      fields={fields}
    />
  )
}
