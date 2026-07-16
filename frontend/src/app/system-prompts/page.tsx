'use client'

import { createCrudPage, type FieldDef } from '@/components/CrudPage'
import type { SystemPromptRow } from '@/types'

const fields: readonly FieldDef<SystemPromptRow>[] = [
  { name: 'id', label: 'ID', kind: 'number', hideInTable: true },
  { name: 'name', label: 'Name', kind: 'text' },
  { name: 'content', label: 'Content', kind: 'textarea', hideInTable: true },
]

const SystemPromptsCrudPage = createCrudPage<SystemPromptRow>()

export default function SystemPromptsPage() {
  return (
    <SystemPromptsCrudPage
      title="System Prompts"
      subtitle="Edit instruction templates shared across agents."
      endpoint="/api/system-prompts"
      rowName={(row) => row.name}
      fields={fields}
    />
  )
}
