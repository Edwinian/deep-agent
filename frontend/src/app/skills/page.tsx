'use client'

import { createCrudPage, type FieldDef } from '@/components/CrudPage'
import type { SkillRow } from '@/types'

const fields: readonly FieldDef<SkillRow>[] = [
  { name: 'id', label: 'ID', kind: 'number', hideInTable: true },
  { name: 'name', label: 'Name', kind: 'text' },
  { name: 'description', label: 'Description', kind: 'textarea', truncateInTable: true },
  { name: 'content', label: 'Content', kind: 'textarea', hideInTable: true },
]

const SkillsCrudPage = createCrudPage<SkillRow>()

export default function SkillsPage() {
  return (
    <SkillsCrudPage
      title="Skills"
      subtitle="Browse and configure agent skills."
      endpoint="/api/skills"
      rowName={(row) => row.name}
      fields={fields}
    />
  )
}
