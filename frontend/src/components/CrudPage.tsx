'use client'

import { useCallback, useEffect, useMemo, useState } from 'react'
import { crudCreate, crudDelete, crudGetMany, crudUpdate } from '@/api'
import { useToastViewport } from '@/components/Toast'

/* -------------------------------------------------------------------------- */
/* Field descriptors — declarative schema for a CRUD resource                */
/* -------------------------------------------------------------------------- */

export type FieldKind = 'text' | 'textarea' | 'number' | 'list'

export type FieldDef<T> = {
  name: keyof T & string
  label: string
  kind: FieldKind
  /** Hide from the main table (use for long content). */
  hideInTable?: boolean
  /** Show a muted truncated preview in the table instead of the full value. */
  truncateInTable?: boolean
  /** Optional placeholder. */
  placeholder?: string
}

/* -------------------------------------------------------------------------- */
/* Helpers                                                                    */
/* -------------------------------------------------------------------------- */

/** Every row shape the CRUD pages render must carry a numeric primary key. */
type HasId = { id: number }

/** Render a value as a string for the table. */
function renderCell<T>(row: T, field: FieldDef<T>): string {
  const raw = row[field.name]
  if (raw == null) return ''
  if (field.kind === 'list' && Array.isArray(raw)) {
    return (raw as unknown[]).join(', ')
  }
  return String(raw)
}

function parseListInput(value: string): number[] {
  return value
    .split(',')
    .map((part) => part.trim())
    .filter(Boolean)
    .map((part) => Number(part))
    .filter((n) => !Number.isNaN(n))
}

/** Extract a plain-string form state from a row (for edit drawer defaults). */
function rowToFormState<T extends Record<string, unknown>>(
  row: T,
  fields: readonly FieldDef<T>[],
): Record<string, string> {
  const state: Record<string, string> = {}
  for (const f of fields) {
    const v = row[f.name]
    if (v == null) state[f.name] = ''
    else if (f.kind === 'list') state[f.name] = Array.isArray(v) ? (v as unknown[]).join(', ') : ''
    else state[f.name] = String(v)
  }
  return state
}

/** Convert a string form state into a typed payload for create/update. */
function formStateToPayload(
  state: Record<string, string>,
  fields: readonly FieldDef<any>[],
): Record<string, unknown> {
  const payload: Record<string, unknown> = {}
  for (const f of fields) {
    const raw = state[f.name] ?? ''
    if (f.kind === 'number') {
      payload[f.name] = raw === '' ? null : Number(raw)
    } else if (f.kind === 'list') {
      payload[f.name] = raw === '' ? null : parseListInput(raw)
    } else {
      payload[f.name] = raw
    }
  }
  return payload
}

/* -------------------------------------------------------------------------- */
/* Field input                                                                */
/* -------------------------------------------------------------------------- */

function FieldInput({
  field,
  value,
  onChange,
  disabled,
}: {
  field: FieldDef<any>
  value: string
  onChange: (v: string) => void
  disabled?: boolean
}) {
  const common = {
    value,
    disabled,
    onChange: (e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>) =>
      onChange(e.target.value),
    placeholder: field.placeholder,
  }
  if (field.kind === 'textarea') {
    return <textarea className="crud-input" rows={8} {...common} />
  }
  return (
    <input
      className="crud-input"
      type={field.kind === 'number' ? 'number' : 'text'}
      {...common}
    />
  )
}

function FieldRow({
  field,
  value,
  onChange,
  disabled,
}: {
  field: FieldDef<any>
  value: string
  onChange: (v: string) => void
  disabled?: boolean
}) {
  return (
    <label className="crud-field">
      <span className="crud-label">{field.label}</span>
      <FieldInput field={field} value={value} onChange={onChange} disabled={disabled} />
    </label>
  )
}

/* -------------------------------------------------------------------------- */
/* Delete confirm modal                                                       */
/* -------------------------------------------------------------------------- */

function DeleteConfirmModal({
  open,
  title,
  message,
  busy,
  onCancel,
  onConfirm,
}: {
  open: boolean
  title: string
  message: string
  busy: boolean
  onCancel: () => void
  onConfirm: () => void
}) {
  useEffect(() => {
    if (!open) return
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && !busy) onCancel()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [open, busy, onCancel])

  if (!open) return null
  return (
    <div className="crud-modal-root">
      <button
        type="button"
        className="crud-modal-backdrop"
        aria-label="Cancel delete"
        onClick={() => !busy && onCancel()}
      />
      <div className="crud-modal crud-modal-sm" role="dialog" aria-modal>
        <h3>{title}</h3>
        <p>{message}</p>
        <div className="crud-modal-actions">
          <button
            type="button"
            className="ghost-btn"
            disabled={busy}
            onClick={onCancel}
          >
            Cancel
          </button>
          <button
            type="button"
            className="danger-btn"
            disabled={busy}
            onClick={onConfirm}
          >
            {busy ? 'Deleting…' : 'Delete'}
          </button>
        </div>
      </div>
    </div>
  )
}

/* -------------------------------------------------------------------------- */
/* Create modal                                                               */
/* -------------------------------------------------------------------------- */

function CreateModal<T extends Record<string, unknown>>({
  open,
  title,
  fields,
  busy,
  onCancel,
  onCreate,
}: {
  open: boolean
  title: string
  fields: readonly FieldDef<T>[]
  busy: boolean
  onCancel: () => void
  onCreate: (payload: Record<string, unknown>) => void
}) {
  const [state, setState] = useState<Record<string, string>>({})
  useEffect(() => {
    if (!open) return
    const fresh: Record<string, string> = {}
    for (const f of fields) fresh[f.name] = ''
    setState(fresh)
  }, [open, fields])

  if (!open) return null

  const submit = () => onCreate(formStateToPayload(state, fields))

  return (
    <div className="crud-modal-root">
      <button
        type="button"
        className="crud-modal-backdrop"
        aria-label="Cancel create"
        onClick={() => !busy && onCancel()}
      />
      <div className="crud-modal" role="dialog" aria-modal>
        <header className="crud-modal-header">
          <h3>{title}</h3>
        </header>
        <div className="crud-form">
          {fields.map((f) => (
            <FieldRow
              key={f.name}
              field={f}
              value={state[f.name] ?? ''}
              onChange={(v) => setState((prev) => ({ ...prev, [f.name]: v }))}
              disabled={busy}
            />
          ))}
        </div>
        <div className="crud-modal-actions">
          <button
            type="button"
            className="ghost-btn"
            disabled={busy}
            onClick={onCancel}
          >
            Cancel
          </button>
          <button
            type="button"
            className="primary-btn"
            disabled={busy}
            onClick={submit}
          >
            {busy ? 'Creating…' : 'Create'}
          </button>
        </div>
      </div>
    </div>
  )
}

/* -------------------------------------------------------------------------- */
/* Edit drawer                                                                */
/* -------------------------------------------------------------------------- */

function EditDrawer<T extends HasId & Record<string, unknown>>({
  row,
  fields,
  busy,
  onClose,
  onUpdate,
}: {
  row: T | null
  fields: readonly FieldDef<T>[]
  busy: boolean
  onClose: () => void
  onUpdate: (payload: Record<string, unknown>) => void
}) {
  const [state, setState] = useState<Record<string, string>>({})

  useEffect(() => {
    if (row) setState(rowToFormState(row, fields))
  }, [row, fields])

  useEffect(() => {
    if (!row) return
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && !busy) onClose()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [row, busy, onClose])

  if (!row) return null

  const submit = () => onUpdate(formStateToPayload(state, fields))

  return (
    <div className="crud-drawer-root">
      <button
        type="button"
        className="crud-drawer-backdrop"
        aria-label="Close drawer"
        onClick={() => !busy && onClose()}
      />
      <aside className="crud-drawer" role="dialog" aria-modal>
        <header className="crud-drawer-header">
          <h3>Edit #{row.id}</h3>
          <button
            type="button"
            className="crud-drawer-close"
            aria-label="Close"
            onClick={() => !busy && onClose()}
          >
            ×
          </button>
        </header>
        <div className="crud-form">
          {fields.map((f) => (
            <FieldRow
              key={f.name}
              field={f}
              value={state[f.name] ?? ''}
              onChange={(v) => setState((prev) => ({ ...prev, [f.name]: v }))}
              disabled={busy}
            />
          ))}
        </div>
        <footer className="crud-drawer-footer">
          <button
            type="button"
            className="ghost-btn"
            disabled={busy}
            onClick={onClose}
          >
            Cancel
          </button>
          <button
            type="button"
            className="primary-btn"
            disabled={busy}
            onClick={submit}
          >
            {busy ? 'Updating…' : 'Update'}
          </button>
        </footer>
      </aside>
    </div>
  )
}

/* -------------------------------------------------------------------------- */
/* Main CrudPage                                                              */
/* -------------------------------------------------------------------------- */

export type CrudPageProps<T extends HasId & Record<string, unknown>> = {
  title: string
  subtitle: string
  endpoint: string
  rowName: (row: T) => string
  fields: readonly FieldDef<T>[]
}

type LoadState = 'idle' | 'loading' | 'error'

function CrudPageImpl<T extends HasId & Record<string, unknown>>({
  title,
  subtitle,
  endpoint,
  rowName,
  fields,
}: CrudPageProps<T>) {
  const [rows, setRows] = useState<T[]>([])
  const [loadState, setLoadState] = useState<LoadState>('idle')
  const [error, setError] = useState<string | null>(null)

  const [createOpen, setCreateOpen] = useState(false)
  const [editRow, setEditRow] = useState<T | null>(null)
  const [deleteRow, setDeleteRow] = useState<T | null>(null)

  const [createBusy, setCreateBusy] = useState(false)
  const [updateBusy, setUpdateBusy] = useState(false)
  const [deleteBusy, setDeleteBusy] = useState(false)

  const { push: pushToast, viewport: toastViewport } = useToastViewport()

  const tableFields = useMemo(
    () => fields.filter((f) => !f.hideInTable),
    [fields],
  )

  const refresh = useCallback(async () => {
    setLoadState('loading')
    setError(null)
    try {
      const data = await crudGetMany<T>(endpoint)
      setRows(data)
      setLoadState('idle')
    } catch (e) {
      setError((e as Error).message || 'Failed to load')
      setLoadState('error')
    }
  }, [endpoint])

  useEffect(() => {
    void refresh()
  }, [refresh])

  const handleCreate = async (payload: Record<string, unknown>) => {
    setCreateBusy(true)
    setError(null)
    try {
      const created = await crudCreate<T>(endpoint, payload)
      setCreateOpen(false)
      await refresh()
      pushToast({
        kind: 'success',
        message: `Created ${rowName(created) || title.replace(/s$/, '')}.`,
      })
    } catch (e) {
      setError((e as Error).message || 'Create failed')
    } finally {
      setCreateBusy(false)
    }
  }

  const handleUpdate = async (payload: Record<string, unknown>) => {
    if (!editRow) return
    setUpdateBusy(true)
    setError(null)
    try {
      await crudUpdate(endpoint, editRow.id, payload)
      const name = rowName(editRow)
      setEditRow(null)
      await refresh()
      pushToast({
        kind: 'success',
        message: `Updated ${name || 'row'}.`,
      })
    } catch (e) {
      setError((e as Error).message || 'Update failed')
    } finally {
      setUpdateBusy(false)
    }
  }

  const handleDelete = async () => {
    if (!deleteRow) return
    setDeleteBusy(true)
    setError(null)
    try {
      const name = rowName(deleteRow)
      await crudDelete(endpoint, deleteRow.id)
      setDeleteRow(null)
      await refresh()
      pushToast({
        kind: 'success',
        message: `Deleted ${name || 'row'}.`,
      })
    } catch (e) {
      setError((e as Error).message || 'Delete failed')
    } finally {
      setDeleteBusy(false)
    }
  }

  return (
    <section className="page-pad crud-section">
      <header className="crud-topbar">
        <div className="page-header">
          <h1>{title}</h1>
          <p>{subtitle}</p>
        </div>
        <button
          type="button"
          className="primary-btn"
          onClick={() => setCreateOpen(true)}
        >
          + Create
        </button>
      </header>

      {error ? <p className="error-banner">{error}</p> : null}

      <div className="crud-table-wrap">
        {loadState === 'loading' && rows.length === 0 ? (
          <p className="crud-muted">Loading…</p>
        ) : loadState === 'error' && rows.length === 0 ? (
          <p className="crud-muted">Failed to load. {error}</p>
        ) : rows.length === 0 ? (
          <p className="crud-muted">No rows yet. Click “Create” to add one.</p>
        ) : (
          <table className="crud-table">
            <thead>
              <tr>
                {tableFields.map((f) => (
                  <th key={f.name}>{f.label}</th>
                ))}
                <th className="crud-th-actions" aria-label="Actions" />
              </tr>
            </thead>
            <tbody>
              {rows.map((row) => (
                <tr
                  key={row.id}
                  className="crud-row"
                  onClick={() => setEditRow(row)}
                >
                  {tableFields.map((f) => {
                    const text = renderCell(row, f)
                    return (
                      <td key={f.name} className={f.truncateInTable ? 'crud-cell-truncate' : ''}>
                        {text}
                      </td>
                    )
                  })}
                  <td className="crud-td-actions">
                    <button
                      type="button"
                      className="crud-delete-btn"
                      aria-label={`Delete ${rowName(row)}`}
                      onClick={(e) => {
                        e.stopPropagation()
                        setDeleteRow(row)
                      }}
                    >
                      Delete
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>

      <CreateModal
        open={createOpen}
        title={`Create ${title}`}
        fields={fields}
        busy={createBusy}
        onCancel={() => setCreateOpen(false)}
        onCreate={handleCreate}
      />

      <EditDrawer
        row={editRow}
        fields={fields}
        busy={updateBusy}
        onClose={() => setEditRow(null)}
        onUpdate={handleUpdate}
      />

      <DeleteConfirmModal
        open={deleteRow != null}
        title={`Delete ${deleteRow ? rowName(deleteRow) : ''}?`}
        message="This soft-deletes the row. You won’t see it in the list anymore."
        busy={deleteBusy}
        onCancel={() => setDeleteRow(null)}
        onConfirm={handleDelete}
      />

      {toastViewport}
    </section>
  )
}

/**
 * Build a typed CrudPage component bound to a specific row shape.
 *
 * TSX cannot express a generic argument on a JSX tag
 * (``<CrudPage<T> />`` is a syntax error), so callers use this factory to
 * lock in ``T`` once and render the returned component as a normal element.
 */
export function createCrudPage<T extends HasId & Record<string, unknown>>() {
  return function CrudPageTyped(props: CrudPageProps<T>) {
    return <CrudPageImpl {...props} />
  }
}
