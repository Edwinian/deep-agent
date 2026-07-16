'use client'

import { useCallback, useEffect, useRef, useState } from 'react'

export type ToastKind = 'success' | 'error'

export type Toast = {
  id: number
  kind: ToastKind
  message: string
}

export type ToastInput = {
  kind?: ToastKind
  message: string
  /** Override the default auto-dismiss delay (ms). */
  durationMs?: number
}

/**
 * Minimal toast hook. Returns the current toasts plus a `push` function.
 * Each pushed toast auto-dismisses after `durationMs` (default 3s). Returns
 * a stable `push` so it can be safely used in effect deps.
 */
export function useToast(defaultDurationMs = 3000) {
  const [toasts, setToasts] = useState<Toast[]>([])
  const counterRef = useRef(0)

  const dismiss = useCallback((id: number) => {
    setToasts((prev) => prev.filter((t) => t.id !== id))
  }, [])

  const push = useCallback(
    (input: ToastInput) => {
      counterRef.current += 1
      const id = counterRef.current
      const toast: Toast = {
        id,
        kind: input.kind ?? 'success',
        message: input.message,
      }
      setToasts((prev) => [...prev, toast])
      const ttl = input.durationMs ?? defaultDurationMs
      if (ttl > 0) {
        window.setTimeout(() => dismiss(id), ttl)
      }
      return id
    },
    [defaultDurationMs, dismiss],
  )

  return { toasts, push, dismiss }
}

/**
 * Fixed-position toast stack, rendered top-right. Pass the `toasts` from
 * :func:`useToast` and an `onDismiss` to clear.
 */
export function ToastViewport({
  toasts,
  onDismiss,
}: {
  toasts: Toast[]
  onDismiss: (id: number) => void
}) {
  return (
    <div className="toast-viewport" role="status" aria-live="polite">
      {toasts.map((t) => (
        <div key={t.id} className={`toast toast-${t.kind}`}>
          <span className="toast-message">{t.message}</span>
          <button
            type="button"
            className="toast-close"
            aria-label="Dismiss notification"
            onClick={() => onDismiss(t.id)}
          >
            ×
          </button>
        </div>
      ))}
    </div>
  )
}

/** Convenience wrapper: render a viewport bound to a hook instance. */
export function useToastViewport(defaultDurationMs = 3000) {
  const { toasts, push, dismiss } = useToast(defaultDurationMs)
  const viewport = <ToastViewport toasts={toasts} onDismiss={dismiss} />
  return { push, dismiss, viewport }
}

/** Auto-dismiss all toasts when the calling component unmounts. */
export function useDismissOnUnmount(toasts: Toast[], dismiss: (id: number) => void) {
  useEffect(() => {
    return () => {
      toasts.forEach((t) => dismiss(t.id))
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])
}
