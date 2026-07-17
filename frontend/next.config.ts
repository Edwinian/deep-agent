import type { NextConfig } from 'next'

const apiBase = process.env.BACKEND_API_BASE_URL ?? 'http://127.0.0.1:8000'

const nextConfig: NextConfig = {
  reactStrictMode: true,
  // Hide the bottom-left Next.js dev indicator widget.
  devIndicators: false,
  // Compression buffers responses and breaks SSE when proxied through rewrites.
  compress: false,
  // Proxy API endpoints from the browser (port 5173) to the FastAPI backend
  // so the frontend and API share an origin for non-streaming CRUD.
  //
  // Stream/cancel intentionally bypass these rewrites (see `STREAM_API_BASE`
  // in `src/api.ts`) because Next.js rewrites buffer `text/event-stream`.
  //
  // Two flavours of proxy:
  //   * /chats/* and /speech-to-text/* — legacy paths the chat client still
  //     calls directly (every call has a sub-path, so no collision with the
  //     /chats Next.js page).
  //   * /api/* — generic prefix for CRUD endpoints. Namespacing under /api
  //     avoids collisions between bare collection paths (e.g. GET /agents)
  //     and Next.js pages at the same URL.
  async rewrites() {
    return [
      { source: '/chats/:path*', destination: `${apiBase}/chats/:path*` },
      { source: '/speech-to-text/:path*', destination: `${apiBase}/speech-to-text/:path*` },
      { source: '/api/:path*', destination: `${apiBase}/:path*` },
    ]
  },
}

export default nextConfig
