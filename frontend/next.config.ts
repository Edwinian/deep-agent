import type { NextConfig } from 'next'

const apiBase = process.env.BACKEND_API_BASE_URL ?? 'http://127.0.0.1:8000'

const nextConfig: NextConfig = {
  reactStrictMode: true,
  // Proxy /chats/* and /speech-to-text from the browser (port 5173)
  // to the FastAPI backend so the frontend and API share an origin.
  async rewrites() {
    return [
      { source: '/chats/:path*', destination: `${apiBase}/chats/:path*` },
      { source: '/speech-to-text/:path*', destination: `${apiBase}/speech-to-text/:path*` },
    ]
  },
}

export default nextConfig
