'use client'

import Link from 'next/link'
import { usePathname } from 'next/navigation'

const NAV_ITEMS = [
  { href: '/chats', label: 'Chats' },
  { href: '/agents', label: 'Agents' },
  { href: '/system-prompts', label: 'System Prompts' },
  { href: '/skills', label: 'Skills' },
] as const

export default function Sidebar() {
  const pathname = usePathname()
  const active = (href: string) =>
    pathname === href || pathname?.startsWith(`${href}/`)

  return (
    <nav className="sidebar" aria-label="Primary">
      <p className="sidebar-brand">Deep Agents</p>
      <ul className="sidebar-nav">
        {NAV_ITEMS.map((item) => (
          <li key={item.href}>
            <Link
              href={item.href}
              className={`sidebar-link${active(item.href) ? ' active' : ''}`}
            >
              {item.label}
            </Link>
          </li>
        ))}
      </ul>
      <div className="sidebar-footer">Deep Agents &middot; v0.1</div>
    </nav>
  )
}
