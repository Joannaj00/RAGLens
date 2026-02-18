"use client"
import Link from "next/link"
import { usePathname } from "next/navigation"

const links = [
  { href: "/",            label: "Overview" },
  { href: "/drill-down",  label: "Drill-down" },
  { href: "/compare",     label: "Compare" },
  { href: "/inspector",   label: "Inspector" },
]

export default function NavBar() {
  const pathname = usePathname()
  return (
    <nav className="bg-slate-900 text-white px-6 py-4 flex items-center gap-8 shadow-md">
      <span className="font-bold text-lg tracking-tight text-blue-400">🔍 RAGLens</span>
      <div className="flex gap-6">
        {links.map(({ href, label }) => (
          <Link
            key={href}
            href={href}
            className={`text-sm font-medium transition-colors hover:text-blue-400 ${
              pathname === href ? "text-blue-400 border-b-2 border-blue-400 pb-0.5" : "text-slate-300"
            }`}
          >
            {label}
          </Link>
        ))}
      </div>
    </nav>
  )
}
