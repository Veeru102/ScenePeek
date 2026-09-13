import { NavLink, Outlet } from 'react-router-dom'
import { Activity, Film, Search, Sparkles } from 'lucide-react'
import { cn } from '@/lib/utils'

const nav = [
  { to: '/', label: 'Library', icon: Film },
  { to: '/search', label: 'Search', icon: Search },
  { to: '/system', label: 'System', icon: Activity },
]

export function Layout() {
  return (
    <div className="min-h-full flex flex-col">
      <header className="sticky top-0 z-40 border-b border-border bg-bg/80 backdrop-blur">
        <div className="mx-auto flex h-14 max-w-7xl items-center gap-6 px-4 sm:px-6">
          <NavLink to="/" className="flex items-center gap-2 font-semibold tracking-tight">
            <span className="grid h-7 w-7 place-items-center rounded-lg bg-accent/20 text-accent">
              <Sparkles size={16} />
            </span>
            ScenePeek
          </NavLink>
          <nav className="flex items-center gap-1 text-sm">
            {nav.map(({ to, label, icon: Icon }) => (
              <NavLink
                key={to}
                to={to}
                end={to === '/'}
                className={({ isActive }) =>
                  cn(
                    'flex items-center gap-1.5 rounded-md px-3 py-1.5 text-fg-muted transition-colors hover:text-fg',
                    isActive && 'bg-bg-elev text-fg',
                  )
                }
              >
                <Icon size={15} />
                {label}
              </NavLink>
            ))}
          </nav>
        </div>
      </header>
      <main className="mx-auto w-full max-w-7xl flex-1 px-4 py-6 sm:px-6">
        <Outlet />
      </main>
    </div>
  )
}
