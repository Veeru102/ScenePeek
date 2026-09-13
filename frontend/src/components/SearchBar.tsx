import { useEffect, useRef, useState } from 'react'
import { Loader2, Search } from 'lucide-react'
import { cn } from '@/lib/utils'

export function SearchBar({
  initial = '',
  placeholder = 'Describe a moment… e.g. "where the professor explains B+ tree leaf nodes"',
  loading,
  onSubmit,
  autoFocus,
  size = 'lg',
}: {
  initial?: string
  placeholder?: string
  loading?: boolean
  onSubmit: (q: string) => void
  autoFocus?: boolean
  size?: 'lg' | 'md'
}) {
  const [q, setQ] = useState(initial)
  const ref = useRef<HTMLInputElement>(null)
  useEffect(() => setQ(initial), [initial])
  useEffect(() => {
    if (autoFocus) ref.current?.focus()
  }, [autoFocus])
  return (
    <form
      onSubmit={(e) => {
        e.preventDefault()
        if (q.trim()) onSubmit(q.trim())
      }}
      className={cn(
        'flex items-center gap-2 rounded-xl border border-border bg-bg-card pr-2 shadow-[0_0_0_1px_transparent] transition-shadow focus-within:border-accent/60 focus-within:shadow-[0_0_0_4px_color-mix(in_oklab,var(--color-accent)_20%,transparent)]',
        size === 'lg' ? 'pl-4' : 'pl-3',
      )}
    >
      {loading ? <Loader2 size={18} className="animate-spin text-accent" /> : <Search size={18} className="text-fg-dim" />}
      <input
        ref={ref}
        value={q}
        onChange={(e) => setQ(e.target.value)}
        placeholder={placeholder}
        className={cn('w-full bg-transparent outline-none placeholder:text-fg-dim', size === 'lg' ? 'h-12 text-base' : 'h-9 text-sm')}
      />
      <button
        type="submit"
        className={cn(
          'rounded-lg bg-accent px-3 font-medium text-white transition-colors hover:bg-accent-strong disabled:opacity-40',
          size === 'lg' ? 'h-9 text-sm' : 'h-7 text-xs',
        )}
        disabled={!q.trim() || loading}
      >
        Search
      </button>
    </form>
  )
}
