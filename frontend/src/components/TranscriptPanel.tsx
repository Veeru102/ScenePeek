import { useEffect, useMemo, useRef, useState } from 'react'
import type { Utterance } from '@/api/types'
import { cn, fmtTime } from '@/lib/utils'

export function TranscriptPanel({
  utterances,
  current,
  onSeek,
  className,
}: {
  utterances: Utterance[]
  current: number
  onSeek: (t: number) => void
  className?: string
}) {
  const [filter, setFilter] = useState('')
  const [follow, setFollow] = useState(true)
  const listRef = useRef<HTMLDivElement>(null)
  const activeIdx = useMemo(() => {
    let idx = -1
    for (let i = 0; i < utterances.length; i++) if (utterances[i].start_s <= current) idx = i
    return idx
  }, [utterances, current])

  useEffect(() => {
    if (!follow || activeIdx < 0) return
    const box = listRef.current
    const el = box?.querySelector<HTMLElement>(`[data-idx="${activeIdx}"]`)
    if (!box || !el) return
    // scroll only the panel, never the page
    box.scrollTo({ top: el.offsetTop - box.clientHeight / 2 + el.clientHeight / 2, behavior: 'smooth' })
  }, [activeIdx, follow])

  const shown = filter
    ? utterances.map((u, i) => [u, i] as const).filter(([u]) => u.text.toLowerCase().includes(filter.toLowerCase()))
    : utterances.map((u, i) => [u, i] as const)

  return (
    <div className={cn('flex min-h-0 flex-col rounded-xl border border-border bg-bg-card', className)}>
      <div className="flex items-center gap-2 border-b border-border px-3 py-2">
        <input
          value={filter}
          onChange={(e) => setFilter(e.target.value)}
          placeholder="Filter transcript…"
          className="h-7 min-w-0 flex-1 rounded-md bg-bg-elev px-2 text-xs outline-none placeholder:text-fg-dim"
        />
        <label className="flex items-center gap-1 text-[11px] text-fg-muted">
          <input type="checkbox" checked={follow} onChange={(e) => setFollow(e.target.checked)} /> follow
        </label>
      </div>
      <div ref={listRef} className="scrollbar-thin min-h-0 flex-1 overflow-y-auto p-2">
        {utterances.length === 0 && <div className="p-4 text-center text-xs text-fg-muted">Transcript appears as chunks finish indexing.</div>}
        {shown.map(([u, i]) => (
          <button
            key={i}
            data-idx={i}
            onClick={() => onSeek(u.start_s)}
            className={cn(
              'flex w-full gap-3 rounded-md px-2 py-1.5 text-left text-sm leading-relaxed transition-colors hover:bg-bg-elev',
              i === activeIdx && 'bg-accent/10 text-fg',
              i !== activeIdx && 'text-fg-muted',
            )}
          >
            <span className="mt-0.5 shrink-0 font-mono text-[11px] text-fg-dim">{fmtTime(u.start_s)}</span>
            <span>{u.text}</span>
          </button>
        ))}
      </div>
    </div>
  )
}
