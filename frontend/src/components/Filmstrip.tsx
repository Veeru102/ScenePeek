import { useEffect, useRef } from 'react'
import type { KeyFrame } from '@/api/types'
import { cn, fmtTime } from '@/lib/utils'

/** Evenly spaced keyframes; the one nearest the playhead is highlighted and kept in view. */
export function Filmstrip({ frames, current, onSeek }: { frames: KeyFrame[]; current: number; onSeek: (t: number) => void }) {
  const ref = useRef<HTMLDivElement>(null)
  let active = 0
  for (let i = 0; i < frames.length; i++) if (frames[i].t_s <= current) active = i
  useEffect(() => {
    const el = ref.current?.children[active] as HTMLElement | undefined
    el?.scrollIntoView({ block: 'nearest', inline: 'center', behavior: 'smooth' })
  }, [active])
  if (frames.length === 0) return null
  return (
    <div ref={ref} className="scrollbar-thin flex gap-1 overflow-x-auto rounded-lg border border-border bg-bg-card p-1" role="list" aria-label="Keyframes">
      {frames.map((f, i) => (
        <button
          key={f.t_s}
          role="listitem"
          onClick={() => onSeek(f.t_s)}
          title={`${fmtTime(f.t_s)}${f.caption ? ' — ' + f.caption : ''}`}
          aria-label={`Seek to ${fmtTime(f.t_s)}`}
          className={cn(
            'relative h-14 w-24 shrink-0 overflow-hidden rounded-md border-2 transition-colors',
            i === active ? 'border-accent' : 'border-transparent hover:border-border-strong',
          )}
        >
          <img src={f.url} alt="" loading="lazy" className="h-full w-full object-cover" />
          <span className="absolute bottom-0.5 right-0.5 rounded bg-black/70 px-1 font-mono text-[9px] text-white">{fmtTime(f.t_s)}</span>
        </button>
      ))}
    </div>
  )
}
