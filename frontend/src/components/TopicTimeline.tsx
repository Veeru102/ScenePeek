import { useMemo, useState } from 'react'
import type { Topic } from '@/api/types'
import { cn, fmtTime } from '@/lib/utils'

const PALETTE = ['#7c9cff', '#ff8fab', '#ffd166', '#6ee7b7', '#c084fc', '#fb923c', '#38bdf8', '#a3e635']

export interface Marker {
  t: number
  end?: number
  label?: string
  score?: number
}

/** Semantic timeline: topic blocks proportional to duration, a playhead, and search-hit markers. */
export function TopicTimeline({
  topics,
  duration,
  current,
  markers = [],
  onSeek,
  className,
}: {
  topics: Topic[]
  duration: number
  current?: number
  markers?: Marker[]
  onSeek?: (t: number) => void
  className?: string
}) {
  const [hover, setHover] = useState<Topic | null>(null)
  const active = useMemo(
    () => (current == null ? null : topics.find((t) => current >= t.start_s && current < t.end_s) ?? null),
    [topics, current],
  )
  if (!duration) return null
  const pct = (t: number) => `${Math.min(100, Math.max(0, (t / duration) * 100))}%`

  return (
    <div className={cn('space-y-2', className)}>
      <div
        className="relative h-9 w-full cursor-pointer select-none overflow-hidden rounded-lg bg-bg-elev"
        onClick={(e) => {
          const r = e.currentTarget.getBoundingClientRect()
          onSeek?.(((e.clientX - r.left) / r.width) * duration)
        }}
      >
        {topics.length === 0 && <div className="absolute inset-0 shimmer opacity-40" />}
        {topics.map((t, i) => (
          <div
            key={t.index}
            onMouseEnter={() => setHover(t)}
            onMouseLeave={() => setHover(null)}
            className={cn('absolute inset-y-0 border-r border-bg/60 transition-opacity', active && active.index !== t.index && 'opacity-60')}
            style={{ left: pct(t.start_s), width: pct(t.end_s - t.start_s), background: PALETTE[i % PALETTE.length] + '55' }}
            title={`${t.label} · ${fmtTime(t.start_s)}–${fmtTime(t.end_s)}`}
          >
            <div className="absolute inset-y-0 left-0 w-0.5" style={{ background: PALETTE[i % PALETTE.length] }} />
            <span className="absolute left-2 top-1/2 -translate-y-1/2 truncate pr-2 text-[11px] font-medium text-fg" style={{ maxWidth: 'calc(100% - 8px)' }}>
              {t.label}
            </span>
          </div>
        ))}
        {markers.map((m, i) => (
          <div
            key={i}
            className="absolute inset-y-0 w-1 bg-ocr shadow-[0_0_8px_var(--color-ocr)]"
            style={{ left: pct(m.t), opacity: 0.5 + (m.score ?? 0.5) * 0.5 }}
            title={m.label ? `${m.label} · ${fmtTime(m.t)}` : fmtTime(m.t)}
          />
        ))}
        {current != null && (
          <div className="absolute inset-y-0 w-0.5 bg-white shadow-[0_0_6px_rgba(255,255,255,0.8)]" style={{ left: pct(current) }} />
        )}
      </div>
      <div className="flex h-4 items-center justify-between text-[11px] text-fg-muted">
        <span className="truncate">
          {hover ? (
            <>
              <span className="text-fg">{hover.label}</span> · {fmtTime(hover.start_s)}–{fmtTime(hover.end_s)}
              {hover.keyphrases && hover.keyphrases.length > 0 && <span className="text-fg-dim"> · {hover.keyphrases.slice(0, 4).join(', ')}</span>}
            </>
          ) : active ? (
            <>
              Now: <span className="text-fg">{active.label}</span>
            </>
          ) : topics.length === 0 ? (
            'Topic timeline will appear once indexing finishes'
          ) : (
            `${topics.length} topics`
          )}
        </span>
        <span className="font-mono">{fmtTime(duration)}</span>
      </div>
    </div>
  )
}
