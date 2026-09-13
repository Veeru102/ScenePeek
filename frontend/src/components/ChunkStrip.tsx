import { cn } from '@/lib/utils'
import type { Chunk, ChunkStatus } from '@/api/types'

const color: Record<ChunkStatus, string> = {
  pending: 'bg-border',
  running: 'bg-accent animate-pulse',
  done: 'bg-ok',
  failed: 'bg-err',
}

/** One cell per chunk: a compact picture of indexing progress and where failures are. */
export function ChunkStrip({
  chunks,
  count,
  done,
  className,
  onSeek,
}: {
  chunks?: Chunk[]
  count?: number
  done?: number
  className?: string
  onSeek?: (t: number) => void
}) {
  const cells: { status: ChunkStatus; start?: number; title?: string }[] = chunks
    ? chunks.map((c) => ({ status: c.status, start: c.start_s, title: `Chunk ${c.index} · ${c.status}${c.stage ? ' · ' + c.stage : ''}` }))
    : Array.from({ length: count ?? 0 }, (_, i) => ({ status: i < (done ?? 0) ? 'done' : 'pending' }))
  if (cells.length === 0) return <div className={cn('h-1.5 w-full rounded-full shimmer', className)} />
  return (
    <div className={cn('flex h-1.5 w-full gap-px overflow-hidden rounded-full', className)}>
      {cells.map((c, i) => (
        <div
          key={i}
          title={c.title}
          onClick={c.start != null && onSeek ? () => onSeek(c.start!) : undefined}
          className={cn('flex-1 transition-colors duration-500', color[c.status], onSeek && 'cursor-pointer')}
        />
      ))}
    </div>
  )
}
