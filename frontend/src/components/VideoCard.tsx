import { useState } from 'react'
import { Link } from 'react-router-dom'
import { motion } from 'framer-motion'
import { Clock, Film, Layers, RotateCcw, Trash2 } from 'lucide-react'
import type { Video } from '@/api/types'
import { useDeleteVideo, useRetryVideo } from '@/api/videos'
import { ChunkStrip } from '@/components/ChunkStrip'
import { StatusBadge } from '@/components/StatusBadge'
import { Button } from '@/components/ui/button'
import { fmtBytes, fmtDuration, fmtTime } from '@/lib/utils'

export function VideoCard({ video, index = 0 }: { video: Video; index?: number }) {
  const del = useDeleteVideo()
  const retry = useRetryVideo()
  const [confirm, setConfirm] = useState(false)
  const failure = (del.error ?? retry.error) as Error | null
  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: Math.min(index * 0.04, 0.3) }}
      className="group overflow-hidden rounded-xl border border-border bg-bg-card transition-colors hover:border-border-strong"
    >
      <Link to={`/videos/${video.id}`} className="relative block aspect-video bg-bg-elev">
        {video.poster_url ? (
          <img src={video.poster_url} alt="" className="h-full w-full object-cover transition-transform duration-500 group-hover:scale-[1.03]" />
        ) : (
          <div className="grid h-full w-full place-items-center text-fg-dim">
            <Film size={28} />
          </div>
        )}
        {video.duration_s != null && (
          <span className="absolute bottom-2 right-2 rounded bg-black/70 px-1.5 py-0.5 font-mono text-[11px] text-white">
            {fmtTime(video.duration_s)}
          </span>
        )}
        <div className="absolute inset-x-0 bottom-0">
          <ChunkStrip count={video.chunk_count} done={video.chunks_done} className="rounded-none opacity-90" />
        </div>
      </Link>
      <div className="space-y-2 p-3">
        <div className="flex items-start justify-between gap-2">
          <Link to={`/videos/${video.id}`} className="line-clamp-2 text-sm font-medium leading-snug hover:text-accent">
            {video.title}
          </Link>
          <StatusBadge video={video} />
        </div>
        <div className="flex flex-wrap items-center gap-x-3 gap-y-1 text-[11px] text-fg-muted">
          <span className="inline-flex items-center gap-1"><Clock size={11} /> {fmtDuration(video.duration_s)}</span>
          <span className="inline-flex items-center gap-1"><Layers size={11} /> {video.chunks_done}/{video.chunk_count} chunks</span>
          <span>{fmtBytes(video.size_bytes)}</span>
          {video.width && <span>{video.width}×{video.height}</span>}
        </div>
        {video.error && <div className="rounded bg-err/10 px-2 py-1 text-xs text-err">{video.error}</div>}
        {failure && (
          <div role="alert" className="rounded bg-err/10 px-2 py-1 text-xs text-err">
            {del.error ? 'Delete failed' : 'Retry failed'}: {failure.message}
          </div>
        )}
        {/* always visible on touch/narrow screens; hover-revealed once a pointer exists */}
        <div className="flex items-center justify-end gap-1 transition-opacity [@media(hover:hover)]:opacity-0 [@media(hover:hover)]:group-hover:opacity-100 [@media(hover:hover)]:group-focus-within:opacity-100">
          {(video.status === 'failed' || video.chunks_failed > 0) && (
            <Button variant="ghost" size="sm" onClick={() => retry.mutate(video.id)} disabled={retry.isPending} aria-label={`Retry indexing ${video.title}`}>
              <RotateCcw size={13} /> Retry
            </Button>
          )}
          {confirm ? (
            <>
              <span className="text-xs text-fg-muted">Delete this video?</span>
              <Button variant="ghost" size="sm" onClick={() => setConfirm(false)} aria-label="Cancel delete">Cancel</Button>
              <Button
                variant="ghost"
                size="sm"
                className="text-err"
                disabled={del.isPending}
                onClick={() => del.mutate(video.id, { onSettled: () => setConfirm(false) })}
                aria-label={`Confirm delete ${video.title}`}
              >
                <Trash2 size={13} /> {del.isPending ? 'Deleting…' : 'Delete'}
              </Button>
            </>
          ) : (
            <Button variant="ghost" size="sm" className="text-err" onClick={() => setConfirm(true)} aria-label={`Delete ${video.title}`}>
              <Trash2 size={13} /> Delete
            </Button>
          )}
        </div>
      </div>
    </motion.div>
  )
}
