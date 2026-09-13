import { Link } from 'react-router-dom'
import { motion } from 'framer-motion'
import { ExternalLink, Play, ScanText } from 'lucide-react'
import type { SearchHit } from '@/api/types'
import { SignalChips } from '@/components/SignalChips'
import { cn, fmtTime } from '@/lib/utils'

export function ResultCard({
  hit,
  index,
  active,
  onPlay,
  showVideo = true,
}: {
  hit: SearchHit
  index: number
  active?: boolean
  onPlay: (hit: SearchHit) => void
  showVideo?: boolean
}) {
  return (
    <motion.article
      initial={{ opacity: 0, y: 6 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: Math.min(index * 0.03, 0.3) }}
      className={cn(
        'group flex gap-3 rounded-xl border border-border bg-bg-card p-3 transition-colors hover:border-border-strong',
        active && 'border-accent/70 bg-accent/5',
      )}
    >
      <button
        onClick={() => onPlay(hit)}
        className="relative aspect-video w-40 shrink-0 overflow-hidden rounded-lg bg-bg-elev sm:w-48"
        title="Play from this moment"
      >
        {hit.keyframe_url ? (
          <img src={hit.keyframe_url} alt="" className="h-full w-full object-cover" loading="lazy" />
        ) : hit.video.poster_url ? (
          <img src={hit.video.poster_url} alt="" className="h-full w-full object-cover opacity-70" loading="lazy" />
        ) : null}
        <span className="absolute inset-0 grid place-items-center bg-black/0 transition-colors group-hover:bg-black/30">
          <span className="grid h-9 w-9 scale-75 place-items-center rounded-full bg-white/90 text-black opacity-0 transition-all group-hover:scale-100 group-hover:opacity-100">
            <Play size={16} className="ml-0.5" />
          </span>
        </span>
        <span className="absolute bottom-1.5 left-1.5 rounded bg-black/75 px-1.5 py-0.5 font-mono text-[11px] text-white">
          {fmtTime(hit.start_s)}
        </span>
        <span className="absolute right-1.5 top-1.5 rounded bg-accent/90 px-1.5 py-0.5 font-mono text-[10px] font-semibold text-white">
          {(hit.score * 100).toFixed(0)}
        </span>
      </button>
      <div className="min-w-0 flex-1 space-y-1.5">
        {showVideo && (
          <div className="flex items-center gap-2 text-xs text-fg-muted">
            <Link to={`/videos/${hit.video.id}?t=${Math.floor(hit.start_s)}`} className="truncate font-medium text-fg hover:text-accent">
              {hit.video.title}
            </Link>
            <span className="font-mono">{fmtTime(hit.start_s)}–{fmtTime(hit.end_s)}</span>
            <Link to={`/videos/${hit.video.id}?t=${Math.floor(hit.start_s)}`} className="ml-auto text-fg-dim hover:text-fg" title="Open video">
              <ExternalLink size={13} />
            </Link>
          </div>
        )}
        <p
          className="line-clamp-3 text-sm leading-relaxed text-fg/90"
          dangerouslySetInnerHTML={{ __html: hit.snippet_html || '<span class="text-fg-dim">No speech in this segment</span>' }}
        />
        {hit.ocr_text && (
          <p className="flex items-start gap-1 truncate text-xs text-ocr/90">
            <ScanText size={12} className="mt-0.5 shrink-0" />
            <span className="truncate">{hit.ocr_text.replace(/\n/g, ' · ')}</span>
          </p>
        )}
        <SignalChips signals={hit.signals} />
      </div>
    </motion.article>
  )
}
