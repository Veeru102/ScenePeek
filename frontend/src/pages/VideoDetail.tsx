import { useMemo, useRef, useState } from 'react'
import { Link, useParams, useSearchParams } from 'react-router-dom'
import { ArrowLeft, Clock, Layers, RotateCcw } from 'lucide-react'
import { useSearchQuery } from '@/api/search'
import { useFrames, useRetryVideo, useTimeline, useTranscript, useVideo } from '@/api/videos'
import { ChunkStrip } from '@/components/ChunkStrip'
import { Filmstrip } from '@/components/Filmstrip'
import { ResultCard } from '@/components/ResultCard'
import { SearchBar } from '@/components/SearchBar'
import { StatusBadge } from '@/components/StatusBadge'
import { TopicTimeline } from '@/components/TopicTimeline'
import { TranscriptPanel } from '@/components/TranscriptPanel'
import { VideoPlayer, type PlayerHandle } from '@/components/VideoPlayer'
import { Button } from '@/components/ui/button'
import { fmtBytes, fmtDuration } from '@/lib/utils'

export function VideoDetailPage() {
  const { id } = useParams()
  const [params] = useSearchParams()
  const startAt = params.get('t') ? Number(params.get('t')) : null
  const { data: video, error: videoError } = useVideo(id)
  const processing = video?.status === 'processing' || video?.status === 'uploaded'
  const { data: transcript } = useTranscript(id, true)
  const { data: topics } = useTimeline(id, processing || (video?.status === 'ready' && video.topic_count === 0))
  const { data: frames } = useFrames(id, processing)
  const retry = useRetryVideo()
  const player = useRef<PlayerHandle>(null)
  const [current, setCurrent] = useState(0)
  const [q, setQ] = useState('')
  const search = useSearchQuery(q && id ? { q, video_ids: [id], limit: 15 } : null)
  const markers = useMemo(
    () => (search.data?.hits ?? []).map((h) => ({ t: h.start_s, end: h.end_s, label: h.text.slice(0, 60), score: h.score })),
    [search.data],
  )
  const seek = (t: number) => player.current?.seek(t)

  // transcript rows refetch while processing so new chunks show up
  const { refetch: refetchTranscript } = useTranscript(id, false)
  const lastDone = video?.chunks_done
  const seenDone = useRef(lastDone)
  if (lastDone !== seenDone.current) {
    seenDone.current = lastDone
    void refetchTranscript()
  }

  if (videoError)
    return (
      <div role="alert" className="max-w-xl space-y-3 rounded-xl border border-err/40 bg-err/10 p-5 text-sm">
        <div className="font-medium">Couldn't load this video</div>
        <div className="text-fg-muted">{(videoError as Error).message}</div>
        <Link to="/" className="inline-flex items-center gap-1 text-fg-muted hover:text-fg"><ArrowLeft size={15} /> Back to library</Link>
      </div>
    )
  if (!video) return <div className="aspect-video max-w-4xl rounded-xl shimmer" aria-busy="true" aria-label="Loading video" />

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center gap-3">
        <Link to="/" className="inline-flex items-center gap-1 text-sm text-fg-muted hover:text-fg"><ArrowLeft size={15} /> Library</Link>
        <h1 className="text-xl font-semibold tracking-tight">{video.title}</h1>
        <StatusBadge video={video} />
        <div className="ml-auto flex items-center gap-3 text-xs text-fg-muted">
          <span className="inline-flex items-center gap-1"><Clock size={12} /> {fmtDuration(video.duration_s)}</span>
          <span className="inline-flex items-center gap-1"><Layers size={12} /> {video.chunks_done}/{video.chunk_count} chunks</span>
          <span>{fmtBytes(video.size_bytes)}</span>
          {video.width && <span>{video.width}×{video.height}</span>}
          {(video.status === 'failed' || video.chunks_failed > 0) && (
            <Button variant="secondary" size="sm" onClick={() => retry.mutate(video.id)}><RotateCcw size={13} /> Retry failed</Button>
          )}
        </div>
      </div>
      {video.error && <div className="rounded-lg border border-err/30 bg-err/10 px-3 py-2 text-sm text-err">{video.error}</div>}

      <div className="grid gap-4 lg:grid-cols-[minmax(0,1fr)_360px]">
        <div className="space-y-3">
          <div className="overflow-hidden rounded-xl border border-border bg-black">
            <VideoPlayer ref={player} src={video.playback_url} poster={video.poster_url} startAt={startAt} onTime={setCurrent} className="aspect-video w-full" />
          </div>
          <div className="space-y-1">
            <div className="flex items-center justify-between text-[11px] text-fg-muted">
              <span>Indexing progress · {video.chunk_count} chunks of 60 s</span>
              <span>{Math.round(video.progress * 100)}%</span>
            </div>
            <ChunkStrip chunks={video.chunks} onSeek={seek} className="h-2" />
          </div>
          <TopicTimeline topics={topics ?? []} duration={video.duration_s ?? 0} current={current} markers={markers} onSeek={seek} />
          <Filmstrip frames={frames ?? []} current={current} onSeek={seek} />

          <div className="space-y-3 pt-2">
            <SearchBar size="md" placeholder="Search inside this video…" loading={search.isFetching} onSubmit={setQ} initial={q} />
            {search.data && (
              <div className="space-y-2">
                <div className="text-xs text-fg-muted">
                  {search.data.hits.length} matches · {search.data.timings_ms.total.toFixed(0)} ms · markers shown on the timeline
                </div>
                {search.data.hits.map((h, i) => (
                  <ResultCard key={h.segment_id} hit={h} index={i} showVideo={false} onPlay={(hit) => seek(hit.start_s)} />
                ))}
              </div>
            )}
          </div>
        </div>
        <TranscriptPanel utterances={transcript ?? []} current={current} onSeek={seek} className="h-[70vh] lg:sticky lg:top-20" />
      </div>
    </div>
  )
}
