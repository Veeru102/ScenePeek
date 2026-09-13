import { Film } from 'lucide-react'
import { useVideos } from '@/api/videos'
import { UploadDropzone } from '@/components/UploadDropzone'
import { VideoCard } from '@/components/VideoCard'

export function LibraryPage() {
  const { data: videos, isLoading } = useVideos()
  const ready = videos?.filter((v) => v.status === 'ready').length ?? 0
  const processing = videos?.filter((v) => v.status === 'processing' || v.status === 'uploaded').length ?? 0

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight">Library</h1>
          <p className="mt-1 text-sm text-fg-muted">
            {videos?.length ?? 0} videos · {ready} indexed{processing > 0 && ` · ${processing} processing`}
          </p>
        </div>
      </div>

      <UploadDropzone />

      {isLoading ? (
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
          {Array.from({ length: 4 }).map((_, i) => (
            <div key={i} className="aspect-[4/3] rounded-xl shimmer" />
          ))}
        </div>
      ) : videos && videos.length > 0 ? (
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
          {videos.map((v, i) => (
            <VideoCard key={v.id} video={v} index={i} />
          ))}
        </div>
      ) : (
        <div className="flex flex-col items-center gap-2 rounded-xl border border-border py-16 text-center text-fg-muted">
          <Film size={28} className="text-fg-dim" />
          <div className="text-sm">No videos yet. Upload one to start indexing.</div>
        </div>
      )}
    </div>
  )
}
