import { CheckCircle2, CircleDashed, Loader2, UploadCloud, XCircle } from 'lucide-react'
import { Badge } from '@/components/ui/badge'
import type { Video } from '@/api/types'

export function StatusBadge({ video }: { video: Video }) {
  switch (video.status) {
    case 'ready':
      return <Badge tone="ok"><CheckCircle2 size={12} /> Indexed</Badge>
    case 'processing':
      return (
        <Badge tone="accent">
          <Loader2 size={12} className="animate-spin" /> Indexing {Math.round(video.progress * 100)}%
        </Badge>
      )
    case 'failed':
      return <Badge tone="err"><XCircle size={12} /> Failed</Badge>
    case 'uploaded':
      return <Badge tone="warn"><CircleDashed size={12} /> Queued</Badge>
    default:
      return <Badge tone="neutral"><UploadCloud size={12} /> Uploading</Badge>
  }
}
