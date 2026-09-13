import { useCallback, useRef, useState } from 'react'
import { useQueryClient } from '@tanstack/react-query'
import { motion, AnimatePresence } from 'framer-motion'
import { CheckCircle2, FileVideo, UploadCloud, XCircle } from 'lucide-react'
import { uploadVideo, type UploadProgress } from '@/api/videos'
import { cn, fmtBytes } from '@/lib/utils'

export function UploadDropzone() {
  const qc = useQueryClient()
  const inputRef = useRef<HTMLInputElement>(null)
  const [drag, setDrag] = useState(false)
  const [uploads, setUploads] = useState<Record<string, UploadProgress>>({})

  const start = useCallback(
    (files: FileList | File[]) => {
      Array.from(files).forEach((file) => {
        const key = `${file.name}-${file.size}-${Date.now()}`
        uploadVideo(file, (p) => {
          setUploads((u) => ({ ...u, [key]: p }))
          if (p.state === 'done') {
            qc.invalidateQueries({ queryKey: ['videos'] })
            setTimeout(() => setUploads((u) => { const { [key]: _, ...rest } = u; return rest }), 2500)
          }
        }).catch((e: Error) =>
          setUploads((u) => ({ ...u, [key]: { file, pct: 0, state: 'error', error: e.message } })),
        )
      })
    },
    [qc],
  )

  const list = Object.entries(uploads)

  return (
    <div>
      <div
        onDragOver={(e) => { e.preventDefault(); setDrag(true) }}
        onDragLeave={() => setDrag(false)}
        onDrop={(e) => { e.preventDefault(); setDrag(false); start(e.dataTransfer.files) }}
        onClick={() => inputRef.current?.click()}
        className={cn(
          'group flex cursor-pointer flex-col items-center justify-center gap-2 rounded-xl border border-dashed border-border-strong bg-bg-elev/50 px-6 py-8 text-center transition-colors hover:border-accent/60 hover:bg-accent/5',
          drag && 'border-accent bg-accent/10',
        )}
      >
        <span className="grid h-10 w-10 place-items-center rounded-full bg-accent/15 text-accent transition-transform group-hover:scale-110">
          <UploadCloud size={20} />
        </span>
        <div className="text-sm font-medium">Drop videos here or click to browse</div>
        <div className="text-xs text-fg-muted">MP4, MOV, MKV, WebM · multiple files welcome · indexing starts automatically</div>
        <input
          ref={inputRef}
          type="file"
          accept="video/*,.mkv"
          multiple
          hidden
          onChange={(e) => e.target.files && start(e.target.files)}
        />
      </div>
      <AnimatePresence>
        {list.length > 0 && (
          <motion.ul initial={{ opacity: 0, y: -4 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }} className="mt-3 space-y-2">
            {list.map(([k, u]) => (
              <li key={k} className="flex items-center gap-3 rounded-lg border border-border bg-bg-card px-3 py-2 text-sm">
                <FileVideo size={16} className="text-fg-muted" />
                <div className="min-w-0 flex-1">
                  <div className="flex justify-between gap-2">
                    <span className="truncate">{u.file.name}</span>
                    <span className="text-xs text-fg-muted">{fmtBytes(u.file.size)}</span>
                  </div>
                  <div className="mt-1 h-1 overflow-hidden rounded-full bg-border">
                    <div
                      className={cn('h-full transition-all', u.state === 'error' ? 'bg-err' : 'bg-accent')}
                      style={{ width: `${Math.round(u.pct * 100)}%` }}
                    />
                  </div>
                  {u.error && <div className="mt-1 text-xs text-err">{u.error}</div>}
                </div>
                {u.state === 'done' ? (
                  <CheckCircle2 size={16} className="text-ok" />
                ) : u.state === 'error' ? (
                  <XCircle size={16} className="text-err" />
                ) : (
                  <span className="w-10 text-right text-xs tabular-nums text-fg-muted">
                    {u.state === 'finalizing' ? '…' : `${Math.round(u.pct * 100)}%`}
                  </span>
                )}
              </li>
            ))}
          </motion.ul>
        )}
      </AnimatePresence>
    </div>
  )
}
