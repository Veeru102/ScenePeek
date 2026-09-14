import { Image, ScanText } from 'lucide-react'
import type { SearchHit } from '@/api/types'

const LANES: { key: string; label: string; color: string; hint: string }[] = [
  { key: 'text', label: 'Speech (dense)', color: 'var(--color-speech)', hint: 'cosine similarity of the transcript embedding' },
  { key: 'lexical', label: 'Keyword (FTS)', color: 'var(--color-lexical)', hint: 'Postgres full-text rank on the transcript' },
  { key: 'visual', label: 'Visual (SigLIP)', color: 'var(--color-visual)', hint: 'image–text similarity of the best keyframe' },
  { key: 'temporal', label: 'Motion (X-CLIP)', color: 'var(--color-visual)', hint: 'text–video similarity of the best 8-second window' },
  { key: 'caption', label: 'Caption', color: 'var(--color-visual)', hint: 'similarity to the keyframe caption' },
  { key: 'ocr', label: 'On-screen text', color: 'var(--color-ocr)', hint: 'full-text / trigram match on OCR text' },
]

/** How a hit earned its rank: the per-lane candidate scores, the fused score, and the reranker's say. */
export function WhyPanel({ hit }: { hit: SearchHit }) {
  const s = hit.signals
  const present = LANES.filter((l) => s[l.key] != null)
  const max = Math.max(...present.map((l) => s[l.key]), 1e-6)
  return (
    <div className="mt-2 space-y-2 rounded-lg border border-border bg-bg-elev/60 p-3 text-xs" role="region" aria-label="Why this result">
      <div className="space-y-1.5">
        {present.map((l) => (
          <div key={l.key} className="grid grid-cols-[120px_1fr_44px] items-center gap-2" title={l.hint}>
            <span className="truncate text-fg-muted">{l.label}</span>
            <div className="h-1.5 overflow-hidden rounded bg-bg-card">
              <div className="h-full rounded" style={{ width: `${Math.max(3, (s[l.key] / max) * 100)}%`, background: l.color }} />
            </div>
            <span className="text-right font-mono text-fg-muted">{s[l.key].toFixed(2)}</span>
          </div>
        ))}
        {present.length === 0 && <div className="text-fg-dim">no lane scores recorded</div>}
      </div>
      <div className="flex flex-wrap gap-x-4 gap-y-1 border-t border-border pt-2 font-mono text-fg-muted">
        {s.fused != null && <span>fused {s.fused.toFixed(4)}</span>}
        {s.rerank != null && <span>cross-encoder {s.rerank.toFixed(3)}</span>}
        <span>final {hit.score.toFixed(3)}</span>
      </div>
      {hit.caption_text && (
        <p className="flex items-start gap-1 text-visual/90">
          <Image size={12} className="mt-0.5 shrink-0" />
          <span>{hit.caption_text}</span>
        </p>
      )}
      {hit.ocr_text && (
        <p className="flex items-start gap-1 text-ocr/90">
          <ScanText size={12} className="mt-0.5 shrink-0" />
          <span className="line-clamp-3">{hit.ocr_text.replace(/\n/g, ' · ')}</span>
        </p>
      )}
    </div>
  )
}
