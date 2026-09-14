import { useEffect, useRef, useState } from 'react'
import { useSearchParams } from 'react-router-dom'
import { AnimatePresence, motion } from 'framer-motion'
import { ChevronDown, SlidersHorizontal, X } from 'lucide-react'
import { useSearch } from '@/api/search'
import type { SearchHit, SearchWeights } from '@/api/types'
import { ResultCard } from '@/components/ResultCard'
import { SearchBar } from '@/components/SearchBar'
import { VideoPlayer, type PlayerHandle } from '@/components/VideoPlayer'
import { Badge } from '@/components/ui/badge'
import { cn, fmtTime } from '@/lib/utils'

const EXAMPLES = [
  'where the professor explains B+ tree leaf nodes',
  'someone holding a red umbrella',
  'mentions latency while showing an architecture diagram',
  'slide titled "Transactions"',
]

const LANES = ['text', 'lexical', 'visual', 'caption', 'ocr'] as const

function readControls(params: URLSearchParams) {
  const weights: SearchWeights = {}
  for (const k of LANES) {
    const v = params.get(`w.${k}`)
    if (v != null && !Number.isNaN(Number(v))) weights[k] = Number(v)
  }
  return {
    weights,
    rerank: params.get('rerank') !== '0',
    fusion: (params.get('fusion') === 'rrf' ? 'rrf' : 'weighted') as 'rrf' | 'weighted',
  }
}

export function SearchPage() {
  const [params, setParams] = useSearchParams()
  const q = params.get('q') ?? ''
  const search = useSearch()
  const initial = readControls(params)
  const [advanced, setAdvanced] = useState(Object.keys(initial.weights).length > 0 || !initial.rerank || initial.fusion !== 'weighted')
  const [weights, setWeights] = useState<SearchWeights>(initial.weights)
  const [rerank, setRerank] = useState(initial.rerank)
  const [fusion, setFusion] = useState<'rrf' | 'weighted'>(initial.fusion)
  const [playing, setPlaying] = useState<SearchHit | null>(null)
  const player = useRef<PlayerHandle>(null)

  const run = (query: string) => {
    // ranking controls live in the URL so a tuned search is shareable / survives reload
    const next: Record<string, string> = { q: query }
    for (const k of LANES) if (weights[k] != null) next[`w.${k}`] = String(weights[k])
    if (!rerank) next.rerank = '0'
    if (fusion !== 'weighted') next.fusion = fusion
    setParams(next)
    search.mutate({ q: query, limit: 20, weights, rerank, fusion })
  }
  useEffect(() => {
    if (q) search.mutate({ q, limit: 20, weights, rerank, fusion })
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const t = e.target as HTMLElement | null
      if (e.key === '/' && t?.tagName !== 'INPUT' && t?.tagName !== 'TEXTAREA') {
        e.preventDefault()
        document.querySelector<HTMLInputElement>('input[data-search-input]')?.focus()
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [])

  const res = search.data
  const hits = res?.hits ?? []
  const plan = res?.plan as { speech_q?: string; visual_q?: string; ocr_q?: string; cues?: string[]; source?: string } | undefined

  return (
    <div className="space-y-6">
      <div className={cn('mx-auto max-w-3xl space-y-4 transition-all', !res && 'pt-16')}>
        {!res && (
          <div className="text-center">
            <h1 className="text-3xl font-semibold tracking-tight">Find any moment</h1>
            <p className="mt-2 text-sm text-fg-muted">
              Search what was <span className="text-speech">said</span>, what was <span className="text-visual">shown</span>, and what was{' '}
              <span className="text-ocr">on screen</span> — across your whole library.
            </p>
          </div>
        )}
        <SearchBar initial={q} onSubmit={run} loading={search.isPending} autoFocus />
        {!res && (
          <div className="flex flex-wrap justify-center gap-2">
            {EXAMPLES.map((e) => (
              <button key={e} onClick={() => run(e)} className="rounded-full border border-border bg-bg-card px-3 py-1 text-xs text-fg-muted transition-colors hover:border-accent/50 hover:text-fg">
                {e}
              </button>
            ))}
          </div>
        )}
        <div className="flex items-center justify-between text-xs text-fg-muted">
          <button onClick={() => setAdvanced((a) => !a)} className="inline-flex items-center gap-1 hover:text-fg">
            <SlidersHorizontal size={13} /> Ranking controls <ChevronDown size={13} className={cn('transition-transform', advanced && 'rotate-180')} />
          </button>
          {res && (
            <span className="font-mono">
              {res.total_candidates} candidates · {res.timings_ms.total.toFixed(0)} ms
              {res.timings_ms.rerank != null && ` (rerank ${res.timings_ms.rerank.toFixed(0)} ms)`}
            </span>
          )}
        </div>
        <AnimatePresence>
          {advanced && (
            <motion.div initial={{ opacity: 0, height: 0 }} animate={{ opacity: 1, height: 'auto' }} exit={{ opacity: 0, height: 0 }} className="overflow-hidden">
              <div className="grid grid-cols-2 gap-4 rounded-xl border border-border bg-bg-card p-4 text-xs sm:grid-cols-5">
                {(['text', 'lexical', 'visual', 'caption', 'ocr'] as const).map((k) => (
                  <label key={k} className="space-y-1">
                    <div className="flex justify-between capitalize text-fg-muted">
                      <span>{k === 'text' ? 'speech' : k === 'ocr' ? 'on-screen' : k}</span>
                      <span className="font-mono">{(weights[k] ?? { text: 1, lexical: 0.8, visual: 0.8, caption: 0, ocr: 0.6 }[k]).toFixed(1)}</span>
                    </div>
                    <input
                      type="range" min={0} max={2} step={0.1}
                      value={weights[k] ?? { text: 1, lexical: 0.8, visual: 0.8, caption: 0, ocr: 0.6 }[k]}
                      onChange={(e) => setWeights((w) => ({ ...w, [k]: Number(e.target.value) }))}
                      className="w-full accent-[var(--color-accent)]"
                    />
                  </label>
                ))}
                <label className="flex items-center gap-2"><input type="checkbox" checked={rerank} onChange={(e) => setRerank(e.target.checked)} /> Cross-encoder rerank</label>
                <label className="flex items-center gap-2">
                  Fusion
                  <select value={fusion} onChange={(e) => setFusion(e.target.value as 'rrf' | 'weighted')} className="rounded border border-border bg-bg-elev px-1 py-0.5">
                    <option value="rrf">RRF</option>
                    <option value="weighted">Weighted</option>
                  </select>
                </label>
                <button onClick={() => q && run(q)} className="col-span-2 justify-self-end rounded-md bg-accent px-3 py-1 font-medium text-white sm:col-span-2">
                  Re-run
                </button>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
        {plan && (
          <div className="flex flex-wrap items-center gap-1.5 text-xs text-fg-muted">
            <span>Query plan:</span>
            {plan.speech_q && <Badge tone="speech">said: {plan.speech_q}</Badge>}
            {plan.visual_q && plan.visual_q !== plan.speech_q && <Badge tone="visual">shown: {plan.visual_q}</Badge>}
            {plan.ocr_q && plan.ocr_q !== plan.speech_q && plan.ocr_q !== plan.visual_q && <Badge tone="ocr">on screen: {plan.ocr_q}</Badge>}
            {plan.cues?.map((c) => <Badge key={c}>{c}</Badge>)}
            {plan.source === 'llm' && <Badge tone="accent">llm</Badge>}
          </div>
        )}
      </div>

      {search.isError && (
        <div role="alert" className="mx-auto max-w-3xl rounded-xl border border-err/40 bg-err/10 p-4 text-sm">
          <div className="font-medium">Search failed</div>
          <div className="mt-1 text-fg-muted">{(search.error as Error)?.message ?? 'The API did not respond.'}</div>
          <button onClick={() => q && run(q)} className="mt-3 rounded-md border border-border px-3 py-1 text-xs hover:border-accent/60">Retry</button>
        </div>
      )}
      {search.isPending && !res && (
        <div className="mx-auto max-w-3xl space-y-3" aria-busy="true" aria-label="Searching">
          {[0, 1, 2].map((i) => (
            <div key={i} className="flex animate-pulse gap-3 rounded-xl border border-border bg-bg-card p-3">
              <div className="aspect-video w-40 rounded-lg bg-bg-elev sm:w-48" />
              <div className="flex-1 space-y-2 py-1">
                <div className="h-3 w-1/3 rounded bg-bg-elev" />
                <div className="h-3 w-full rounded bg-bg-elev" />
                <div className="h-3 w-5/6 rounded bg-bg-elev" />
              </div>
            </div>
          ))}
        </div>
      )}

      {res && (
        <div className={cn('grid gap-6', playing ? 'lg:grid-cols-[minmax(0,1fr)_420px]' : '')}>
          <div className="min-w-0 space-y-3">
            {hits.length === 0 && <div className="rounded-xl border border-border py-12 text-center text-sm text-fg-muted">No matches. Try different wording or index more videos.</div>}
            {hits.map((h, i) => (
              <ResultCard
                key={h.segment_id}
                hit={h}
                index={i}
                active={playing?.segment_id === h.segment_id}
                onPlay={(hit) => {
                  if (playing?.video.id === hit.video.id) player.current?.seek(hit.start_s)
                  setPlaying(hit)
                }}
              />
            ))}
          </div>
          <AnimatePresence>
            {playing && (
              <motion.aside
                initial={{ opacity: 0, x: 16 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: 16 }}
                className="lg:sticky lg:top-20 lg:self-start"
              >
                <div className="overflow-hidden rounded-xl border border-border bg-bg-card">
                  <VideoPlayer ref={player} src={playing.video.playback_url} startAt={playing.start_s} autoPlay className="aspect-video w-full bg-black" />
                  <div className="space-y-1 p-3">
                    <div className="flex items-start justify-between gap-2">
                      <div className="text-sm font-medium leading-snug">{playing.video.title}</div>
                      <button onClick={() => setPlaying(null)} className="text-fg-dim hover:text-fg" aria-label="Close player"><X size={15} /></button>
                    </div>
                    <div className="font-mono text-xs text-fg-muted">
                      playing from {fmtTime(playing.start_s)} · segment {fmtTime(playing.start_s)}–{fmtTime(playing.end_s)}
                    </div>
                  </div>
                </div>
              </motion.aside>
            )}
          </AnimatePresence>
        </div>
      )}
    </div>
  )
}
