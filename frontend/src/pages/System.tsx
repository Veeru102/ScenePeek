import { useState } from 'react'
import { Activity, AlertTriangle, Cpu, Gauge, Search, Server, Timer, Zap } from 'lucide-react'
import { useJobs, useMetrics } from '@/api/system'
import { Badge } from '@/components/ui/badge'
import { cn, fmtDuration } from '@/lib/utils'

function Stat({
  icon: Icon,
  label,
  value,
  sub,
  tone,
}: {
  icon: typeof Zap
  label: string
  value: string
  sub?: string
  tone?: string
}) {
  return (
    <div className="rounded-xl border border-border bg-bg-card p-4">
      <div className="flex items-center gap-2 text-xs text-fg-muted">
        <Icon size={14} className={tone} /> {label}
      </div>
      <div className="mt-2 text-2xl font-semibold tabular-nums tracking-tight">{value}</div>
      {sub && <div className="mt-1 text-xs text-fg-dim">{sub}</div>}
    </div>
  )
}

const ms = (v?: number | null) => (v == null ? '—' : v >= 1000 ? `${(v / 1000).toFixed(1)} s` : `${v.toFixed(0)} ms`)
const secs = (v?: number | null) => (v == null ? '—' : fmtDuration(v))

const statusTone: Record<string, string> = {
  queued: 'text-warn',
  running: 'text-accent',
  succeeded: 'text-ok',
  failed: 'text-err',
  dead: 'text-err',
}

export function SystemPage() {
  const [win, setWin] = useState(30)
  const { data: m } = useMetrics(win)
  const { data: jobs } = useJobs()
  if (!m) return <div className="grid grid-cols-2 gap-4 lg:grid-cols-4">{Array.from({ length: 8 }).map((_, i) => <div key={i} className="h-24 rounded-xl shimmer" />)}</div>

  const js = m.jobs.by_status
  const inference = Object.entries(m.inference).sort((a, b) => (b[1].p50_ms ?? 0) - (a[1].p50_ms ?? 0))

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight">System</h1>
          <p className="mt-1 text-sm text-fg-muted">Processing throughput, queue health, and search latency. Live from Postgres.</p>
        </div>
        <div className="flex items-center gap-1 rounded-lg border border-border bg-bg-card p-1 text-xs">
          {[10, 30, 120, 1440].map((w) => (
            <button key={w} onClick={() => setWin(w)} className={cn('rounded-md px-2 py-1 text-fg-muted', win === w && 'bg-bg-elev text-fg')}>
              {w >= 60 ? `${w / 60}h` : `${w}m`}
            </button>
          ))}
        </div>
      </div>

      <div className="grid grid-cols-2 gap-4 lg:grid-cols-4">
        <Stat icon={Zap} tone="text-ok" label="Indexing speed" value={m.throughput.realtime_factor ? `${m.throughput.realtime_factor.toFixed(1)}× realtime` : '—'} sub={m.throughput.seconds_to_index_one_hour ? `≈ ${fmtDuration(m.throughput.seconds_to_index_one_hour)} per hour of video` : 'no chunks indexed in window'} />
        <Stat icon={Timer} tone="text-accent" label="Time to first searchable" value={secs(m.time_to_first_searchable_s.p50)} sub={`p95 ${secs(m.time_to_first_searchable_s.p95)} · n=${m.time_to_first_searchable_s.n}`} />
        <Stat icon={Gauge} tone="text-visual" label="Full index wall time" value={secs(m.index_wall_s.p50)} sub={m.index_wall_s.per_hour_s ? `≈ ${fmtDuration(m.index_wall_s.per_hour_s)} per hour incl. queueing` : `n=${m.index_wall_s.n}`} />
        <Stat icon={Search} tone="text-ocr" label="Search latency" value={ms(m.search.p50_ms)} sub={`p95 ${ms(m.search.p95_ms)} · ${m.search.n} searches`} />
      </div>

      <div className="grid gap-4 lg:grid-cols-3">
        <div className="rounded-xl border border-border bg-bg-card p-4">
          <div className="mb-3 flex items-center gap-2 text-sm font-medium"><Activity size={15} /> Job queue</div>
          <div className="grid grid-cols-2 gap-2 text-sm">
            {(['queued', 'running', 'succeeded', 'dead'] as const).map((k) => (
              <div key={k} className="flex items-center justify-between rounded-lg bg-bg-elev px-3 py-2">
                <span className={cn('capitalize', statusTone[k])}>{k}</span>
                <span className="font-mono tabular-nums">{js[k] ?? 0}</span>
              </div>
            ))}
          </div>
          <div className="mt-3 flex justify-between text-xs text-fg-muted">
            <span>retries total</span><span className="font-mono">{m.jobs.retries_total}</span>
          </div>
          <div className="mt-1 flex justify-between text-xs text-fg-muted">
            <span>chunks indexed (window)</span><span className="font-mono">{m.throughput.chunks} · {fmtDuration(m.throughput.indexed_video_seconds)} of video</span>
          </div>
        </div>

        <div className="rounded-xl border border-border bg-bg-card p-4">
          <div className="mb-3 flex items-center gap-2 text-sm font-medium"><Cpu size={15} /> Model inference (p50 / p95)</div>
          {inference.length === 0 && <div className="text-xs text-fg-dim">No samples in window.</div>}
          <div className="space-y-1.5 text-xs">
            {inference.map(([name, p]) => (
              <div key={name} className="flex items-center gap-2">
                <span className="w-28 truncate font-mono text-fg-muted" title={name}>{name}</span>
                <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-bg-elev">
                  <div className="h-full bg-accent" style={{ width: `${Math.min(100, ((p.p50_ms ?? 0) / Math.max(1, inference[0][1].p50_ms ?? 1)) * 100)}%` }} />
                </div>
                <span className="w-32 whitespace-nowrap text-right font-mono tabular-nums">{ms(p.p50_ms)} / {ms(p.p95_ms)}</span>
                <span className="w-10 text-right font-mono text-fg-dim">n={p.n}</span>
              </div>
            ))}
          </div>
        </div>

        <div className="rounded-xl border border-border bg-bg-card p-4">
          <div className="mb-3 flex items-center gap-2 text-sm font-medium"><Server size={15} /> Workers</div>
          {m.workers.length === 0 && <div className="text-xs text-fg-dim">No workers active in window. Start one with <code className="font-mono">make worker</code>.</div>}
          <div className="space-y-1.5 text-xs">
            {m.workers.map((w) => (
              <div key={w.id} className="flex items-center justify-between rounded-lg bg-bg-elev px-3 py-2">
                <span className="truncate font-mono">{w.id}</span>
                <span className="flex items-center gap-2 text-fg-muted">
                  {w.running > 0 && <Badge tone="accent">busy</Badge>}
                  <span className="font-mono">{w.jobs} jobs</span>
                </span>
              </div>
            ))}
          </div>
          {Object.keys(m.job_latency).length > 0 && (
            <div className="mt-3 space-y-1 border-t border-border pt-3 text-xs">
              {Object.entries(m.job_latency).map(([t, p]) => (
                <div key={t} className="flex justify-between text-fg-muted">
                  <span className="font-mono">{t}</span><span className="font-mono">{ms(p.p50_ms)} / {ms(p.p95_ms)}</span>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>

      <div className="grid gap-4 lg:grid-cols-2">
        <div className="rounded-xl border border-border bg-bg-card p-4">
          <div className="mb-3 text-sm font-medium">Recent jobs</div>
          <div className="scrollbar-thin max-h-80 overflow-auto text-xs">
            <table className="w-full">
              <thead className="text-left text-fg-dim">
                <tr><th className="pb-1 font-normal">type</th><th className="pb-1 font-normal">status</th><th className="pb-1 font-normal">attempt</th><th className="pb-1 font-normal">payload</th></tr>
              </thead>
              <tbody>
                {jobs?.map((j) => (
                  <tr key={j.id} className="border-t border-border/60">
                    <td className="py-1 font-mono">{j.type}</td>
                    <td className={cn('py-1', statusTone[j.status])}>{j.status}</td>
                    <td className="py-1 font-mono">{j.attempts}/{j.max_attempts}</td>
                    <td className="py-1 font-mono text-fg-muted">{j.payload.chunk_index != null ? `chunk ${String(j.payload.chunk_index)}` : ''} {String(j.video_id ?? '').slice(0, 8)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
        <div className="rounded-xl border border-border bg-bg-card p-4">
          <div className="mb-3 flex items-center gap-2 text-sm font-medium"><AlertTriangle size={15} className="text-warn" /> Failures &amp; retries</div>
          {m.recent_failures.length === 0 && <div className="text-xs text-fg-dim">No failures recorded.</div>}
          <div className="space-y-2 text-xs">
            {m.recent_failures.map((f) => (
              <div key={f.id} className="rounded-lg bg-bg-elev p-2">
                <div className="flex items-center justify-between">
                  <span className="font-mono">{f.type} <span className={statusTone[f.status]}>{f.status}</span></span>
                  <span className="font-mono text-fg-dim">attempt {f.attempts}/{f.max_attempts}</span>
                </div>
                <div className="mt-1 line-clamp-2 font-mono text-[11px] text-fg-muted">{f.last_error}</div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  )
}
