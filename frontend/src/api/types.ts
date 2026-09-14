export type VideoStatus = 'uploading' | 'uploaded' | 'processing' | 'ready' | 'failed'
export type ChunkStatus = 'pending' | 'running' | 'done' | 'failed'

export interface Chunk {
  index: number
  start_s: number
  end_s: number
  status: ChunkStatus
  stage: string | null
  error: string | null
}

export interface Video {
  id: string
  title: string
  status: VideoStatus
  error: string | null
  duration_s: number | null
  width: number | null
  height: number | null
  size_bytes: number | null
  chunk_count: number
  chunks_done: number
  chunks_failed: number
  progress: number
  poster_url: string | null
  playback_url: string | null
  created_at: string
  upload_completed_at: string | null
  first_searchable_at: string | null
  completed_at: string | null
  topic_count: number
}

export interface VideoDetail extends Video {
  chunks: Chunk[]
}

export interface Utterance {
  start_s: number
  end_s: number
  text: string
  words: { w: string; s: number; e: number }[] | null
}

export interface KeyFrame {
  t_s: number
  url: string
  caption: string
}

export interface Topic {
  index: number
  start_s: number
  end_s: number
  label: string
  keyphrases: string[] | null
}

export interface SearchWeights {
  text?: number
  lexical?: number
  visual?: number
  ocr?: number
  caption?: number
  temporal?: number
}

export interface SearchRequest {
  q: string
  video_ids?: string[]
  limit?: number
  weights?: SearchWeights
  rerank?: boolean
  fusion?: 'rrf' | 'weighted'
}

export interface SearchHit {
  segment_id: string
  video: { id: string; title: string; duration_s: number | null; poster_url: string | null; playback_url: string | null }
  start_s: number
  end_s: number
  score: number
  signals: Record<string, number>
  text: string
  snippet_html: string
  ocr_text: string
  caption_text?: string
  keyframe_url: string | null
}

export interface SearchResponse {
  query: string
  plan: Record<string, unknown>
  hits: SearchHit[]
  timings_ms: Record<string, number>
  total_candidates: number
}

export interface Job {
  id: string
  type: string
  queue: string
  status: string
  priority: number
  attempts: number
  max_attempts: number
  video_id: string | null
  payload: Record<string, unknown>
  last_error: string | null
  locked_by: string | null
  run_after: string
  started_at: string | null
  finished_at: string | null
  created_at: string
}

export interface Pct {
  p50_ms?: number | null
  p95_ms?: number | null
  p50?: number | null
  p95?: number | null
  n: number
}

export interface MetricsSummary {
  window_min: number
  jobs: { by_status: Record<string, number>; by_queue_status: Record<string, number>; retries_total: number }
  videos: Record<string, number>
  throughput: {
    indexed_video_seconds: number
    chunks: number
    worker_busy_seconds: number
    realtime_factor: number | null
    wall_seconds: number | null
    seconds_to_index_one_hour: number | null
  }
  time_to_first_searchable_s: Pct
  index_wall_s: Pct & { per_hour_s: number | null }
  search: Pct
  inference: Record<string, Pct>
  job_latency: Record<string, Pct>
  workers: { id: string; last_seen: string; jobs: number; running: number }[]
  recent_failures: {
    id: string
    type: string
    status: string
    attempts: number
    max_attempts: number
    video_id: string | null
    last_error: string | null
    created_at: string
    finished_at: string | null
  }[]
}
