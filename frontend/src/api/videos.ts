import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { api } from './client'
import type { Topic, Utterance, Video, VideoDetail } from './types'

const isActive = (v: Video) => v.status === 'processing' || v.status === 'uploaded' || v.status === 'uploading'

export function useVideos() {
  return useQuery({
    queryKey: ['videos'],
    queryFn: () => api<Video[]>('/api/videos'),
    refetchInterval: (q) => (q.state.data?.some(isActive) ? 2000 : 10000),
  })
}

export function useVideo(id: string | undefined) {
  return useQuery({
    queryKey: ['video', id],
    queryFn: () => api<VideoDetail>(`/api/videos/${id}`),
    enabled: !!id,
    refetchInterval: (q) => (q.state.data && isActive(q.state.data) ? 2000 : false),
  })
}

export function useTranscript(id: string | undefined, enabled = true) {
  return useQuery({
    queryKey: ['transcript', id],
    queryFn: () => api<Utterance[]>(`/api/videos/${id}/transcript`),
    enabled: !!id && enabled,
  })
}

export function useTimeline(id: string | undefined, refetch = false) {
  return useQuery({
    queryKey: ['timeline', id],
    queryFn: () => api<Topic[]>(`/api/videos/${id}/timeline`),
    enabled: !!id,
    refetchInterval: refetch ? 4000 : false,
  })
}

export function useDeleteVideo() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (id: string) => api<void>(`/api/videos/${id}`, { method: 'DELETE' }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['videos'] }),
  })
}

export function useRetryVideo() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (id: string) => api<Video>(`/api/videos/${id}/retry`, { method: 'POST' }),
    onSuccess: (_d, id) => {
      qc.invalidateQueries({ queryKey: ['videos'] })
      qc.invalidateQueries({ queryKey: ['video', id] })
    },
  })
}

export interface UploadProgress {
  file: File
  videoId?: string
  pct: number
  state: 'pending' | 'uploading' | 'finalizing' | 'done' | 'error'
  error?: string
}

/** Presigned direct-to-storage upload with XHR so we get progress events. */
export async function uploadVideo(file: File, onProgress: (p: UploadProgress) => void): Promise<string> {
  const base: UploadProgress = { file, pct: 0, state: 'pending' }
  onProgress(base)
  const target = await api<{ video_id: string; upload_url: string; key: string }>('/api/videos', {
    method: 'POST',
    body: JSON.stringify({
      filename: file.name,
      content_type: file.type || 'video/mp4',
      size_bytes: file.size,
    }),
  })
  await new Promise<void>((resolve, reject) => {
    const xhr = new XMLHttpRequest()
    xhr.open('PUT', target.upload_url)
    xhr.setRequestHeader('Content-Type', file.type || 'video/mp4')
    xhr.upload.onprogress = (e) => {
      if (e.lengthComputable) onProgress({ ...base, videoId: target.video_id, state: 'uploading', pct: e.loaded / e.total })
    }
    xhr.onload = () => (xhr.status < 300 ? resolve() : reject(new Error(`upload failed (${xhr.status})`)))
    xhr.onerror = () => reject(new Error('upload failed (network)'))
    xhr.send(file)
  })
  onProgress({ ...base, videoId: target.video_id, state: 'finalizing', pct: 1 })
  await api(`/api/videos/${target.video_id}/complete`, { method: 'POST' })
  onProgress({ ...base, videoId: target.video_id, state: 'done', pct: 1 })
  return target.video_id
}
