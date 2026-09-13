import { useQuery } from '@tanstack/react-query'
import { api } from './client'
import type { Job, MetricsSummary } from './types'

export function useMetrics(windowMin = 30) {
  return useQuery({
    queryKey: ['metrics', windowMin],
    queryFn: () => api<MetricsSummary>(`/api/metrics/summary?window_min=${windowMin}`),
    refetchInterval: 3000,
  })
}

export function useJobs(status?: string) {
  return useQuery({
    queryKey: ['jobs', status],
    queryFn: () => api<Job[]>(`/api/jobs?limit=40${status ? `&status=${status}` : ''}`),
    refetchInterval: 3000,
  })
}
