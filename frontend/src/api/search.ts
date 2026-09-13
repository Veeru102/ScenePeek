import { useMutation, useQuery } from '@tanstack/react-query'
import { api } from './client'
import type { SearchRequest, SearchResponse } from './types'

export function runSearch(req: SearchRequest) {
  return api<SearchResponse>('/api/search', { method: 'POST', body: JSON.stringify(req) })
}

export function useSearch() {
  return useMutation({ mutationFn: runSearch })
}

/** Query-key based search for pages that want caching/back-navigation (e.g. in-video search). */
export function useSearchQuery(req: SearchRequest | null) {
  return useQuery({
    queryKey: ['search', req],
    queryFn: () => runSearch(req!),
    enabled: !!req && req.q.trim().length > 0,
    staleTime: 60_000,
  })
}
