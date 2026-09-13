import { BrowserRouter, Route, Routes } from 'react-router-dom'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { Layout } from '@/components/Layout'
import { LibraryPage } from '@/pages/Library'
import { SearchPage } from '@/pages/Search'
import { SystemPage } from '@/pages/System'
import { VideoDetailPage } from '@/pages/VideoDetail'

const queryClient = new QueryClient({
  defaultOptions: { queries: { staleTime: 2000, retry: 1 } },
})

export default function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <Routes>
          <Route element={<Layout />}>
            <Route path="/" element={<LibraryPage />} />
            <Route path="/search" element={<SearchPage />} />
            <Route path="/videos/:id" element={<VideoDetailPage />} />
            <Route path="/system" element={<SystemPage />} />
          </Route>
        </Routes>
      </BrowserRouter>
    </QueryClientProvider>
  )
}
