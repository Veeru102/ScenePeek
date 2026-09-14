import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import { SignalChips } from '@/components/SignalChips'
import { WhyPanel } from '@/components/WhyPanel'
import type { SearchHit } from '@/api/types'

const hit: SearchHit = {
  segment_id: 's1',
  video: { id: 'v1', title: 'Lecture', duration_s: 100, poster_url: null, playback_url: null },
  start_s: 12,
  end_s: 22,
  score: 0.71,
  signals: { text: 0.8, caption: 0.64, ocr: 0.33, rerank: 0.9, fused: 0.04 },
  text: 'hello',
  snippet_html: 'hello',
  ocr_text: 'SLIDE TITLE',
  caption_text: 'a robot standing in a room',
  keyframe_url: null,
}

describe('SignalChips', () => {
  it('renders one chip per lane present, including captions, and skips missing lanes', () => {
    render(<SignalChips signals={hit.signals} />)
    expect(screen.getByText('Speech')).toBeInTheDocument()
    expect(screen.getByText('Caption')).toBeInTheDocument()
    expect(screen.getByText('On-screen')).toBeInTheDocument()
    expect(screen.queryByText('Visual')).not.toBeInTheDocument()
  })
})

describe('WhyPanel', () => {
  it('explains the ranking with per-lane scores, fusion, rerank, caption and OCR text', () => {
    render(<WhyPanel hit={hit} />)
    expect(screen.getByRole('region', { name: /why this result/i })).toBeInTheDocument()
    expect(screen.getByText('Caption')).toBeInTheDocument()
    expect(screen.getByText('0.64')).toBeInTheDocument()
    expect(screen.getByText(/cross-encoder 0.900/)).toBeInTheDocument()
    expect(screen.getByText(/final 0.710/)).toBeInTheDocument()
    expect(screen.getByText('a robot standing in a room')).toBeInTheDocument()
    expect(screen.getByText('SLIDE TITLE')).toBeInTheDocument()
  })
})
