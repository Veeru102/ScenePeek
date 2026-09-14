import { Eye, Image, Mic, ScanText, Sparkles, Type } from 'lucide-react'
import { Badge } from '@/components/ui/badge'

const meta: Record<string, { label: string; tone: 'speech' | 'visual' | 'ocr' | 'lexical' | 'accent'; icon: typeof Mic }> = {
  text: { label: 'Speech', tone: 'speech', icon: Mic },
  lexical: { label: 'Keyword', tone: 'lexical', icon: Type },
  visual: { label: 'Visual', tone: 'visual', icon: Eye },
  ocr: { label: 'On-screen', tone: 'ocr', icon: ScanText },
  caption: { label: 'Caption', tone: 'visual', icon: Image },
  rerank: { label: 'Rerank', tone: 'accent', icon: Sparkles },
}

/** Which modalities contributed to a hit, with their raw scores — the "why did this match" view. */
export function SignalChips({ signals, compact = false }: { signals: Record<string, number>; compact?: boolean }) {
  const order = ['text', 'lexical', 'visual', 'caption', 'ocr', 'rerank']
  return (
    <div className="flex flex-wrap gap-1">
      {order
        .filter((k) => signals[k] != null)
        .map((k) => {
          const m = meta[k]
          const Icon = m.icon
          return (
            <Badge key={k} tone={m.tone} title={`${m.label} score`}>
              <Icon size={11} />
              {!compact && m.label}
              <span className="font-mono opacity-80">{signals[k].toFixed(2)}</span>
            </Badge>
          )
        })}
    </div>
  )
}
