import * as React from 'react'
import { cn } from '@/lib/utils'

const tones: Record<string, string> = {
  neutral: 'bg-bg-elev text-fg-muted border-border',
  ok: 'bg-ok/10 text-ok border-ok/30',
  warn: 'bg-warn/10 text-warn border-warn/30',
  err: 'bg-err/10 text-err border-err/30',
  accent: 'bg-accent/10 text-accent border-accent/30',
  speech: 'bg-speech/10 text-speech border-speech/30',
  visual: 'bg-visual/10 text-visual border-visual/30',
  ocr: 'bg-ocr/10 text-ocr border-ocr/30',
  lexical: 'bg-lexical/10 text-lexical border-lexical/30',
}

export function Badge({
  tone = 'neutral',
  className,
  ...props
}: React.HTMLAttributes<HTMLSpanElement> & { tone?: keyof typeof tones }) {
  return (
    <span
      className={cn(
        'inline-flex items-center gap-1 rounded-md border px-2 py-0.5 text-[11px] font-medium leading-4',
        tones[tone],
        className,
      )}
      {...props}
    />
  )
}
