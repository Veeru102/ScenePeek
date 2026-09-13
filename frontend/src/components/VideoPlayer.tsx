import { forwardRef, useEffect, useImperativeHandle, useRef } from 'react'

export interface PlayerHandle {
  seek: (t: number, play?: boolean) => void
  el: () => HTMLVideoElement | null
}

/** Thin wrapper over the native video element that exposes seek() and reports time updates. */
export const VideoPlayer = forwardRef<
  PlayerHandle,
  { src: string | null; poster?: string | null; onTime?: (t: number) => void; className?: string; startAt?: number | null; autoPlay?: boolean }
>(function VideoPlayer({ src, poster, onTime, className, startAt, autoPlay }, ref) {
  const vref = useRef<HTMLVideoElement>(null)
  useImperativeHandle(ref, () => ({
    seek: (t, play = true) => {
      const v = vref.current
      if (!v) return
      v.currentTime = Math.max(0, t)
      if (play) void v.play().catch(() => {})
    },
    el: () => vref.current,
  }))
  useEffect(() => {
    const v = vref.current
    if (!v || startAt == null) return
    const apply = () => {
      v.currentTime = startAt
    }
    if (v.readyState >= 1) apply()
    else v.addEventListener('loadedmetadata', apply, { once: true })
  }, [startAt, src])
  return (
    <video
      ref={vref}
      src={src ?? undefined}
      poster={poster ?? undefined}
      controls
      autoPlay={autoPlay}
      playsInline
      preload="metadata"
      onTimeUpdate={(e) => onTime?.((e.target as HTMLVideoElement).currentTime)}
      className={className}
    />
  )
})
