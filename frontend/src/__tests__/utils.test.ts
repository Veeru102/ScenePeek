import { describe, expect, it } from 'vitest'
import { fmtBytes, fmtTime } from '@/lib/utils'

describe('fmtTime', () => {
  it('formats seconds as m:ss and h:mm:ss', () => {
    expect(fmtTime(0)).toBe('0:00')
    expect(fmtTime(65)).toBe('1:05')
    expect(fmtTime(3725)).toBe('1:02:05')
  })
  it('handles missing values', () => {
    expect(fmtTime(null)).toBe('--:--')
    expect(fmtTime(Number.NaN)).toBe('--:--')
  })
})

describe('fmtBytes', () => {
  it('picks a sensible unit', () => {
    expect(fmtBytes(512)).toBe('512 B')
    expect(fmtBytes(1536)).toBe('1.5 KB')
    expect(fmtBytes(5 * 1024 * 1024)).toBe('5.0 MB')
  })
})
