'use client';
import { useRef } from 'react';
export function useSpeechMetrics(){
  const questionEndedAt = useRef(Date.now());
  function markAiFinished(){ questionEndedAt.current = Date.now(); }
  function analyze(text:string){
    const words = text.trim().split(/\s+/).filter(Boolean).length;
    const latencyMs = Date.now() - questionEndedAt.current;
    const minutes = Math.max(0.1, latencyMs/60000);
    return { words, latencyMs, wpm: Math.round(words/minutes) };
  }
  return { markAiFinished, analyze };
}
