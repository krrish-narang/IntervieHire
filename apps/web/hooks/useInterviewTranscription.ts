'use client';

import { useCallback, useEffect, useRef, useState } from 'react';
import { API_URL } from '@/lib/api';

export type InterviewTranscriptEntry = {
  text: string;
  timestamp: string;
};

type SpeechRecognitionEventLike = Event & {
  resultIndex: number;
  results: {
    length: number;
    [index: number]: {
      isFinal: boolean;
      0: { transcript: string };
    };
  };
};

type SpeechRecognitionErrorEventLike = Event & {
  error: string;
};

type SpeechRecognitionLike = {
  continuous: boolean;
  interimResults: boolean;
  lang: string;
  onresult: ((event: SpeechRecognitionEventLike) => void) | null;
  onerror: ((event: SpeechRecognitionErrorEventLike) => void) | null;
  onend: (() => void) | null;
  start: () => void;
  stop: () => void;
  abort: () => void;
};

type SpeechRecognitionConstructor = new () => SpeechRecognitionLike;

declare global {
  interface Window {
    SpeechRecognition?: SpeechRecognitionConstructor;
    webkitSpeechRecognition?: SpeechRecognitionConstructor;
  }
}

const TRANSCRIPT_SAVE_INTERVAL_MS = 5000;
const LIVE_TRANSCRIPT_ID = 'browser-speech-recognition';

function getSpeechRecognitionConstructor() {
  if (typeof window === 'undefined') return undefined;
  return window.SpeechRecognition ?? window.webkitSpeechRecognition;
}

export function useInterviewTranscription(sessionId: string) {
  const [micPermissionGranted, setMicPermissionGranted] = useState(false);
  const [isRequestingMicrophone, setIsRequestingMicrophone] = useState(false);
  const [isTranscribing, setIsTranscribing] = useState(false);
  const [interviewTranscript, setInterviewTranscript] = useState<InterviewTranscriptEntry[]>([]);
  const [interimTranscript, setInterimTranscript] = useState('');
  const [microphoneError, setMicrophoneError] = useState<string | null>(null);
  const [transcriptionError, setTranscriptionError] = useState<string | null>(null);

  const microphoneStreamRef = useRef<MediaStream | null>(null);
  const recognitionRef = useRef<SpeechRecognitionLike | null>(null);
  const transcriptRef = useRef<InterviewTranscriptEntry[]>([]);
  const shouldTranscribeRef = useRef(false);
  const lastSavedPayloadRef = useRef('');
  const createdAtRef = useRef(new Date().toISOString());

  const speechRecognitionSupported = typeof window === 'undefined' || Boolean(getSpeechRecognitionConstructor());

  const persistTranscript = useCallback(async (finalized = false, useBeacon = false) => {
    const transcript = transcriptRef.current;
    if (!transcript.length) return true;

    const payload = JSON.stringify({
      transcriptId: LIVE_TRANSCRIPT_ID,
      transcript,
      fullText: transcript.map((entry) => entry.text).join(' '),
      finalized,
      createdAt: createdAtRef.current,
    });

    if (!finalized && payload === lastSavedPayloadRef.current) return true;

    const url = `${API_URL}/api/interview/sessions/${sessionId}/transcript`;
    if (useBeacon && typeof navigator !== 'undefined' && navigator.sendBeacon) {
      const queued = navigator.sendBeacon(url, new Blob([payload], { type: 'application/json' }));
      if (queued) lastSavedPayloadRef.current = payload;
      return queued;
    }

    try {
      const response = await fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: payload,
        keepalive: finalized,
      });
      if (!response.ok) throw new Error(`Transcript save failed (${response.status})`);
      lastSavedPayloadRef.current = payload;
      return true;
    } catch (error) {
      console.error('Transcript save failed', error);
      setTranscriptionError('The live transcript could not be saved. We will retry automatically.');
      return false;
    }
  }, [sessionId]);

  const requestMicrophonePermission = useCallback(async () => {
    setIsRequestingMicrophone(true);
    setMicrophoneError(null);

    try {
      if (!navigator.mediaDevices?.getUserMedia) {
        throw new Error('Microphone access is not supported in this browser.');
      }

      const currentStream = microphoneStreamRef.current;
      if (currentStream?.getAudioTracks().some((track) => track.readyState === 'live')) {
        setMicPermissionGranted(true);
        return true;
      }

      currentStream?.getTracks().forEach((track) => track.stop());
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        },
        video: false,
      });

      microphoneStreamRef.current = stream;
      setMicPermissionGranted(true);

      stream.getAudioTracks().forEach((track) => {
        track.onended = () => {
          if (microphoneStreamRef.current === stream) {
            setMicPermissionGranted(false);
            setMicrophoneError('Microphone access ended. Enable it again before starting the interview.');
          }
        };
      });

      return true;
    } catch (error) {
      const permissionDenied =
        error instanceof DOMException && (error.name === 'NotAllowedError' || error.name === 'SecurityError');
      setMicPermissionGranted(false);
      setMicrophoneError(
        permissionDenied
          ? 'Microphone access was denied. Allow it in your browser settings, then click Enable Microphone again.'
          : error instanceof Error
            ? error.message
            : 'Unable to access the microphone.',
      );
      return false;
    } finally {
      setIsRequestingMicrophone(false);
    }
  }, []);

  const startTranscription = useCallback(() => {
    if (!micPermissionGranted) {
      setTranscriptionError('Enable microphone access before starting transcription.');
      return false;
    }

    const SpeechRecognition = getSpeechRecognitionConstructor();
    if (!SpeechRecognition) {
      setTranscriptionError(
        'Live speech-to-text is not supported in this browser. Use a Chromium-based browser such as Chrome or Edge.',
      );
      return false;
    }

    setTranscriptionError(null);
    shouldTranscribeRef.current = true;

    const recognition = new SpeechRecognition();
    recognition.continuous = true;
    recognition.interimResults = true;
    recognition.lang = 'en-IN';
    recognition.onresult = (event) => {
      const finalizedEntries: InterviewTranscriptEntry[] = [];
      let interim = '';

      for (let index = event.resultIndex; index < event.results.length; index += 1) {
        const result = event.results[index];
        const text = result[0]?.transcript?.trim();
        if (!text) continue;

        if (result.isFinal) {
          finalizedEntries.push({ text, timestamp: new Date().toISOString() });
        } else {
          interim += `${text} `;
        }
      }

      setInterimTranscript(interim.trim());
      if (finalizedEntries.length) {
        const updated = [...transcriptRef.current, ...finalizedEntries];
        transcriptRef.current = updated;
        setInterviewTranscript(updated);
      }
    };
    recognition.onerror = (event) => {
      if (event.error === 'no-speech' || event.error === 'aborted') return;
      if (event.error === 'not-allowed' || event.error === 'service-not-allowed') {
        shouldTranscribeRef.current = false;
        setMicPermissionGranted(false);
      }
      setTranscriptionError(`Speech-to-text stopped: ${event.error.replaceAll('-', ' ')}.`);
    };
    recognition.onend = () => {
      if (shouldTranscribeRef.current) {
        window.setTimeout(() => {
          try {
            recognition.start();
          } catch {
            // The browser can throw while the previous recognition cycle is still closing.
          }
        }, 250);
      } else {
        setIsTranscribing(false);
      }
    };

    recognitionRef.current = recognition;
    try {
      recognition.start();
      setIsTranscribing(true);
      return true;
    } catch (error) {
      shouldTranscribeRef.current = false;
      recognitionRef.current = null;
      setTranscriptionError(error instanceof Error ? error.message : 'Unable to start speech-to-text.');
      return false;
    }
  }, [micPermissionGranted]);

  const stopTranscription = useCallback(async () => {
    shouldTranscribeRef.current = false;
    setInterimTranscript('');

    try {
      recognitionRef.current?.stop();
    } catch {
      recognitionRef.current?.abort();
    }
    recognitionRef.current = null;
    setIsTranscribing(false);

    await new Promise((resolve) => window.setTimeout(resolve, 300));
    return persistTranscript(true);
  }, [persistTranscript]);

  const releaseMicrophone = useCallback(() => {
    microphoneStreamRef.current?.getTracks().forEach((track) => {
      track.onended = null;
      track.stop();
    });
    microphoneStreamRef.current = null;
    setMicPermissionGranted(false);
  }, []);

  const getMicrophoneStream = useCallback(() => microphoneStreamRef.current, []);

  useEffect(() => {
    transcriptRef.current = [];
    setInterviewTranscript([]);
    setInterimTranscript('');
    lastSavedPayloadRef.current = '';
    createdAtRef.current = new Date().toISOString();
  }, [sessionId]);

  useEffect(() => {
    if (!isTranscribing) return;
    const timer = window.setInterval(() => {
      void persistTranscript(false);
    }, TRANSCRIPT_SAVE_INTERVAL_MS);
    return () => window.clearInterval(timer);
  }, [isTranscribing, persistTranscript]);

  useEffect(() => {
    const handlePageHide = () => {
      void persistTranscript(true, true);
    };
    window.addEventListener('pagehide', handlePageHide);
    return () => window.removeEventListener('pagehide', handlePageHide);
  }, [persistTranscript]);

  useEffect(() => {
    return () => {
      shouldTranscribeRef.current = false;
      recognitionRef.current?.abort();
      microphoneStreamRef.current?.getTracks().forEach((track) => track.stop());
      void persistTranscript(true, true);
    };
  }, [persistTranscript]);

  return {
    micPermissionGranted,
    isRequestingMicrophone,
    isTranscribing,
    interviewTranscript,
    interimTranscript,
    microphoneError,
    transcriptionError,
    speechRecognitionSupported,
    requestMicrophonePermission,
    startTranscription,
    stopTranscription,
    releaseMicrophone,
    getMicrophoneStream,
    flushTranscript: persistTranscript,
  };
}
