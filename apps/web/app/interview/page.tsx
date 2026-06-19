'use client';
 import { useEffect, useRef, useState } from 'react';
 import { WS_URL, API_URL, parseApiResponse } from '@/lib/api';
 import { GazeCalibration } from '@/hooks/GazeCalibration';
 import { useInterviewTranscription } from '@/hooks/useInterviewTranscription';
 import { useProctoring } from '@/hooks/useProctoring';
 import { useMediaDeviceMonitoring } from '@/hooks/useMediaDeviceMonitoring';
 import { useSpeechMetrics } from '@/hooks/useSpeechMetrics';
 import { BarChart3, Maximize2, Mic, MonitorUp, Send, ShieldCheck, Timer, Video } from 'lucide-react';
 import type { CalibrationResult } from '@/hooks/useGazeCalibration';

type SpeechRecognitionInstance = {
  continuous: boolean;
  interimResults: boolean;
  lang: string;
  onstart: (() => void) | null;
  onend: (() => void) | null;
  onerror: ((event: { error?: string }) => void) | null;
  onresult: ((event: {
    resultIndex: number;
    results: ArrayLike<{
      isFinal: boolean;
      0: { transcript: string };
    }>;
  }) => void) | null;
  start: () => void;
  stop: () => void;
};

type SpeechRecognitionConstructor = new () => SpeechRecognitionInstance;

function getSpeechRecognitionConstructor(): SpeechRecognitionConstructor | null {
  if (typeof window === 'undefined') return null;
  const speechWindow = window as Window & {
    SpeechRecognition?: SpeechRecognitionConstructor;
    webkitSpeechRecognition?: SpeechRecognitionConstructor;
  };

  return speechWindow.SpeechRecognition ?? speechWindow.webkitSpeechRecognition ?? null;
}

 export default function Interview(){
   const [sessionId,setSessionId]=useState('demo-session');
   const [calibration, setCalibration] = useState<CalibrationResult | null>(null);
  const [socket,setSocket]=useState<WebSocket|null>(null);
  const [messages,setMessages]=useState<any[]>([{speaker:'ai',text:'Welcome. I will ask a few structured questions. Please answer naturally with examples.'}]);
  const [text,setText]=useState('');
  const [duration, setDuration] = useState('');
  const {markAiFinished, analyze}=useSpeechMetrics();
  const wsRef=useRef<WebSocket|null>(null);

  useEffect(()=>{
    let alive = true;
    let bootstrapTimer: number | undefined;
    let bootstrapAttempts = 0;

    async function bootstrapDemoSession() {
      if (sessionId !== 'demo-session') return;
      try {
        const res = await fetch(`${API_URL}/api/interview/demo-session`);
        const json = await parseApiResponse<{ sessionId?: string }>(res);
        if (alive && json?.sessionId) {
          setSessionId(json.sessionId);
        }
      } catch (error) {
        console.error('demo-session bootstrap failed', error);
        bootstrapAttempts += 1;
        if (alive && bootstrapAttempts < 3) {
          bootstrapTimer = window.setTimeout(bootstrapDemoSession, 5000);
        }
      }
    }
    bootstrapDemoSession();
    const ws=new WebSocket(WS_URL);
    wsRef.current=ws;
    ws.onopen=()=>ws.send(JSON.stringify({type:'register',role:'candidate',sessionId}));
    ws.onmessage=(e)=>{const msg=JSON.parse(e.data); if(msg.type==='ai_response'){setMessages(m=>[...m,{speaker:'ai',text:msg.text}]); markAiFinished();}};
    setSocket(ws);
    return()=>{
      alive = false;
      if (bootstrapTimer !== undefined) window.clearTimeout(bootstrapTimer);
      ws.close();
    };
  },[sessionId]);

  const {
    videoRef,
    events,
    emit, state,
    requestRequiredPermissions,
    finalProctoringScore,
    showProctoringScore,
    sessionStarted,
    startProctoringSession,
    endProctoringSession,
  } = useProctoring(sessionId, socket, calibration);
  useMediaDeviceMonitoring({
    enabled: state.cameraActive,
    emit,
  });
  const videoElement = (
    <video
      ref={videoRef}
      muted
      playsInline
      className="absolute bottom-5 right-5 h-36 w-52 rounded-2xl border border-white/20 object-cover shadow-2xl"
      style={{ display: calibration ? undefined : 'none' }}
    />
  );
  const [isRecording, setIsRecording] = useState(false);
  const [isStartingSession, setIsStartingSession] = useState(false);
  const [recordingStatus, setRecordingStatus] = useState('Idle');
  const [speechSupported, setSpeechSupported] = useState(false);
  const [speechStatus, setSpeechStatus] = useState('Speech-to-text idle');
  const [isListening, setIsListening] = useState(false);
  const [isTranscribingAnswer, setIsTranscribingAnswer] = useState(false);
  const mediaRecorderRef = useRef<MediaRecorder|null>(null);
  const recordedChunksRef = useRef<BlobPart[]>([]);
  const answerRecorderRef = useRef<MediaRecorder | null>(null);
  const answerChunksRef = useRef<BlobPart[]>([]);
  const proctoringReportRef = useRef<HTMLDivElement | null>(null);
  const speechRecognitionRef = useRef<SpeechRecognitionInstance | null>(null);
  const [sessionData, setSessionData] = useState<any|null>(null);
  const [evaluationReport, setEvaluationReport] = useState<any|null>(null);
  const [isEvaluating, setIsEvaluating] = useState(false);
  const [isSubmittingAnswer, setIsSubmittingAnswer] = useState(false);
  const [answerStatus, setAnswerStatus] = useState('Type your answer, then submit it for processing.');

  useEffect(() => {
    const supported = Boolean(getSpeechRecognitionConstructor());
    setSpeechSupported(supported);
    if (!supported) {
      setSpeechStatus('Browser live captions unavailable. Record spoken answers for transcription.');
    }
    return () => {
      speechRecognitionRef.current?.stop();
      speechRecognitionRef.current = null;
    };
  }, []);
  const {
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
  } = useInterviewTranscription(sessionId);

  useEffect(()=>{
    let mounted = true;
    async function load(){
      try{
        const res = await fetch(`${API_URL}/api/interview/sessions/${sessionId}`);
        if(!res.ok) return;
        const json = await parseApiResponse<any>(res);
        if(mounted) {
          setSessionData(json);
          if (json?.evaluation) setEvaluationReport(json.evaluation);
        }
      }catch(e){/*ignore*/}
    }
    load();
    const t = setInterval(load, 5000);
    return ()=>{ mounted = false; clearInterval(t); };
  },[sessionId]);

  async function startRecording(){
    try{
      if(!videoRef.current) return;
      const original = videoRef.current.srcObject as MediaStream | null;
      let recorderStream: MediaStream | null = null;
      setRecordingStatus('Preparing recording...');

      if (original) {
        const microphoneTracks = getMicrophoneStream()
          ?.getAudioTracks()
          .filter((track) => track.readyState === 'live') ?? [];
        recorderStream = new MediaStream([...original.getVideoTracks(), ...microphoneTracks]);
        setRecordingStatus(microphoneTracks.length ? 'Recording video + audio' : 'Recording video only');
      } else {
        setRecordingStatus('Grant camera access first');
        return;
      }

      // Create MediaRecorder
      const mr = new MediaRecorder(recorderStream as MediaStream, { mimeType: 'video/webm' });
      recordedChunksRef.current = [];
      mr.ondataavailable = (ev:any)=>{ if(ev.data && ev.data.size>0) recordedChunksRef.current.push(ev.data); };
      mr.onstop = async ()=>{
        const blob = new Blob(recordedChunksRef.current, { type: 'video/webm' });
        const form = new FormData();
        form.append('file', blob, `recording-${Date.now()}.webm`);
        setRecordingStatus('Uploading recording...');
        try{
          const res = await fetch(`${API_URL}/api/interview/sessions/${sessionId}/recording`, { method: 'POST', body: form });
          const json = await parseApiResponse<any>(res);
          console.log('upload result', json);
          setRecordingStatus('Recording uploaded');
        }catch(err){
          console.error('Upload failed', err);
          setRecordingStatus('Recording upload failed');
        }
      };
      mediaRecorderRef.current = mr;
      mr.start();
      setIsRecording(true);
    }catch(err){
      console.error('startRecording error', err);
      setRecordingStatus(`Recording failed: ${err instanceof Error ? err.message : 'unknown error'}`);
    }
  }

  function stopRecording(){
    try{
      if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
        mediaRecorderRef.current.stop();
      }
      setIsRecording(false);
      setRecordingStatus('Stopping...');
    }catch(e){console.error(e);}
  }

  function startSpeechToText(){
    if (speechRecognitionRef.current) return;

    const Recognition = getSpeechRecognitionConstructor();
    if (!Recognition) {
      setSpeechSupported(false);
      setSpeechStatus('Browser live captions unavailable. Record spoken answers for transcription.');
      return;
    }

    const recognition = new Recognition();
    recognition.continuous = true;
    recognition.interimResults = true;
    recognition.lang = 'en-US';
    recognition.onstart = () => {
      setSpeechSupported(true);
      setIsListening(true);
      setSpeechStatus('Listening for spoken answers');
    };
    recognition.onresult = (event) => {
      let finalText = '';
      let interimText = '';

      for (let index = event.resultIndex; index < event.results.length; index += 1) {
        const transcript = event.results[index][0]?.transcript ?? '';
        if (event.results[index].isFinal) {
          finalText += transcript;
        } else {
          interimText += transcript;
        }
      }

      if (finalText.trim()) {
        setText((current) => `${current} ${finalText}`.trim());
      }
      
    };
    recognition.onerror = (event) => {
      setSpeechStatus(event.error ? `Speech-to-text error: ${event.error}` : 'Speech-to-text error');
      setIsListening(false);
    };
    recognition.onend = () => {
      setIsListening(false);
      speechRecognitionRef.current = null;
      setSpeechStatus('Speech-to-text idle');
    };

    speechRecognitionRef.current = recognition;
    recognition.start();
  }

  async function uploadAnswerAudio(blob: Blob) {
    const form = new FormData();
    form.append('file', blob, `answer-${Date.now()}.webm`);
    setIsTranscribingAnswer(true);
    setSpeechStatus('Transcribing spoken answer...');

    try {
      const res = await fetch(`${API_URL}/api/interview/sessions/${sessionId}/answer-transcription`, {
        method: 'POST',
        body: form,
      });
      const json = await parseApiResponse<any>(res);

      const transcript = String(json?.text || '').trim();
      if (transcript && transcript !== 'Transcript unavailable.') {
        setText((current) => `${current} ${transcript}`.trim());
        setSpeechStatus('Spoken answer transcribed');
      } else {
        setSpeechStatus('Transcription returned no usable text');
      }
    } catch (error) {
      setSpeechStatus(error instanceof Error ? error.message : 'Answer transcription failed');
    } finally {
      setIsTranscribingAnswer(false);
    }
  }

  function startAnswerRecording(){
    if (answerRecorderRef.current || isTranscribingAnswer) return;

    // The camera stream (videoRef) is captured with audio:false, so its audio tracks are empty.
    // Use the dedicated microphone stream from the transcription hook instead.
    const liveMicTracks = getMicrophoneStream()
      ?.getAudioTracks()
      .filter((track) => track.readyState === 'live') ?? [];
    if (!liveMicTracks.length) {
      setSpeechStatus('Microphone stream is not ready');
      return;
    }

    try {
      const audioStream = new MediaStream(liveMicTracks);
      const recorder = new MediaRecorder(audioStream, { mimeType: 'audio/webm' });
      answerChunksRef.current = [];
      recorder.ondataavailable = (event) => {
        if (event.data && event.data.size > 0) answerChunksRef.current.push(event.data);
      };
      recorder.onstop = () => {
        const blob = new Blob(answerChunksRef.current, { type: 'audio/webm' });
        answerChunksRef.current = [];
        answerRecorderRef.current = null;
        setIsListening(false);
        if (blob.size > 0) {
          void uploadAnswerAudio(blob);
        } else {
          setSpeechStatus('No spoken audio captured');
        }
      };
      answerRecorderRef.current = recorder;
      recorder.start();
      setIsListening(true);
      setSpeechStatus('Recording spoken answer');
    } catch (error) {
      answerRecorderRef.current = null;
      setIsListening(false);
      setSpeechStatus(error instanceof Error ? error.message : 'Could not start answer recording');
    }
  }

  function stopAnswerRecording(){
    const recorder = answerRecorderRef.current;
    if (!recorder || recorder.state === 'inactive') return;
    recorder.stop();
    setSpeechStatus('Preparing spoken answer for transcription...');
  }

  function stopSpeechToText(){
    speechRecognitionRef.current?.stop();
    speechRecognitionRef.current = null;
    setIsListening(false);
    setSpeechStatus(speechSupported ? 'Speech-to-text idle' : 'Browser live captions unavailable. Record spoken answers for transcription.');
  }

  function toggleSpeechToText(){
    if (isListening) {
      if (speechRecognitionRef.current) {
        stopSpeechToText();
      } else {
        stopAnswerRecording();
      }
      return;
    }

    if (speechSupported) {
      startSpeechToText();
    } else {
      startAnswerRecording();
    }
  }

  async function handleStartSession(){
    if (sessionStarted || isStartingSession) return;
    if (!micPermissionGranted) {
      setRecordingStatus('Enable microphone access before starting the session');
      return;
    }

    setIsStartingSession(true);
    try{
      setRecordingStatus('Starting session...');
      const started = startProctoringSession();
      if (!started) {
        setRecordingStatus('Session could not be started');
        return;
      }

      setRecordingStatus('Session started. Starting recording...');
      startTranscription();

      // Local monitoring must not depend on backend latency or availability.
      void fetch(`${API_URL}/api/interview/sessions/${sessionId}/start`, { method: 'POST' })
        .then((response) => {
          if (!response.ok) {
            console.error(`Backend session start failed (${response.status})`);
          }
        })
        .catch((error) => {
          console.error('Backend session start failed', error);
        });

      const res = await fetch(`${API_URL}/api/interview/sessions/${sessionId}/start`, { method: 'POST' });
      const json = await parseApiResponse<any>(res);
      if (json?.initialQuestion) {
        setMessages([{speaker:'ai', text: json.initialQuestion}]);
        markAiFinished();
      }
      // begin recording automatically
      await startRecording();
    }catch(err){
      console.error('handleStartSession failed', err);
      setRecordingStatus(`Session start error: ${err instanceof Error ? err.message : 'unknown error'}`);
    } finally {
      setIsStartingSession(false);
    }
  }

  async function completeSession(){
    if (!sessionStarted) return;

    try{
      setIsEvaluating(true);
      stopSpeechToText();
      stopAnswerRecording();
      stopRecording();
      await stopTranscription();
      releaseMicrophone();
      endProctoringSession();
      const completeResponse = await fetch(`${API_URL}/api/interview/sessions/${sessionId}/complete`, { method: 'POST' });
      if (!completeResponse.ok) throw new Error(`Session completion failed (${completeResponse.status})`);
      setRecordingStatus('Session completed');
      const evalRes = await fetch(`${API_URL}/api/interview/sessions/${sessionId}/evaluate`, { method: 'POST' });
      if (!evalRes.ok) throw new Error(`Session evaluation failed (${evalRes.status})`);
      const evalJson = await parseApiResponse<any>(evalRes);
      if (evalJson?.evaluation) setEvaluationReport(evalJson.evaluation);
      const sessionRes = await fetch(`${API_URL}/api/interview/sessions/${sessionId}`);
      if (sessionRes.ok) setSessionData(await parseApiResponse<any>(sessionRes));
      setRecordingStatus('Session completed and evaluated');
    }catch(err){ console.error('completeSession failed', err); }
    finally { setIsEvaluating(false); }
  }

  async function send(){
    const answerText = text.trim();
    if(!answerText || isSubmittingAnswer) return;

    const metrics=analyze(answerText);
    setIsSubmittingAnswer(true);
    setAnswerStatus('Processing typed answer...');

    try {
      const response = await fetch(`${API_URL}/api/interview/sessions/${sessionId}/answers`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: answerText, metrics }),
      });
      const json = await parseApiResponse<any>(response);

      setMessages((current) => [
        ...current,
        { speaker: 'candidate', text: answerText, metrics },
        ...(json?.ai?.text ? [{ speaker: 'ai', text: json.ai.text }] : []),
      ]);
      setText('');
      markAiFinished();
      setAnswerStatus('Typed answer processed and saved for evaluation.');
    } catch (error) {
      setAnswerStatus(error instanceof Error ? error.message : 'Could not process the typed answer');
    } finally {
      setIsSubmittingAnswer(false);
    }
  }

  useEffect(()=>{ setDuration(new Date().toLocaleTimeString([], {hour:'numeric', minute:'2-digit', second:'2-digit'})); },[]);

  const cameraReady = state.cameraActive && !state.permissionDenied;
  const screenShareReady = !state.screenShareSupported || state.screenShareReadyBeforeInterview;
  const fullscreenReady = !state.fullscreenSupported || state.fullscreenReadyBeforeInterview || state.fullscreenActive;
  const permissionsReadyForCalibration = cameraReady && screenShareReady && fullscreenReady && micPermissionGranted;

  async function requestAccessBeforeCalibration(){
    await requestRequiredPermissions();
  }

  const systemChecks = [
    { label: 'Camera stream', ok: state.cameraActive, detail: state.cameraActive ? 'Active' : 'Inactive' },
    { label: 'Typed answer input', ok: true, detail: 'Answers are submitted directly for evaluation' },
    { label: 'Face detector', ok: state.faceDetectorActive, detail: state.faceDetectorActive ? `Tracking ${state.faceCount} face${state.faceCount === 1 ? '' : 's'}` : 'Starting' },
    { label: 'Object detector', ok: state.objectDetectorActive, detail: state.phoneDetected ? 'Phone flagged' : 'Scanning for phone-like objects' },
    { label: 'Gaze monitor', ok: !state.gazeAwayDetected, detail: state.gazeAwayDetected ? `Looking ${state.gazeDirection}` : 'Centered on camera' },
    { label: 'WebSocket loop', ok: socket?.readyState === WebSocket.OPEN, detail: socket?.readyState === WebSocket.OPEN ? 'Connected' : 'Connecting' },
    { label: 'Backend logging', ok: events.length >= 0, detail: 'Proctoring events persist to the API' },
  ];
  const proctoringScoreItems: Array<[string, number]> = finalProctoringScore ? [
    ['Gaze', finalProctoringScore.gazeScore],
    ['Face presence', finalProctoringScore.faceScore],
    ['Tab discipline', finalProctoringScore.tabScore],
    ['Phone/object', finalProctoringScore.phoneScore],
    ['Head pose', finalProctoringScore.headPoseScore],
    ['Session control', finalProctoringScore.sessionControlScore],
  ] : [];

  useEffect(() => {
    if (!showProctoringScore) return;
    proctoringReportRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }, [showProctoringScore]);

  return (
    <main className="min-h-screen bg-ink p-5 text-white">
      {!calibration && !permissionsReadyForCalibration && (
        <div className="fixed inset-0 z-[9999] flex items-center justify-center bg-[#0a0f1a] px-6 text-slate-100">
          <div className="w-full max-w-xl">
            <div className="mb-8 text-center">
              <p className="text-xs uppercase tracking-[0.35em] text-cyan-300">Pre-interview access</p>
              <h1 className="mt-4 text-3xl font-black">Grant access before gaze calibration</h1>
              <p className="mt-3 text-sm leading-6 text-slate-400">
                Camera, microphone, screen sharing, and fullscreen are checked now. Typed answers do not require microphone access.
              </p>
            </div>

            <div className="space-y-3 rounded-2xl border border-white/10 bg-white/5 p-4">
              {[
                { label: 'Camera', ok: cameraReady, detail: state.permissionDenied ? 'Permission denied' : state.cameraActive ? 'Ready' : 'Waiting for browser permission', icon: Video },
                { label: 'Microphone', ok: micPermissionGranted, detail: micPermissionGranted ? 'Microphone enabled' : microphoneError || 'Required before the interview', icon: Mic },
                { label: 'Screen share', ok: screenShareReady, detail: !state.screenShareSupported ? 'Unavailable in this browser' : state.screenShareReadyBeforeInterview ? 'Ready' : 'Required before calibration', icon: MonitorUp },
                { label: 'Fullscreen', ok: fullscreenReady, detail: !state.fullscreenSupported ? 'Unavailable in this browser' : fullscreenReady ? 'Ready' : 'Required before calibration', icon: Maximize2 },
              ].map(({ label, ok, detail, icon: Icon }) => (
                <div key={label} className="flex items-center justify-between gap-4 rounded-xl bg-slate-950/70 px-4 py-3">
                  <div className="flex items-center gap-3">
                    <Icon size={18} className={ok ? 'text-emerald-300' : 'text-cyan-300'} />
                    <div>
                      <p className="text-sm font-semibold">{label}</p>
                      <p className="text-xs text-slate-400">{detail}</p>
                    </div>
                  </div>
                  <span className={`h-2.5 w-2.5 rounded-full ${ok ? 'bg-emerald-400' : 'bg-amber-300'}`} />
                </div>
              ))}
            </div>

            <button
              type="button"
              onClick={requestMicrophonePermission}
              disabled={isRequestingMicrophone || micPermissionGranted}
              className="mt-6 flex w-full items-center justify-center gap-2 rounded-xl bg-emerald-300 px-5 py-3 text-sm font-black text-slate-950 disabled:cursor-not-allowed disabled:opacity-70"
            >
              <Mic size={18} />
              {micPermissionGranted ? 'Microphone enabled' : isRequestingMicrophone ? 'Requesting microphone...' : 'Enable Microphone'}
            </button>
            <button
              onClick={requestAccessBeforeCalibration}
              className="mt-3 flex w-full items-center justify-center gap-2 rounded-xl bg-cyan-300 px-5 py-3 text-sm font-black text-slate-950 shadow-[0_0_40px_rgba(103,232,249,.18)]"
            >
              <ShieldCheck size={18} />
              Grant required access
            </button>
            {state.permissionDenied && (
              <p className="mt-4 text-center text-sm text-rose-200">
                Camera access was denied. Allow it in your browser settings, then refresh this page.
              </p>
            )}
            {microphoneError && (
              <p className="mt-4 text-center text-sm text-rose-200">{microphoneError}</p>
            )}
            {!speechRecognitionSupported && micPermissionGranted && (
              <p className="mt-4 text-center text-sm text-amber-200">
                Microphone recording is available, but live speech-to-text requires Chrome or Edge.
              </p>
            )}
          </div>
        </div>
      )}
      {!calibration && permissionsReadyForCalibration && (
        <GazeCalibration
          videoRef={videoRef}
          onComplete={setCalibration}
          onSkip={() => setCalibration({
            thresholdX: 0.18,
            thresholdY: 0.22,
            neutralX: 0,
            neutralY: 0,
            pointData: [],
            qualityScore: 0,
          })}
        />
      )}
      <div className="mx-auto grid max-w-7xl gap-5 lg:grid-cols-[1fr_420px]">
        <section className="rounded-[2rem] bg-slate-950 p-5 shadow-2xl">
          <div className="mb-4 flex items-center justify-between">
            <div>
              <p className="text-sm text-cyan-100">Candidate interview</p>
              <h1 className="text-2xl font-black">Junior SDE Screening</h1>
            </div>
            <div className="flex gap-2 text-xs">
              {!sessionStarted && (
                <button
                  type="button"
                  onClick={requestMicrophonePermission}
                  disabled={isRequestingMicrophone || micPermissionGranted}
                  className={`rounded-full px-3 py-2 text-xs font-semibold disabled:cursor-not-allowed disabled:opacity-70 ${
                    micPermissionGranted ? 'bg-emerald-500/20 text-emerald-100' : 'bg-cyan-300 text-slate-950'
                  }`}
                >
                  <Mic size={14} className="mr-1 inline" />
                  {micPermissionGranted ? 'Microphone enabled' : isRequestingMicrophone ? 'Requesting...' : 'Enable Microphone'}
                </button>
              )}
              <button
                type="button"
                onClick={handleStartSession}
                disabled={sessionStarted || isStartingSession || showProctoringScore || !calibration || !micPermissionGranted}
                className="rounded-full bg-emerald-500/20 px-3 py-2 text-xs font-semibold disabled:cursor-not-allowed disabled:opacity-60"
              >
                {sessionStarted ? 'Session started' : isStartingSession ? 'Starting...' : 'Start Session'}
              </button>
              <button onClick={completeSession} disabled={!sessionStarted || showProctoringScore || isEvaluating || isSubmittingAnswer} className="rounded-full bg-rose-500/10 px-3 py-2 text-xs font-semibold disabled:cursor-not-allowed disabled:opacity-60">{isEvaluating ? 'Evaluating...' : 'Complete session'}</button>
              <span className="rounded-full bg-white/10 px-3 py-2"><Timer size={14} className="mr-1 inline"/>{duration}</span>
              <span className={`rounded-full px-3 py-2 ${state.permissionDenied ? 'bg-rose-500/20 text-rose-100' : state.initialized ? 'bg-emerald-400/20 text-emerald-100' : 'bg-amber-400/20 text-amber-100'}`}><ShieldCheck size={14} className="mr-1 inline"/>{state.status}</span>
            </div>
          </div>
          <p className={`mb-4 text-xs font-semibold ${sessionStarted ? 'text-emerald-300' : 'text-slate-400'}`}>
            {sessionStarted
              ? `Proctoring is active. ${isTranscribing ? 'Live transcription is running.' : 'Live transcription is unavailable.'}`
              : micPermissionGranted
                ? 'Microphone enabled. Proctoring is paused until Start Session is clicked.'
                : 'Enable the microphone before starting the interview.'}
          </p>
          {(microphoneError || transcriptionError) && (
            <p className="mb-4 rounded-2xl bg-rose-500/10 px-4 py-3 text-sm text-rose-100">
              {microphoneError || transcriptionError}
            </p>
          )}

          <div className="relative aspect-video overflow-hidden rounded-[2rem] bg-gradient-to-br from-cyan-300 via-slate-800 to-slate-950">
            <div className="absolute inset-0 grid place-items-center">
              <div className="text-center">
                <div className="mx-auto mb-5 flex h-40 w-40 items-center justify-center rounded-full bg-white/20 shadow-[0_0_80px_rgba(103,232,249,.5)] ring-8 ring-white/10">
                  <img
                    src="/avatar-placeholder.svg"
                    alt="AI interviewer avatar"
                    className="h-32 w-32 rounded-full object-cover"
                  />
                </div>
                <h2 className="text-2xl font-bold">AI Interviewer</h2>
                <p className="text-cyan-100">Avatar bridge ready: UE5 / WebRTC / Convai lip-sync payloads</p>
              </div>
            </div>
            {videoElement}
          </div>

          <div className="mt-5 rounded-3xl bg-white p-4 text-ink">
            <div className="max-h-64 space-y-3 overflow-auto pr-2">
              {messages.map((m,i)=>(
                <div key={i} className={`rounded-2xl p-3 ${m.speaker==='ai'?'bg-slate-100':'bg-cyan-50'}`}>
                  <b className="text-xs uppercase text-slate-500">{m.speaker}</b>
                  <p className="text-sm leading-6">{m.text}</p>
                  {m.metrics&&<p className="mt-1 text-xs text-slate-500">WPM {m.metrics.wpm} • latency {m.metrics.latencyMs}ms</p>}
                </div>
              ))}
            </div>
            {(interviewTranscript.length > 0 || interimTranscript) && (
              <div className="mt-4 rounded-2xl border border-cyan-100 bg-cyan-50 p-3">
                <div className="flex items-center justify-between text-xs font-semibold uppercase text-cyan-800">
                  <span>Live transcript</span>
                  <span>{isTranscribing ? 'Listening' : 'Stopped'}</span>
                </div>
                <div className="mt-2 max-h-32 space-y-2 overflow-auto text-sm text-slate-700">
                  {interviewTranscript.slice(-5).map((entry) => (
                    <p key={`${entry.timestamp}-${entry.text}`}>
                      <span className="mr-2 text-xs text-slate-400">{new Date(entry.timestamp).toLocaleTimeString()}</span>
                      {entry.text}
                    </p>
                  ))}
                  {interimTranscript ? <p className="italic text-slate-400">{interimTranscript}</p> : null}
                </div>
              </div>
            )}
            <div className="mt-4 flex gap-3">
              <input
                value={text}
                onChange={e=>setText(e.target.value)}
                onKeyDown={e=>{ if(e.key==='Enter'){ e.preventDefault(); void send(); } }}
                disabled={isSubmittingAnswer}
                className="flex-1 rounded-2xl border px-4 py-3 outline-none focus:ring-2 focus:ring-brand"
                placeholder="Type your answer here..."
              />
              <button
                onClick={()=>void send()}
                disabled={!text.trim() || isSubmittingAnswer}
                className="rounded-2xl bg-ink px-5 text-white disabled:cursor-not-allowed disabled:opacity-50"
                title="Submit typed answer"
              >
                <Send size={18}/>
              </button>
            </div>
            <div className="mt-2 flex flex-wrap items-center justify-between gap-2 text-xs text-slate-500">
              <span>{answerStatus}</span>
            </div>
          </div>
        </section>

        <aside className="space-y-5">
          {evaluationReport && (
            <div className="rounded-[2rem] bg-white p-6 text-ink shadow-2xl">
              <h2 className="font-bold"><BarChart3 className="mr-2 inline text-brand"/>Final evaluation report</h2>
              <div className="mt-4 grid grid-cols-2 gap-3">
                <div className="rounded-2xl bg-slate-50 p-4">
                  <p className="text-xs uppercase text-slate-500">Overall</p>
                  <p className="text-3xl font-black">{evaluationReport.overallScore ?? '-'}<span className="text-base text-slate-400">/100</span></p>
                </div>
                <div className="rounded-2xl bg-slate-50 p-4">
                  <p className="text-xs uppercase text-slate-500">Recommendation</p>
                  <p className="mt-1 text-sm font-black capitalize">{String(evaluationReport.recommendation ?? '-').replaceAll('_', ' ')}</p>
                  <p className="text-xs text-slate-500">Evaluation confidence: {evaluationReport.recommendationConfidence ?? '-'}</p>
                </div>
              </div>
              <p className="mt-4 text-sm leading-6 text-slate-600">{evaluationReport.summary}</p>
              {evaluationReport.candidateConfidence && (
                <div className="mt-4 rounded-2xl bg-cyan-50 p-4">
                  <h3 className="text-sm font-bold">Expressed confidence</h3>
                  <p className="mt-1 text-2xl font-black">
                    {evaluationReport.candidateConfidence.score}<span className="text-sm text-slate-400">/100</span>
                  </p>
                  <p className="text-xs capitalize text-slate-500">
                    {evaluationReport.candidateConfidence.level} confidence, {evaluationReport.candidateConfidence.reliability} assessment reliability
                  </p>
                  <p className="mt-2 text-xs leading-5 text-slate-600">{evaluationReport.candidateConfidence.summary}</p>
                </div>
              )}

              <div className="mt-5">
                <h3 className="text-sm font-bold">Demonstrated strengths</h3>
                <ul className="mt-2 space-y-2 text-sm text-slate-600">
                  {(evaluationReport.strengths ?? []).map((item:string, index:number)=>(
                    <li key={index} className="rounded-xl bg-emerald-50 px-3 py-2">{item}</li>
                  ))}
                </ul>
              </div>

              <div className="mt-5">
                <h3 className="text-sm font-bold">Development areas</h3>
                <ul className="mt-2 space-y-2 text-sm text-slate-600">
                  {(evaluationReport.weaknesses ?? []).map((item:string, index:number)=>(
                    <li key={index} className="rounded-xl bg-amber-50 px-3 py-2">{item}</li>
                  ))}
                </ul>
              </div>

              <div className="mt-5">
                <h3 className="text-sm font-bold">Question breakdown</h3>
                <div className="mt-2 space-y-3">
                  {(evaluationReport.questionBreakdown ?? []).map((item:any, index:number)=>(
                    <details key={item.answerId ?? index} className="rounded-2xl bg-slate-50 p-3 text-sm">
                      <summary className="cursor-pointer font-semibold">
                        Question {index + 1}: {item.questionText ?? item.question ?? 'Asked question'} ({item.overallScore}/100)
                      </summary>
                      <p className="mt-2 text-slate-600">{item.summary}</p>
                      {item.transcriptConfidence && (
                        <p className="mt-2 text-xs text-slate-500">
                          Expressed confidence: {item.transcriptConfidence.confidenceScore}/100 ({item.transcriptConfidence.confidenceLevel}, {item.transcriptConfidence.reliability} reliability). Fillers: {item.transcriptConfidence.fillerCount}, uncertainty: {item.transcriptConfidence.strongUncertaintyCount}, hedges: {item.transcriptConfidence.hedgeCount}.
                        </p>
                      )}
                      {item.aiAuthorshipAssessment && (
                        <div className="mt-3 rounded-xl border border-violet-100 bg-violet-50 px-3 py-2">
                          <div className="flex flex-wrap items-center justify-between gap-2">
                            <span className="text-xs font-semibold text-violet-950">AI-authorship likelihood</span>
                            <span className="text-sm font-black text-violet-700">
                              {item.aiAuthorshipAssessment.probability}%
                            </span>
                          </div>
                          <details className="mt-1 text-xs text-violet-900">
                            <summary className="cursor-pointer font-medium">
                              Why this estimate?
                            </summary>
                            <ul className="mt-2 space-y-1">
                              {(item.aiAuthorshipAssessment.reasons ?? []).map((reason:string, reasonIndex:number)=>(
                                <li key={reasonIndex}>- {reason}</li>
                              ))}
                            </ul>
                            <p className="mt-2 text-[11px] leading-4 text-violet-700">
                              {item.aiAuthorshipAssessment.confidence} assessment confidence. {item.aiAuthorshipAssessment.disclaimer}
                            </p>
                          </details>
                        </div>
                      )}
                      <div className="mt-2 space-y-1 text-xs text-slate-500">
                        {(item.modelAnswerComparison?.requiredPointCoverage ?? []).map((point:any)=>(
                          <div key={point.pointId} className="flex justify-between gap-3">
                            <span>{point.description}</span>
                            <span className="text-right font-semibold">{point.status} - {point.score}</span>
                          </div>
                        ))}
                      </div>
                    </details>
                  ))}
                </div>
              </div>
            </div>
          )}

          <div className="rounded-[2rem] bg-white p-6 text-ink shadow-2xl">
            <h2 className="font-bold"><Video className="mr-2 inline text-brand"/>System check</h2>
            <ul className="mt-4 space-y-3 text-sm text-slate-600">
              {systemChecks.map((check)=>(
                <li key={check.label} className="flex items-start justify-between gap-3 rounded-2xl bg-slate-50 px-3 py-2">
                  <span><span className={`mr-2 inline-flex h-2.5 w-2.5 rounded-full ${check.ok ? 'bg-emerald-500' : 'bg-rose-500'}`} />{check.label}</span>
                  <span className="text-right text-xs text-slate-500">{check.detail}</span>
                </li>
              ))}
            </ul>
            <p className="mt-4 text-xs text-slate-500">Last observation: {state.lastObservationAt ? new Date(state.lastObservationAt).toLocaleTimeString() : 'waiting for camera input'}</p>
          </div>

          <div className="rounded-[2rem] bg-white p-6 text-ink shadow-2xl">
            <h2 className="font-bold"><Mic className="mr-2 inline text-brand"/>Live integrity events</h2>
            <div className="mt-4 space-y-3">
              {events.length?events.map((e,i)=>(
                <div key={i} className="rounded-2xl bg-slate-50 p-3 text-sm">
                  <b>{e.severity}</b>
                  <p>{e.eventType}</p>
                  <pre className="mt-1 whitespace-pre-wrap text-xs text-slate-500">{e.metadata ? JSON.stringify(e.metadata, null, 2) : ''}</pre>
                </div>
              )):<p className="text-sm text-slate-500">No events flagged yet.</p>}
            </div>
          </div>

          {showProctoringScore && finalProctoringScore ? (
            <div ref={proctoringReportRef} className="rounded-[2rem] border border-cyan-200 bg-white p-6 text-ink shadow-2xl">
              <div className="flex items-start justify-between gap-4">
                <div>
                  <p className="text-xs font-bold uppercase tracking-[0.25em] text-cyan-600">Final report</p>
                  <h2 className="mt-2 text-xl font-black">Proctoring report</h2>
                  <p className="mt-1 text-sm text-slate-500">Session ended. Monitoring is stopped and the final integrity score is ready.</p>
                </div>
                <div className="rounded-3xl bg-ink px-5 py-4 text-center text-white">
                  <p className="text-xs text-cyan-100">Score</p>
                  <p className="text-3xl font-black">{finalProctoringScore.final}</p>
                  <p className="text-xs text-cyan-100">/100</p>
                </div>
              </div>

              <div className="mt-5 rounded-2xl bg-cyan-50 p-4">
                <p className="text-sm font-bold text-slate-700">Integrity band</p>
                <p className="mt-1 text-2xl font-black text-ink">{finalProctoringScore.band}</p>
              </div>

              <div className="mt-5 space-y-3">
                {proctoringScoreItems.map(([label, value]) => (
                  <div key={label} className="rounded-2xl bg-slate-50 p-3">
                    <div className="flex items-center justify-between text-sm">
                      <span className="font-semibold text-slate-700">{label}</span>
                      <span className="font-bold text-ink">{value}/100</span>
                    </div>
                    <div className="mt-2 h-2 rounded-full bg-slate-200">
                      <div className="h-2 rounded-full bg-cyan-400" style={{ width: `${value}%` }} />
                    </div>
                  </div>
                ))}
              </div>

              <div className="mt-5">
                <h3 className="font-bold">Flags</h3>
                {finalProctoringScore.flags.length ? (
                  <ul className="mt-3 space-y-2 text-sm text-slate-600">
                    {finalProctoringScore.flags.map((flag) => (
                      <li key={flag} className="rounded-2xl bg-rose-50 px-3 py-2 text-rose-900">{flag}</li>
                    ))}
                  </ul>
                ) : (
                  <p className="mt-3 rounded-2xl bg-emerald-50 px-3 py-2 text-sm text-emerald-900">No major proctoring flags were detected.</p>
                )}
              </div>

              <div className="mt-5 grid grid-cols-2 gap-3 text-sm">
                <div className="rounded-2xl bg-slate-50 p-3">
                  <p className="text-slate-500">Integrity events</p>
                  <p className="text-2xl font-black">{events.length}</p>
                </div>
                <div className="rounded-2xl bg-slate-50 p-3">
                  <p className="text-slate-500">Completed at</p>
                  <p className="mt-1 font-semibold">{state.lastObservationAt ? new Date(state.lastObservationAt).toLocaleTimeString() : 'Just now'}</p>
                </div>
              </div>
            </div>
          ) : null}

          <div className="rounded-[2rem] bg-white p-6 text-ink shadow-2xl">
            <h2 className="font-bold">Recordings & Transcripts</h2>
            <p className="mt-2 text-sm text-slate-500">Recorded candidate responses and automated transcriptions / question-fit scoring.</p>
            <p className="mt-2 text-xs text-slate-500">Recording status: {recordingStatus}</p>
            <div className="mt-4 space-y-3">
              {sessionData?.transcript?.length ? sessionData.transcript.slice().reverse().map((entry:any, idx:number)=>(
                <div key={idx} className="rounded-2xl border bg-slate-50 p-3">
                  <div className="text-xs text-slate-500">{entry.type} • {entry.createdAt ? new Date(entry.createdAt).toLocaleString() : ''}</div>
                  {entry.type === 'recording' ? (
                    <video className="mt-2 w-full" controls src={`${API_URL}${entry.url}`} />
                  ) : null}
                  {(entry.speaker === 'candidate' || entry.speaker === 'ai') && entry.text ? (
                    <p className="mt-2 text-sm">
                      <span className="mr-2 font-semibold capitalize">{entry.speaker}:</span>
                      {entry.text}
                    </p>
                  ) : null}
                  {entry.type === 'transcription' || entry.type === 'speech_to_text_transcript' ? (
                    <pre className="mt-2 text-sm whitespace-pre-wrap">{entry.text}</pre>
                  ) : null}
                </div>
              )) : <div className="text-sm text-slate-500">No recordings yet.</div>}

              {sessionData?.evaluation?.partialQuestionFit?.length ? (
                <div className="mt-3">
                  <h4 className="font-semibold">Question-fit</h4>
                  <ul className="mt-2 space-y-2">
                    {sessionData.evaluation.partialQuestionFit.map((q:any, i:number)=>(
                      <li key={i} className="rounded-2xl bg-white p-3 text-sm">
                        <div className="font-semibold">Score: {q.score}/5</div>
                        <div className="text-xs text-slate-500">{q.reasoning}</div>
                      </li>
                    ))}
                  </ul>
                </div>
              ): null}
            </div>
          </div>

          <div className="rounded-[2rem] bg-cyan-50 p-6 text-ink">
            <h2 className="font-bold">Session ID</h2>
            <input value={sessionId} onChange={e=>setSessionId(e.target.value)} className="mt-3 w-full rounded-2xl border bg-white px-4 py-3 text-sm"/>
          </div>
        </aside>
      </div>
    </main>
  );
}
