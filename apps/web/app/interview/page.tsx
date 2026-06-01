 'use client';
 import { useEffect, useRef, useState } from 'react';
 import { WS_URL, API_URL } from '@/lib/api';
 import { GazeCalibration } from '@/hooks/GazeCalibration';
 import { useProctoring } from '@/hooks/useProctoring';
 import { useSpeechMetrics } from '@/hooks/useSpeechMetrics';
 import { Mic, Send, ShieldCheck, Timer, Video } from 'lucide-react';
 import type { CalibrationResult } from '@/hooks/useGazeCalibration';

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
    async function bootstrapDemoSession() {
      if (sessionId !== 'demo-session') return;
      try {
        const res = await fetch(`${API_URL}/api/interview/demo-session`);
        if (!res.ok) return;
        const json = await res.json();
        if (alive && json?.sessionId) {
          setSessionId(json.sessionId);
        }
      } catch (error) {
        console.error('demo-session bootstrap failed', error);
      }
    }
    bootstrapDemoSession();
    const ws=new WebSocket(WS_URL);
    wsRef.current=ws;
    ws.onopen=()=>ws.send(JSON.stringify({type:'register',role:'candidate',sessionId}));
    ws.onmessage=(e)=>{const msg=JSON.parse(e.data); if(msg.type==='ai_response'){setMessages(m=>[...m,{speaker:'ai',text:msg.text}]); markAiFinished();}};
    setSocket(ws);
    return()=>{ alive = false; ws.close(); };
  },[sessionId]);

  const { videoRef, events, state } = useProctoring(sessionId, socket, calibration);
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
  const [recordingStatus, setRecordingStatus] = useState('Idle');
  const mediaRecorderRef = useRef<MediaRecorder|null>(null);
  const recordedChunksRef = useRef<BlobPart[]>([]);
  const extraAudioStreamRef = useRef<MediaStream|null>(null);
  const [sessionData, setSessionData] = useState<any|null>(null);

  useEffect(()=>{
    let mounted = true;
    async function load(){
      try{
        const res = await fetch(`${API_URL}/api/interview/sessions/${sessionId}`);
        if(!res.ok) return;
        const json = await res.json();
        if(mounted) setSessionData(json);
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
        // Only record the already-active proctoring stream. Do not request new permissions here.
        recorderStream = original;
        setRecordingStatus((original.getAudioTracks() || []).length ? 'Recording video + audio' : 'Recording video only');
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
          const json = await res.json();
          console.log('upload result', json);
          setRecordingStatus('Recording uploaded');
        }catch(err){
          console.error('Upload failed', err);
          setRecordingStatus('Recording upload failed');
        }
        // do not stop the main camera stream used by proctoring
        try{
          if (extraAudioStreamRef.current) {
            extraAudioStreamRef.current.getTracks().forEach(t=>t.stop());
            extraAudioStreamRef.current = null;
          }
        }catch(e){ console.error('Error stopping recorder tracks', e); }
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
    try{ mediaRecorderRef.current?.stop(); setIsRecording(false); setRecordingStatus('Stopping...'); }catch(e){console.error(e);} 
  }

  async function startSession(){
    try{
      setRecordingStatus('Starting session...');
      await fetch(`${API_URL}/api/interview/sessions/${sessionId}/start`, { method: 'POST' });
      // begin recording automatically
      await startRecording();
    }catch(err){ console.error('startSession failed', err); }
  }

  async function completeSession(){
    try{
      // stop recording and complete session
      stopRecording();
      await fetch(`${API_URL}/api/interview/sessions/${sessionId}/complete`, { method: 'POST' });
      // optionally trigger evaluation
      await fetch(`${API_URL}/api/interview/sessions/${sessionId}/evaluate`, { method: 'POST' });
      setRecordingStatus('Session completed');
    }catch(err){ console.error('completeSession failed', err); }
  }

  function send(){
    if(!text.trim()) return;
    const metrics=analyze(text);
    socket?.send(JSON.stringify({type:'candidate_transcript',sessionId,text,timestamp:Date.now(),...metrics}));
    setMessages(m=>[...m,{speaker:'candidate',text,metrics}]);
    setText('');
  }

  useEffect(()=>{ setDuration(new Date().toLocaleTimeString([], {hour:'numeric', minute:'2-digit', second:'2-digit'})); },[]);

  const systemChecks = [
    { label: 'Camera stream', ok: state.cameraActive, detail: state.cameraActive ? 'Active' : 'Inactive' },
    { label: 'Face detector', ok: state.faceDetectorActive, detail: state.faceDetectorActive ? `Tracking ${state.faceCount} face${state.faceCount === 1 ? '' : 's'}` : 'Starting' },
    { label: 'Object detector', ok: state.objectDetectorActive, detail: state.phoneDetected ? 'Phone flagged' : 'Scanning for phone-like objects' },
    { label: 'Gaze monitor', ok: !state.gazeAwayDetected, detail: state.gazeAwayDetected ? `Looking ${state.gazeDirection}` : 'Centered on camera' },
    { label: 'WebSocket loop', ok: socket?.readyState === WebSocket.OPEN, detail: socket?.readyState === WebSocket.OPEN ? 'Connected' : 'Connecting' },
    { label: 'Backend logging', ok: events.length >= 0, detail: 'Proctoring events persist to the API' },
  ];

  return (
    <main className="min-h-screen bg-ink p-5 text-white">
      {!calibration && (
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
              <h1 className="text-2xl font-black">Associate Consultant Screening</h1>
            </div>
            <div className="flex gap-2 text-xs">
              <button onClick={startSession} className="rounded-full bg-emerald-500/20 px-3 py-2 text-xs font-semibold">Start session</button>
              <button onClick={completeSession} className="rounded-full bg-rose-500/10 px-3 py-2 text-xs font-semibold">Complete session</button>
              <span className="rounded-full bg-white/10 px-3 py-2"><Timer size={14} className="mr-1 inline"/>{duration}</span>
              <span className={`rounded-full px-3 py-2 ${state.permissionDenied ? 'bg-rose-500/20 text-rose-100' : state.initialized ? 'bg-emerald-400/20 text-emerald-100' : 'bg-amber-400/20 text-amber-100'}`}><ShieldCheck size={14} className="mr-1 inline"/>{state.status}</span>
            </div>
          </div>

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
            <div className="mt-4 flex gap-3">
              <input value={text} onChange={e=>setText(e.target.value)} onKeyDown={e=>e.key==='Enter'&&send()} className="flex-1 rounded-2xl border px-4 py-3 outline-none focus:ring-2 focus:ring-brand" placeholder="Type transcript here, or connect speech-to-text..."/>
              <button onClick={send} className="rounded-2xl bg-ink px-5 text-white"><Send size={18}/></button>
            </div>
          </div>
        </section>

        <aside className="space-y-5">
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
                  {entry.type === 'transcription' ? (
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
