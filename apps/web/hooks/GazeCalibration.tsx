'use client';

import { useEffect, useRef, useState } from 'react';
import { CALIBRATION_POINTS, useGazeCalibration, type CalibrationResult } from './useGazeCalibration';

type Props = {
  videoRef: React.RefObject<HTMLVideoElement>;
  onComplete: (result: CalibrationResult) => void;
  onSkip?: () => void;
};

// ── tiny helpers ────────────────────────────────────────────────────────────

function QualityBar({ score }: { score: number }) {
  const pct = Math.round(score * 100);
  const colour =
    pct >= 70 ? '#4ade80' :
    pct >= 40 ? '#facc15' : '#f87171';
  const label =
    pct >= 70 ? 'Good' :
    pct >= 40 ? 'Fair — consider recalibrating' : 'Poor — please recalibrate';

  return (
    <div style={{ width: '100%', maxWidth: 340 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 6 }}>
        <span style={{ fontSize: 13, color: '#94a3b8', fontFamily: 'monospace' }}>Calibration quality</span>
        <span style={{ fontSize: 13, color: colour, fontFamily: 'monospace', fontWeight: 700 }}>{label}</span>
      </div>
      <div style={{ height: 6, background: '#1e293b', borderRadius: 99, overflow: 'hidden' }}>
        <div
          style={{
            height: '100%',
            width: `${pct}%`,
            background: colour,
            borderRadius: 99,
            transition: 'width 0.8s cubic-bezier(0.4,0,0.2,1)',
          }}
        />
      </div>
    </div>
  );
}

function Dot({
  x, y, active, done, waiting,
}: {
  x: number; y: number; active: boolean; done: boolean; waiting: boolean;
}) {
  const size = active ? 28 : 16;
  return (
    <div
      style={{
        position: 'absolute',
        left: `${x * 100}%`,
        top: `${y * 100}%`,
        transform: 'translate(-50%, -50%)',
        width: size,
        height: size,
        borderRadius: '50%',
        background: done
          ? '#4ade80'
          : active
          ? '#38bdf8'
          : '#334155',
        border: active ? '2px solid #7dd3fc' : '2px solid #475569',
        boxShadow: active
          ? '0 0 0 6px rgba(56,189,248,0.18), 0 0 20px rgba(56,189,248,0.35)'
          : 'none',
        transition: 'all 0.25s ease',
        zIndex: active ? 10 : 1,
      }}
    >
      {waiting && active && (
        <div
          style={{
            position: 'absolute',
            inset: -10,
            borderRadius: '50%',
            border: '2px solid rgba(56,189,248,0.4)',
            animation: 'ping 1s ease-out infinite',
          }}
        />
      )}
    </div>
  );
}

function SamplingRing({ progress }: { progress: number }) {
  const r = 22;
  const circ = 2 * Math.PI * r;
  return (
    <svg
      width={60}
      height={60}
      style={{ position: 'absolute', inset: 0, margin: 'auto', top: 0, left: 0, right: 0, bottom: 0 }}
    >
      <circle cx={30} cy={30} r={r} fill="none" stroke="#1e293b" strokeWidth={4} />
      <circle
        cx={30}
        cy={30}
        r={r}
        fill="none"
        stroke="#38bdf8"
        strokeWidth={4}
        strokeDasharray={circ}
        strokeDashoffset={circ * (1 - progress)}
        strokeLinecap="round"
        transform="rotate(-90 30 30)"
        style={{ transition: 'stroke-dashoffset 0.1s linear' }}
      />
    </svg>
  );
}

// ── main component ───────────────────────────────────────────────────────────

export function GazeCalibration({ videoRef, onComplete, onSkip }: Props) {
  const { calState, startCalibration, beginPoints, abort } = useGazeCalibration(videoRef);
  const hasStarted = useRef(false);

  // Kick off MediaPipe load as soon as the component mounts
  useEffect(() => {
    if (!hasStarted.current) {
      hasStarted.current = true;
      startCalibration();
    }
  }, [startCalibration]);

  // When calibration finishes, surface result to parent
  useEffect(() => {
    if (calState.phase === 'done' && calState.result) {
      // Give user 1.5 s to see the quality screen before closing
      const t = setTimeout(() => onComplete(calState.result!), 2200);
      return () => clearTimeout(t);
    }
  }, [calState.phase, calState.result, onComplete]);

  const currentPoint =
    calState.currentPointIndex >= 0
      ? CALIBRATION_POINTS[calState.currentPointIndex]
      : null;

  const samplingProgress =
    calState.phase === 'sampling'
      ? calState.samplesCollected / 20
      : 0;

  return (
    <>
      <style>{`
        @keyframes ping {
          0%   { transform: scale(1); opacity: 0.8; }
          100% { transform: scale(2.2); opacity: 0; }
        }
        @keyframes fadeIn {
          from { opacity: 0; transform: translateY(12px); }
          to   { opacity: 1; transform: translateY(0); }
        }
        @keyframes shimmer {
          0%   { background-position: -400px 0; }
          100% { background-position: 400px 0; }
        }
      `}</style>

      <div
        style={{
          position: 'fixed',
          inset: 0,
          background: '#0a0f1a',
          zIndex: 9999,
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          fontFamily: "'DM Mono', 'Fira Code', 'Courier New', monospace",
          color: '#e2e8f0',
        }}
      >

        {/* ── INTRO ─────────────────────────────────────────────────── */}
        {calState.phase === 'intro' && (
          <div
            style={{
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              gap: 28,
              maxWidth: 520,
              padding: '0 24px',
              animation: 'fadeIn 0.4s ease',
            }}
          >
            <div style={{ fontSize: 11, letterSpacing: 4, color: '#475569', textTransform: 'uppercase' }}>
              Eye Tracking Setup
            </div>
            <h1 style={{ fontSize: 26, fontWeight: 700, margin: 0, textAlign: 'center', lineHeight: 1.3, color: '#f1f5f9' }}>
              Calibrate your gaze
            </h1>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 12, width: '100%' }}>
              {[
                ['01', 'Nine dots will appear on screen one at a time.'],
                ['02', 'Look directly at each dot — keep your head still, move only your eyes.'],
                ['03', 'Wait for the dot to turn green before the next one appears.'],
                ['04', 'The whole process takes about 30 seconds.'],
              ].map(([n, text]) => (
                <div key={n} style={{ display: 'flex', gap: 14, alignItems: 'flex-start' }}>
                  <span style={{ fontSize: 11, color: '#38bdf8', minWidth: 22, paddingTop: 2 }}>{n}</span>
                  <span style={{ fontSize: 14, color: '#94a3b8', lineHeight: 1.5 }}>{text}</span>
                </div>
              ))}
            </div>
            <div style={{ display: 'flex', gap: 12, marginTop: 8 }}>
              <button
                onClick={beginPoints}
                style={{
                  padding: '12px 32px',
                  background: '#38bdf8',
                  color: '#0a0f1a',
                  border: 'none',
                  borderRadius: 6,
                  fontSize: 14,
                  fontWeight: 700,
                  cursor: 'pointer',
                  fontFamily: 'inherit',
                  letterSpacing: 1,
                }}
              >
                Begin calibration →
              </button>
              {onSkip && (
                <button
                  onClick={() => { abort(); onSkip(); }}
                  style={{
                    padding: '12px 20px',
                    background: 'transparent',
                    color: '#475569',
                    border: '1px solid #1e293b',
                    borderRadius: 6,
                    fontSize: 13,
                    cursor: 'pointer',
                    fontFamily: 'inherit',
                  }}
                >
                  Skip
                </button>
              )}
            </div>
          </div>
        )}

        {/* ── DOT GRID (waiting / sampling / between) ───────────────── */}
        {(calState.phase === 'waiting' || calState.phase === 'sampling' || calState.phase === 'between') && (
          <>
            {/* instruction strip */}
            <div
              style={{
                position: 'absolute',
                top: 24,
                left: '50%',
                transform: 'translateX(-50%)',
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                gap: 6,
              }}
            >
              <div style={{ fontSize: 11, letterSpacing: 3, color: '#475569', textTransform: 'uppercase' }}>
                Eye Tracking Calibration
              </div>
              <div style={{ fontSize: 14, color: '#64748b' }}>
                {calState.phase === 'waiting' && 'Look at the blue dot'}
                {calState.phase === 'sampling' && 'Hold your gaze…'}
                {calState.phase === 'between' && 'Good — next point coming'}
              </div>
              <div style={{ fontSize: 12, color: '#334155' }}>
                Point {(calState.currentPointIndex ?? 0) + 1} of {CALIBRATION_POINTS.length}
              </div>
            </div>

            {/* dot grid */}
            <div style={{ position: 'absolute', inset: '80px 40px 60px 40px' }}>
              {CALIBRATION_POINTS.map((pt, idx) => (
                <Dot
                  key={pt.id}
                  x={pt.xFrac}
                  y={pt.yFrac}
                  active={idx === calState.currentPointIndex}
                  done={idx < (calState.currentPointIndex ?? 0)}
                  waiting={calState.phase === 'waiting' && idx === calState.currentPointIndex}
                />
              ))}

              {/* sampling progress ring on active dot */}
              {calState.phase === 'sampling' && currentPoint && (
                <div
                  style={{
                    position: 'absolute',
                    left: `${currentPoint.xFrac * 100}%`,
                    top: `${currentPoint.yFrac * 100}%`,
                    transform: 'translate(-50%, -50%)',
                    width: 60,
                    height: 60,
                  }}
                >
                  <SamplingRing progress={samplingProgress} />
                </div>
              )}
            </div>
          </>
        )}

        {/* ── DONE ──────────────────────────────────────────────────── */}
        {calState.phase === 'done' && calState.result && (
          <div
            style={{
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              gap: 24,
              animation: 'fadeIn 0.5s ease',
            }}
          >
            <div
              style={{
                width: 56,
                height: 56,
                borderRadius: '50%',
                background: '#052e16',
                border: '2px solid #4ade80',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                fontSize: 24,
              }}
            >
              ✓
            </div>
            <h2 style={{ margin: 0, fontSize: 20, color: '#f1f5f9' }}>Calibration complete</h2>
            <QualityBar score={calState.result.qualityScore} />
            <div
              style={{
                display: 'grid',
                gridTemplateColumns: '1fr 1fr',
                gap: '8px 24px',
                fontSize: 12,
                color: '#475569',
              }}
            >
              <span>Threshold X</span>
              <span style={{ color: '#94a3b8' }}>{calState.result.thresholdX.toFixed(3)}</span>
              <span>Threshold Y</span>
              <span style={{ color: '#94a3b8' }}>{calState.result.thresholdY.toFixed(3)}</span>
              <span>Neutral X</span>
              <span style={{ color: '#94a3b8' }}>{calState.result.neutralX.toFixed(3)}</span>
              <span>Neutral Y</span>
              <span style={{ color: '#94a3b8' }}>{calState.result.neutralY.toFixed(3)}</span>
            </div>
            <div style={{ fontSize: 12, color: '#334155' }}>Applying settings…</div>
          </div>
        )}

        {/* ── ERROR ─────────────────────────────────────────────────── */}
        {calState.phase === 'error' && (
          <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 16 }}>
            <div style={{ fontSize: 32 }}>⚠</div>
            <div style={{ color: '#f87171', fontSize: 15 }}>{calState.error}</div>
            <button
              onClick={startCalibration}
              style={{
                padding: '10px 24px',
                background: '#1e293b',
                color: '#e2e8f0',
                border: '1px solid #334155',
                borderRadius: 6,
                cursor: 'pointer',
                fontFamily: 'inherit',
                fontSize: 13,
              }}
            >
              Retry
            </button>
          </div>
        )}
      </div>
    </>
  );
}
