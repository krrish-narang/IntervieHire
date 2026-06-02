'use client';

import { useCallback, useEffect, useRef, useState } from 'react';
import type { MutableRefObject } from 'react';
import {
  FaceLandmarker,
  FilesetResolver,
  ObjectDetector,
  type FaceLandmarkerResult,
  type ObjectDetectorResult,
} from '@mediapipe/tasks-vision';
import type { ProctoringPayload, Severity } from '@interviehire/shared';
import type { CalibrationResult } from './useGazeCalibration';
import { buildSafeEightDotCalibration, type SafeEightDotCalibration } from './eightDotCalibrationGuardV3';
import { FALLBACK_GAZE_THRESHOLD_X, FALLBACK_GAZE_THRESHOLD_Y } from './proctoringGazeThresholdsV3';

type ProctoringEvent = {
  eventType: string;
  severity: Severity;
  timestamp: number;
  metadata?: Record<string, unknown>;
};


type ProctoringViolationType =
  | 'TAB_SWITCH'
  | 'FULLSCREEN_EXIT'
  | 'CAMERA_OFF'
  | 'MULTIPLE_FACES'
  | 'NO_FACE'
  | 'MOBILE_PHONE'
  | 'GAZE_AWAY'
  | 'HEAD_MOVEMENT'
  | 'SCREEN_SHARE_STOPPED'
  | 'UNKNOWN';

type ViolationRecordingEvent = {
  type: ProctoringViolationType;
  at: string;
  details?: Record<string, unknown>;
};

export type ViolationRecordingClip = {
  id: string;
  startedAt: string;
  endedAt: string;
  durationMs: number;
  violationTypes: ProctoringViolationType[];
  events: ViolationRecordingEvent[];
  blob: Blob;
  url: string;
  mimeType: string;
};

type ActiveViolationRecordingMeta = {
  id: string;
  startedAtISO: string;
  startedAtMs: number;
  violationTypes: Set<ProctoringViolationType>;
  events: ViolationRecordingEvent[];
};

type UseViolationScreenRecorderOptions = {
  defaultClipMs?: number;
  minClipMs?: number;
  maxClipMs?: number;
  timesliceMs?: number;
  onClipReady?: (clip: ViolationRecordingClip) => void | Promise<void>;
  onError?: (error: Error) => void;
  onScreenShareStopped?: () => void;
};

type DetectionState = {
  initialized: boolean;
  status: string;
  permissionDenied: boolean;
  cameraActive: boolean;
  faceDetectorActive: boolean;
  objectDetectorActive: boolean;
  faceCount: number;
  phoneDetected: boolean;
  gazeAwayDetected: boolean;
  gazeDirection: string;
  headMovementDetected: boolean;
  headPoseDeviationDegrees: number;
  tabSwitchDetected: boolean;
  tabSwitchReason: string | null;
  lastTabSwitchAt: number | null;
  lastTabSwitchDurationMs: number | null;
  activeTabSwitchDurationMs: number;
  totalTabSwitchDurationMs: number;
  tabSwitchCount: number;
  fullscreenActive: boolean;
  fullscreenExitDetected: boolean;
  fullscreenExitReason: string | null;
  lastFullscreenExitAt: number | null;
  fullscreenSupported: boolean;
  fullscreenReadyBeforeInterview: boolean;
  fullscreenPromptRequired: boolean;
  preInterviewFullscreenRequestedAt: number | null;
  preInterviewFullscreenEnteredAt: number | null;
  screenShareSupported: boolean;
  screenShareReadyBeforeInterview: boolean;
  screenSharePromptRequired: boolean;
  preInterviewScreenShareRequestedAt: number | null;
  preInterviewScreenShareGrantedAt: number | null;
  lastObservationAt: number | null;
};

const FACE_MODEL_URL = 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task';
const OBJECT_MODEL_URL = 'https://storage.googleapis.com/mediapipe-tasks/object_detector/efficientdet_lite0_uint8.tflite';
const ALERT_COOLDOWN_MS = 15000;
const NO_FACE_THRESHOLD_MS = 4000;
const LIVE_INTERVAL_MS = 500; // check twice as often for lower lag
const MULTI_FACE_CONFIRM_MS = 1500;
const PHONE_CONFIRM_MS = 1000;
const GAZE_CONFIRM_MS = 1000; // faster gaze confirmation
const HEAD_POSE_CONFIRM_MS = 1200;
const CAMERA_OFF_CONFIRM_MS = 1200;
const TAB_SWITCH_ALERT_COOLDOWN_MS = 500;
const FULLSCREEN_ALERT_COOLDOWN_MS = 1000;
const HEAD_POSE_ROTATION_THRESHOLD_DEG = 23;
const HEAD_POSE_AXIS_THRESHOLD_DEG = 23;
// Small head tilts naturally move the eyes in the opposite direction relative to the face.
// These constants compensate that face-relative gaze shift before deciding whether gaze is away.
const HEAD_POSE_GAZE_COMPENSATION_START_DEG = 2.5;
const HEAD_POSE_GAZE_COMPENSATION_MAX_DEG = 16;
const HEAD_YAW_TO_GAZE_X_FACTOR = 0.01;
const HEAD_PITCH_TO_GAZE_Y_FACTOR = 0.01;
const GAZE_BLENDSHAPE_THRESHOLD = 0.8; // more sensitive blendshape threshold
// Blendshapes are used only to correct vertical direction when geometry already says gaze is away.
// Looking down can make the iris partially occluded by the eyelid, which sometimes makes raw iris geometry look like "up".
const VERTICAL_BLENDSHAPE_DIRECTION_THRESHOLD = 0.33;
const VERTICAL_BLENDSHAPE_MARGIN = 0.08;
// Downward gaze is usually weaker in iris geometry because the eyelids partially cover the iris.
// Keep normal up/left/right sensitivity unchanged, but make positive-Y/downward movement easier to trigger.
const DOWNWARD_GAZE_THRESHOLD_FACTOR = 1.6;
const MIN_DOWNWARD_GAZE_THRESHOLD = 0.065;
const DOWNWARD_BLENDSHAPE_AWAY_THRESHOLD = 0.4;
const DOWNWARD_BLENDSHAPE_MARGIN = 0.1;
const DOWNWARD_GEOMETRY_SUPPORT_FACTOR = 0.4;
const MIN_DOWNWARD_GEOMETRY_SUPPORT = 0.03;
// Fallback geometry thresholds used when no calibration has been run
const DEFAULT_GAZE_THRESHOLD_X = FALLBACK_GAZE_THRESHOLD_X;
const DEFAULT_GAZE_THRESHOLD_Y = FALLBACK_GAZE_THRESHOLD_Y;
const DEFAULT_VIOLATION_SCREEN_CLIP_MS = 8000;
const MIN_VIOLATION_SCREEN_CLIP_MS = 3000;
const MAX_VIOLATION_SCREEN_CLIP_MS = 30000;
const VIOLATION_SCREEN_RECORDING_TIMESLICE_MS = 1000;



function makeViolationRecordingId(prefix = 'violation-recording') {
  const cryptoObj = globalThis.crypto;
  if (cryptoObj && 'randomUUID' in cryptoObj) {
    return `${prefix}-${cryptoObj.randomUUID()}`;
  }

  return `${prefix}-${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

function getBestViolationRecordingMimeType() {
  if (typeof MediaRecorder === 'undefined' || !MediaRecorder.isTypeSupported) {
    return '';
  }

  const candidates = [
    'video/webm;codecs=vp9,opus',
    'video/webm;codecs=vp8,opus',
    'video/webm',
    'video/mp4',
  ];

  return candidates.find((type) => MediaRecorder.isTypeSupported(type)) ?? '';
}

function isScreenShareStreamActive(stream: MediaStream | null) {
  return Boolean(stream && stream.getVideoTracks().some((track) => track.readyState === 'live'));
}

function isScreenShareRecordingSupported() {
  if (typeof navigator === 'undefined') return true;
  return Boolean(navigator.mediaDevices?.getDisplayMedia && typeof MediaRecorder !== 'undefined');
}

function useViolationScreenRecorder(options: UseViolationScreenRecorderOptions = {}) {
  const {
    defaultClipMs = DEFAULT_VIOLATION_SCREEN_CLIP_MS,
    minClipMs = MIN_VIOLATION_SCREEN_CLIP_MS,
    maxClipMs = MAX_VIOLATION_SCREEN_CLIP_MS,
    timesliceMs = VIOLATION_SCREEN_RECORDING_TIMESLICE_MS,
    onClipReady,
    onError,
    onScreenShareStopped,
  } = options;

  const [hasScreenSharePermission, setHasScreenSharePermission] = useState(false);
  const [isRecordingViolation, setIsRecordingViolation] = useState(false);
  const [screenShareError, setScreenShareError] = useState<string | null>(null);

  const screenStreamRef = useRef<MediaStream | null>(null);
  const recorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<BlobPart[]>([]);
  const activeMetaRef = useRef<ActiveViolationRecordingMeta | null>(null);
  const autoStopTimerRef = useRef<number | null>(null);
  const pendingStopTimerRef = useRef<number | null>(null);

  const clearTimer = useCallback((timerRef: MutableRefObject<number | null>) => {
    if (timerRef.current !== null && typeof window !== 'undefined') {
      window.clearTimeout(timerRef.current);
      timerRef.current = null;
    }
  }, []);

  const emitRecorderError = useCallback(
    (error: Error) => {
      setScreenShareError(error.message);
      onError?.(error);
    },
    [onError],
  );

  const stopScreenShare = useCallback(() => {
    screenStreamRef.current?.getTracks().forEach((track) => track.stop());
    screenStreamRef.current = null;
    setHasScreenSharePermission(false);
  }, []);

  const requestScreenShare = useCallback(async () => {
    try {
      setScreenShareError(null);

      if (isScreenShareStreamActive(screenStreamRef.current)) {
        setHasScreenSharePermission(true);
        return true;
      }

      if (typeof navigator === 'undefined' || !navigator.mediaDevices?.getDisplayMedia || typeof MediaRecorder === 'undefined') {
        throw new Error('Screen recording is not supported in this browser.');
      }

      const stream = await navigator.mediaDevices.getDisplayMedia({
        video: {
          frameRate: 15,
          width: { ideal: 1280 },
          height: { ideal: 720 },
          displaySurface: 'monitor',
        } as MediaTrackConstraints,
        audio: false,
      });

      stopScreenShare();
      screenStreamRef.current = stream;
      setHasScreenSharePermission(true);

      const [videoTrack] = stream.getVideoTracks();
      if (videoTrack) {
        videoTrack.onended = () => {
          setHasScreenSharePermission(false);
          screenStreamRef.current = null;
          onScreenShareStopped?.();
        };
      }

      return true;
    } catch (error) {
      const err = error instanceof Error ? error : new Error('Failed to start screen sharing.');
      emitRecorderError(err);
      return false;
    }
  }, [emitRecorderError, onScreenShareStopped, stopScreenShare]);

  const finalizeRecording = useCallback(() => {
    const meta = activeMetaRef.current;
    const recorder = recorderRef.current;

    if (!meta || !recorder) return;

    const endedAtMs = Date.now();
    const endedAtISO = new Date(endedAtMs).toISOString();
    const mimeType = recorder.mimeType || getBestViolationRecordingMimeType() || 'video/webm';
    const blob = new Blob(chunksRef.current, { type: mimeType });
    const url = URL.createObjectURL(blob);

    const clip: ViolationRecordingClip = {
      id: meta.id,
      startedAt: meta.startedAtISO,
      endedAt: endedAtISO,
      durationMs: endedAtMs - meta.startedAtMs,
      violationTypes: Array.from(meta.violationTypes),
      events: meta.events,
      blob,
      url,
      mimeType,
    };

    chunksRef.current = [];
    recorderRef.current = null;
    activeMetaRef.current = null;
    setIsRecordingViolation(false);

    void onClipReady?.(clip);
  }, [onClipReady]);

  const stopViolationRecording = useCallback(() => {
    const recorder = recorderRef.current;
    const meta = activeMetaRef.current;

    if (!recorder || !meta || recorder.state === 'inactive') return;

    const elapsedMs = Date.now() - meta.startedAtMs;
    const stopNow = () => {
      clearTimer(autoStopTimerRef);
      clearTimer(pendingStopTimerRef);
      if (recorder.state !== 'inactive') {
        recorder.stop();
      }
    };

    if (elapsedMs < minClipMs && typeof window !== 'undefined') {
      clearTimer(pendingStopTimerRef);
      pendingStopTimerRef.current = window.setTimeout(stopNow, minClipMs - elapsedMs);
      return;
    }

    stopNow();
  }, [clearTimer, minClipMs]);

  const startViolationRecording = useCallback(
    (type: ProctoringViolationType, details?: Record<string, unknown>) => {
      try {
        setScreenShareError(null);

        if (typeof MediaRecorder === 'undefined') {
          throw new Error('Violation screen recording is not supported in this browser.');
        }

        if (!isScreenShareStreamActive(screenStreamRef.current)) {
          throw new Error('Screen share permission is missing. Ask the candidate to share their screen before starting the interview.');
        }

        const nowISO = new Date().toISOString();
        const activeMeta = activeMetaRef.current;

        if (activeMeta && recorderRef.current?.state === 'recording') {
          activeMeta.violationTypes.add(type);
          activeMeta.events.push({ type, at: nowISO, details });
          return true;
        }

        const mimeType = getBestViolationRecordingMimeType();
        chunksRef.current = [];

        const recorder = mimeType
          ? new MediaRecorder(screenStreamRef.current!, { mimeType })
          : new MediaRecorder(screenStreamRef.current!);

        const meta: ActiveViolationRecordingMeta = {
          id: makeViolationRecordingId(),
          startedAtISO: nowISO,
          startedAtMs: Date.now(),
          violationTypes: new Set([type]),
          events: [{ type, at: nowISO, details }],
        };

        recorder.ondataavailable = (event) => {
          if (event.data && event.data.size > 0) {
            chunksRef.current.push(event.data);
          }
        };

        recorder.onerror = () => {
          emitRecorderError(new Error('Screen violation recording failed.'));
        };

        recorder.onstop = finalizeRecording;

        recorderRef.current = recorder;
        activeMetaRef.current = meta;
        recorder.start(timesliceMs);
        setIsRecordingViolation(true);

        clearTimer(autoStopTimerRef);
        if (typeof window !== 'undefined') {
          autoStopTimerRef.current = window.setTimeout(() => {
            stopViolationRecording();
          }, maxClipMs);
        }

        return true;
      } catch (error) {
        const err = error instanceof Error ? error : new Error('Could not start violation recording.');
        emitRecorderError(err);
        return false;
      }
    },
    [clearTimer, emitRecorderError, finalizeRecording, maxClipMs, stopViolationRecording, timesliceMs],
  );

  const recordViolationClip = useCallback(
    (type: ProctoringViolationType, details?: Record<string, unknown>, clipMs = defaultClipMs) => {
      const started = startViolationRecording(type, details);
      if (!started) return false;
      if (typeof window === 'undefined') return true;

      window.setTimeout(() => {
        stopViolationRecording();
      }, Math.min(Math.max(clipMs, minClipMs), maxClipMs));

      return true;
    },
    [defaultClipMs, maxClipMs, minClipMs, startViolationRecording, stopViolationRecording],
  );

  useEffect(() => {
    return () => {
      clearTimer(autoStopTimerRef);
      clearTimer(pendingStopTimerRef);
      if (recorderRef.current && recorderRef.current.state !== 'inactive') {
        recorderRef.current.stop();
      }
      screenStreamRef.current?.getTracks().forEach((track) => track.stop());
      screenStreamRef.current = null;
    };
  }, [clearTimer]);

  return {
    hasScreenSharePermission,
    isRecordingViolation,
    screenShareError,
    requestScreenShare,
    stopScreenShare,
    startViolationRecording,
    stopViolationRecording,
    recordViolationClip,
  };
}

function isPhoneDetection(result: ObjectDetectorResult) {
  return (result.detections || []).some((detection) =>
    (detection.categories || []).some((category) => {
      const name = (category.categoryName || category.displayName || '').toLowerCase();
      return name.includes('cell phone') || name.includes('mobile phone') || name.includes('phone');
    }),
  );
}

function latestDetectionScore(result: ObjectDetectorResult) {
  const topDetection = result.detections?.[0];
  const topCategory = topDetection?.categories?.[0];
  return topCategory?.score ?? 0;
}

function getFacePoint(landmarks: FaceLandmarkerResult['faceLandmarks'][number] | undefined, index: number) {
  return landmarks?.[index];
}


type HeadPose = {
  yaw: number;
  pitch: number;
  roll: number;
  source: 'matrix' | 'landmarks';
};

type HeadPoseDeviation = {
  yaw: number;
  pitch: number;
  roll: number;
  magnitude: number;
  maxAxis: number;
  tooMuch: boolean;
};

function radiansToDegrees(value: number) {
  return (value * 180) / Math.PI;
}

function normalizeAngleDelta(current: number, baseline: number) {
  let delta = current - baseline;
  while (delta > 180) delta -= 360;
  while (delta < -180) delta += 360;
  return delta;
}

function getHeadPoseFromMatrix(result: FaceLandmarkerResult | null): HeadPose | null {
  const matrix = (result as any)?.facialTransformationMatrixes?.[0];
  const data = matrix?.data ?? matrix?.matrix ?? matrix;

  if (!data || typeof data.length !== 'number' || data.length < 16) return null;

  // MediaPipe exposes a 4x4 facial transformation matrix. We only need the 3x3 rotation portion.
  // These Euler values are used as relative deltas from calibration, so tiny convention differences are okay.
  const values = Array.from(data as ArrayLike<number>);
  const r00 = values[0];
  const r10 = values[4];
  const r20 = values[8];
  const r21 = values[9];
  const r22 = values[10];

  const sy = Math.sqrt(r00 * r00 + r10 * r10);
  const singular = sy < 1e-6;
  const pitch = singular ? 0 : radiansToDegrees(Math.atan2(r21, r22));
  const yaw = radiansToDegrees(Math.atan2(-r20, sy));
  const roll = radiansToDegrees(Math.atan2(r10, r00));

  if (![yaw, pitch, roll].every(Number.isFinite)) return null;
  return { yaw, pitch, roll, source: 'matrix' };
}

function getHeadPoseFromLandmarks(result: FaceLandmarkerResult | null): HeadPose | null {
  const landmarks = result?.faceLandmarks?.[0];
  const leftEyeOuter = getFacePoint(landmarks, 33);
  const rightEyeOuter = getFacePoint(landmarks, 263);
  const leftFace = getFacePoint(landmarks, 234);
  const rightFace = getFacePoint(landmarks, 454);
  const noseTip = getFacePoint(landmarks, 1);
  const forehead = getFacePoint(landmarks, 10);
  const chin = getFacePoint(landmarks, 152);

  if (!leftEyeOuter || !rightEyeOuter || !leftFace || !rightFace || !noseTip || !forehead || !chin) return null;

  const faceWidth = Math.max(Math.abs(rightFace.x - leftFace.x), 0.0001);
  const faceHeight = Math.max(Math.abs(chin.y - forehead.y), 0.0001);
  const eyeMidX = (leftEyeOuter.x + rightEyeOuter.x) / 2;
  const eyeMidY = (leftEyeOuter.y + rightEyeOuter.y) / 2;

  const yaw = ((noseTip.x - eyeMidX) / faceWidth) * 90;
  const pitch = ((noseTip.y - eyeMidY) / faceHeight) * 90;
  const roll = radiansToDegrees(Math.atan2(rightEyeOuter.y - leftEyeOuter.y, rightEyeOuter.x - leftEyeOuter.x));

  if (![yaw, pitch, roll].every(Number.isFinite)) return null;
  return { yaw, pitch, roll, source: 'landmarks' };
}

function estimateHeadPose(result: FaceLandmarkerResult | null): HeadPose | null {
  return getHeadPoseFromMatrix(result) ?? getHeadPoseFromLandmarks(result);
}

function calculateHeadPoseDeviation(current: HeadPose, baseline: HeadPose): HeadPoseDeviation {
  const yaw = normalizeAngleDelta(current.yaw, baseline.yaw);
  const pitch = normalizeAngleDelta(current.pitch, baseline.pitch);
  const roll = normalizeAngleDelta(current.roll, baseline.roll);
  const magnitude = Math.sqrt(yaw * yaw + pitch * pitch + roll * roll);
  const maxAxis = Math.max(Math.abs(yaw), Math.abs(pitch), Math.abs(roll));

  return {
    yaw,
    pitch,
    roll,
    magnitude,
    maxAxis,
    tooMuch: magnitude >= HEAD_POSE_ROTATION_THRESHOLD_DEG || maxAxis >= HEAD_POSE_AXIS_THRESHOLD_DEG,
  };
}

function clampHeadPoseForGazeCompensation(deltaDegrees: number) {
  if (!Number.isFinite(deltaDegrees)) return 0;
  const absDelta = Math.abs(deltaDegrees);
  if (absDelta <= HEAD_POSE_GAZE_COMPENSATION_START_DEG) return 0;

  const clampedAbsDelta = Math.min(absDelta, HEAD_POSE_GAZE_COMPENSATION_MAX_DEG);
  return Math.sign(deltaDegrees) * (clampedAbsDelta - HEAD_POSE_GAZE_COMPENSATION_START_DEG);
}

function reduceMagnitudeByHeadPose(value: number, deltaDegrees: number, factor: number) {
  const reduction = Math.abs(clampHeadPoseForGazeCompensation(deltaDegrees)) * factor;
  if (reduction <= 0) return value;
  const remainingMagnitude = Math.max(0, Math.abs(value) - reduction);
  return Math.sign(value) * remainingMagnitude;
}

function compensateGazeWithHeadPose(value: number, deltaDegrees: number, factor: number) {
  const clampedDelta = clampHeadPoseForGazeCompensation(deltaDegrees);
  if (clampedDelta === 0) return value;

  // Try signed compensation first. If MediaPipe/browser coordinate convention is flipped on a device,
  // fall back to a conservative magnitude reduction so compensation never makes the false positive worse.
  const signedCompensated = value + clampedDelta * factor;
  if (Math.abs(signedCompensated) < Math.abs(value)) return signedCompensated;

  return reduceMagnitudeByHeadPose(value, deltaDegrees, factor);
}

function detectGazeAway(
  result: FaceLandmarkerResult | null,
  thresholdX = DEFAULT_GAZE_THRESHOLD_X,
  thresholdY = DEFAULT_GAZE_THRESHOLD_Y,
  neutralX = 0,
  neutralY = 0,
  headPoseDeviation?: HeadPoseDeviation | null,
  // optional smoothing ref to reduce jitter / sensitivity
  filterRef?: { current: { x: number; y: number; initialized: boolean } } | null,
  smoothingAlpha = 0.25,
) {
  const faceLandmarks = result?.faceLandmarks?.[0];
  const faceBlendshapes = result?.faceBlendshapes?.[0]?.categories ?? [];

  const getBlendshapeScore = (name: string) =>
    faceBlendshapes.find((category) => (category.categoryName || category.displayName || '').toLowerCase() === name.toLowerCase())?.score ?? 0;

  const upLeft = getBlendshapeScore('eyeLookUpLeft');
  const upRight = getBlendshapeScore('eyeLookUpRight');
  const downLeft = getBlendshapeScore('eyeLookDownLeft');
  const downRight = getBlendshapeScore('eyeLookDownRight');
  const outLeft = getBlendshapeScore('eyeLookOutLeft');
  const outRight = getBlendshapeScore('eyeLookOutRight');
  const inLeft = getBlendshapeScore('eyeLookInLeft');
  const inRight = getBlendshapeScore('eyeLookInRight');

  const directionScores = [
    { direction: 'up', score: Math.min(upLeft, upRight) },
    { direction: 'down', score: Math.min(downLeft, downRight) },
    { direction: 'left', score: Math.min(outLeft, inRight) },
    { direction: 'right', score: Math.min(inLeft, outRight) },
  ];

  const bestBlendshape = directionScores
    .slice()
    .sort((a, b) => b.score - a.score)[0];

  // Iris-based gaze: iris position within the eye socket cancels out head rotation.
  // Indices: left eye corners 33/133, left iris 468, right eye corners 362/263, right iris 473.
  const leftEyeOuter  = getFacePoint(faceLandmarks, 33);
  const leftEyeInner  = getFacePoint(faceLandmarks, 133);
  const leftIris      = getFacePoint(faceLandmarks, 468);
  const rightEyeInner = getFacePoint(faceLandmarks, 362);
  const rightEyeOuter = getFacePoint(faceLandmarks, 263);
  const rightIris     = getFacePoint(faceLandmarks, 473);

  if (!leftEyeOuter || !leftEyeInner || !leftIris || !rightEyeInner || !rightEyeOuter || !rightIris) {
    if (bestBlendshape && bestBlendshape.score >= GAZE_BLENDSHAPE_THRESHOLD) {
      return {
        away: true,
        direction: bestBlendshape.direction,
        confidence: bestBlendshape.score,
        source: 'blendshape' as const,
        headPoseCompensated: false,
        uncompensatedX: 0,
        uncompensatedY: 0,
        compensatedX: 0,
        compensatedY: 0,
        yawDelta: headPoseDeviation?.yaw ?? 0,
        pitchDelta: headPoseDeviation?.pitch ?? 0,
      };
    }
    return {
      away: false,
      direction: 'center',
      confidence: 0,
      source: 'geometry' as const,
      headPoseCompensated: false,
      uncompensatedX: 0,
      uncompensatedY: 0,
      compensatedX: 0,
      compensatedY: 0,
      yawDelta: headPoseDeviation?.yaw ?? 0,
      pitchDelta: headPoseDeviation?.pitch ?? 0,
    };
  }

  const leftW  = Math.max(Math.abs(leftEyeOuter.x  - leftEyeInner.x),  0.0001);
  const rightW = Math.max(Math.abs(rightEyeOuter.x - rightEyeInner.x), 0.0001);
  const lUp    = getFacePoint(faceLandmarks, 159);
  const lDown  = getFacePoint(faceLandmarks, 145);
  const rUp    = getFacePoint(faceLandmarks, 386);
  const rDown  = getFacePoint(faceLandmarks, 374);
  const leftH  = Math.max(Math.abs((lUp?.y ?? leftIris.y)  - (lDown?.y ?? leftIris.y)),  0.0001);
  const rightH = Math.max(Math.abs((rUp?.y ?? rightIris.y) - (rDown?.y ?? rightIris.y)), 0.0001);

  const leftMidX  = (leftEyeOuter.x  + leftEyeInner.x)  / 2;
  const rightMidX = (rightEyeOuter.x + rightEyeInner.x) / 2;
  const leftMidY  = ((lUp?.y ?? leftIris.y) + (lDown?.y ?? leftIris.y)) / 2;
  const rightMidY = ((rUp?.y ?? rightIris.y) + (rDown?.y ?? rightIris.y)) / 2;

  const rawOffsetX = ((leftIris.x - leftMidX) / (leftW / 2) + (rightIris.x - rightMidX) / (rightW / 2)) / 2;
  const rawOffsetY = ((leftIris.y - leftMidY) / (leftH / 2) + (rightIris.y - rightMidY) / (rightH / 2)) / 2;

  // Subtract the calibrated neutral so eyes-forward is always (0,0)
  const adjOffsetX = rawOffsetX - neutralX;
  const adjOffsetY = rawOffsetY - neutralY;

  // Thresholds are sanitized by eightDotCalibrationGuardV3 before live monitoring uses them.
  // Do not expand them here using calibration extremes; that would allow fake calibration to enlarge the safe zone.
  const effectiveThresholdX = thresholdX;
  const effectiveThresholdUp = thresholdY * 0.975;
  const effectiveThresholdDown = Math.max(
    thresholdY * DOWNWARD_GAZE_THRESHOLD_FACTOR,
    MIN_DOWNWARD_GAZE_THRESHOLD,
  );

  // Apply optional exponential smoothing to reduce spurious detections from jitter.
  let useX = adjOffsetX;
  let useY = adjOffsetY;
  if (filterRef) {
    const f = filterRef.current;
    if (!f.initialized) {
      f.x = adjOffsetX;
      f.y = adjOffsetY;
      f.initialized = true;
      useX = adjOffsetX;
      useY = adjOffsetY;
    } else {
      // low-pass: new = old*(1-a) + current*a
      f.x = f.x * (1 - smoothingAlpha) + adjOffsetX * smoothingAlpha;
      f.y = f.y * (1 - smoothingAlpha) + adjOffsetY * smoothingAlpha;
      useX = f.x;
      useY = f.y;
    }
  }

  const uncompensatedX = useX;
  const uncompensatedY = useY;
  const compensatedX = headPoseDeviation
    ? compensateGazeWithHeadPose(uncompensatedX, headPoseDeviation.yaw, HEAD_YAW_TO_GAZE_X_FACTOR)
    : uncompensatedX;
  const compensatedY = headPoseDeviation
    ? compensateGazeWithHeadPose(uncompensatedY, headPoseDeviation.pitch, HEAD_PITCH_TO_GAZE_Y_FACTOR)
    : uncompensatedY;
  const headPoseCompensated =
    Math.abs(compensatedX - uncompensatedX) > 0.0001 ||
    Math.abs(compensatedY - uncompensatedY) > 0.0001;

  useX = compensatedX;
  useY = compensatedY;

  const upBlendshapeScore = Math.min(upLeft, upRight);
  const downBlendshapeScore = Math.min(downLeft, downRight);
  const horizontalGazeAway = Math.abs(useX) >= effectiveThresholdX;
  const upwardGazeAway = useY <= -effectiveThresholdUp;
  const downwardGazeAway = useY >= effectiveThresholdDown;
  const downwardGeometrySupport = Math.max(
    effectiveThresholdDown * DOWNWARD_GEOMETRY_SUPPORT_FACTOR,
    MIN_DOWNWARD_GEOMETRY_SUPPORT,
  );
  const headPoseExplainsVerticalCompensation = Boolean(
    headPoseDeviation &&
    Math.abs(headPoseDeviation.pitch) > HEAD_POSE_GAZE_COMPENSATION_START_DEG &&
    Math.sign(uncompensatedY) !== 0 &&
    Math.abs(useY) < Math.abs(uncompensatedY) &&
    !upwardGazeAway &&
    !downwardGazeAway,
  );
  const downwardBlendshapeAway =
    !headPoseExplainsVerticalCompensation &&
    downBlendshapeScore >= DOWNWARD_BLENDSHAPE_AWAY_THRESHOLD &&
    downBlendshapeScore > upBlendshapeScore + DOWNWARD_BLENDSHAPE_MARGIN &&
    // Avoid triggering "down" from tiny positive-Y jitter. Blendshape can still help,
    // but it now needs either mild geometry support or a very strong down score.
    (
      useY >= downwardGeometrySupport ||
      downBlendshapeScore >= DOWNWARD_BLENDSHAPE_AWAY_THRESHOLD + 0.22
    );

  if (horizontalGazeAway || upwardGazeAway || downwardGazeAway || downwardBlendshapeAway) {
    const horizontal = useX > 0 ? 'left' : 'right';

    // Geometry is still the primary signal, but MediaPipe's eyeLookUp/Down blendshapes
    // are more reliable for distinguishing vertical direction when the eyelid hides the iris.
    let vertical = useY > 0 || downwardBlendshapeAway ? 'down' : 'up';
    let verticalSource: 'geometry' | 'blendshape' = downwardBlendshapeAway ? 'blendshape' : 'geometry';

    if (
      downBlendshapeScore >= VERTICAL_BLENDSHAPE_DIRECTION_THRESHOLD &&
      downBlendshapeScore > upBlendshapeScore + VERTICAL_BLENDSHAPE_MARGIN
    ) {
      vertical = 'down';
      verticalSource = 'blendshape';
    } else if (
      upBlendshapeScore >= VERTICAL_BLENDSHAPE_DIRECTION_THRESHOLD &&
      upBlendshapeScore > downBlendshapeScore + VERTICAL_BLENDSHAPE_MARGIN
    ) {
      vertical = 'up';
      verticalSource = 'blendshape';
    }

    // Use normalized strength so the lower downward threshold actually affects direction choice.
    // Without this, small X jitter could still beat a real downward gaze.
    const horizontalStrength = Math.abs(useX) / Math.max(effectiveThresholdX, 0.0001);
    const verticalThreshold = useY > 0 || downwardBlendshapeAway ? effectiveThresholdDown : effectiveThresholdUp;
    const verticalStrength = Math.abs(useY) / Math.max(verticalThreshold, 0.0001);
    const horizontalDeadzoneMargin = 1.15;
    const direction = horizontalStrength * horizontalDeadzoneMargin > verticalStrength && !downwardBlendshapeAway ? horizontal : vertical;

    return {
      away: true,
      direction,
      confidence: Math.max(Math.abs(useX), Math.abs(useY), downwardBlendshapeAway ? downBlendshapeScore : 0),
      source: direction === vertical ? verticalSource : 'geometry',
      headPoseCompensated,
      uncompensatedX,
      uncompensatedY,
      compensatedX: useX,
      compensatedY: useY,
      yawDelta: headPoseDeviation?.yaw ?? 0,
      pitchDelta: headPoseDeviation?.pitch ?? 0,
    };
  }

  return {
    away: false,
    direction: 'center',
    confidence: 0,
    source: 'geometry' as const,
    headPoseCompensated,
    uncompensatedX,
    uncompensatedY,
    compensatedX: useX,
    compensatedY: useY,
    yawDelta: headPoseDeviation?.yaw ?? 0,
    pitchDelta: headPoseDeviation?.pitch ?? 0,
  };
}


function getVideoTrack(stream: MediaStream | null | undefined): MediaStreamTrack | null {
  return stream?.getVideoTracks?.()[0] ?? null;
}

function getCameraHealthIssue(video: HTMLVideoElement | null, stream: MediaStream | null | undefined) {
  const track = getVideoTrack(stream);

  if (!stream) return 'Camera stream is not available';
  if (!stream.active) return 'Camera stream is inactive';
  if (!track) return 'No video track found in camera stream';
  if (track.readyState !== 'live') return `Video track is ${track.readyState}`;
  if (!track.enabled) return 'Video track is disabled';
  if (track.muted) return 'Video track is muted or unavailable';
  if (!video) return 'Video element is unavailable';
  if (video.srcObject !== stream) return 'Video element is not connected to the active camera stream';
  if (video.ended) return 'Video element has ended';
  if (video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) return 'Camera is not sending video frames';
  if (video.videoWidth <= 0 || video.videoHeight <= 0) return 'Camera video dimensions are unavailable';

  return null;
}

function getFullscreenElement() {
  if (typeof document === 'undefined') return null;
  const doc = document as Document & {
    webkitFullscreenElement?: Element | null;
    msFullscreenElement?: Element | null;
  };

  return document.fullscreenElement ?? doc.webkitFullscreenElement ?? doc.msFullscreenElement ?? null;
}

function isFullscreenSupported() {
  if (typeof document === 'undefined') return false;
  const doc = document as Document & {
    webkitFullscreenEnabled?: boolean;
    msFullscreenEnabled?: boolean;
  };
  const element = document.documentElement as HTMLElement & {
    webkitRequestFullscreen?: () => Promise<void> | void;
    msRequestFullscreen?: () => Promise<void> | void;
  };

  const fullscreenEnabled = document.fullscreenEnabled ?? doc.webkitFullscreenEnabled ?? doc.msFullscreenEnabled ?? false;
  const requestFullscreen = element.requestFullscreen ?? element.webkitRequestFullscreen ?? element.msRequestFullscreen;

  return Boolean(fullscreenEnabled && requestFullscreen);
}

function getRequestFullscreen() {
  if (typeof document === 'undefined') return null;
  const element = document.documentElement as HTMLElement & {
    webkitRequestFullscreen?: () => Promise<void> | void;
    msRequestFullscreen?: () => Promise<void> | void;
  };

  return element.requestFullscreen ?? element.webkitRequestFullscreen ?? element.msRequestFullscreen ?? null;
}

export function useProctoring(sessionId: string, socket?: WebSocket | null, calibration?: CalibrationResult | null) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [events, setEvents] = useState<ProctoringEvent[]>([]);
  const [state, setState] = useState<DetectionState>({
    initialized: false,
    status: 'Initializing camera...',
    permissionDenied: false,
    cameraActive: false,
    faceDetectorActive: false,
    objectDetectorActive: false,
    faceCount: 0,
    phoneDetected: false,
    gazeAwayDetected: false,
    gazeDirection: 'center',
    headMovementDetected: false,
    headPoseDeviationDegrees: 0,
    tabSwitchDetected: false,
    tabSwitchReason: null,
    lastTabSwitchAt: null,
    lastTabSwitchDurationMs: null,
    activeTabSwitchDurationMs: 0,
    totalTabSwitchDurationMs: 0,
    tabSwitchCount: 0,
    fullscreenActive: false,
    fullscreenExitDetected: false,
    fullscreenExitReason: null,
    lastFullscreenExitAt: null,
    fullscreenSupported: typeof document === 'undefined' ? true : isFullscreenSupported(),
    fullscreenReadyBeforeInterview: false,
    fullscreenPromptRequired: typeof document === 'undefined' ? false : isFullscreenSupported(),
    preInterviewFullscreenRequestedAt: null,
    preInterviewFullscreenEnteredAt: null,
    screenShareSupported: typeof navigator === 'undefined' ? true : isScreenShareRecordingSupported(),
    screenShareReadyBeforeInterview: false,
    screenSharePromptRequired: true,
    preInterviewScreenShareRequestedAt: null,
    preInterviewScreenShareGrantedAt: null,
    lastObservationAt: null,
  });
  const [violationRecordings, setViolationRecordings] = useState<ViolationRecordingClip[]>([]);
  const violationRecordingErrorAlertAt = useRef<number>(0);
  const missingSince = useRef<number | null>(null);
  const faceAlertAt = useRef<number>(0);
  const phoneAlertAt = useRef<number>(0);
  const gazeAlertAt = useRef<number>(0);
  const headPoseAlertAt = useRef<number>(0);
  const cameraOffAlertAt = useRef<number>(0);
  const tabSwitchAlertAt = useRef<number>(0);
  const multiFaceAlertAt = useRef<number>(0);
  const multiFaceSince = useRef<number | null>(null);
  const phoneSince = useRef<number | null>(null);
  const gazeSince = useRef<number | null>(null);
  const headPoseSince = useRef<number | null>(null);
  const cameraOffSince = useRef<number | null>(null);
  const cameraOffIncidentActive = useRef(false);
  const tabSwitchSince = useRef<number | null>(null);
  const tabSwitchIncidentActive = useRef(false);
  const tabSwitchReasonRef = useRef<string | null>(null);
  const tabSwitchDurationTimerRef = useRef<number | null>(null);
  const fullscreenAlertAt = useRef<number>(0);
  const fullscreenExitSince = useRef<number | null>(null);
  const fullscreenIncidentActive = useRef(false);
  const fullscreenEverEnteredRef = useRef(false);
  const fullscreenRequestInFlightRef = useRef(false);
  const screenShareGrantedRef = useRef(false);
  const screenShareRequestInFlightRef = useRef(false);
  const fullscreenExitReasonRef = useRef<string | null>(null);
  const faceLandmarkerRef = useRef<FaceLandmarker | null>(null);
  const objectDetectorRef = useRef<ObjectDetector | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const frameTimerRef = useRef<number | null>(null);
  const aliveRef = useRef(true);
  const safeCalibrationRef = useRef<SafeEightDotCalibration>(buildSafeEightDotCalibration(calibration));
  const gazeFilterRef = useRef<{ x: number; y: number; initialized: boolean }>({ x: 0, y: 0, initialized: false });
  const headPoseBaselineRef = useRef<HeadPose | null>(null);

  useEffect(() => {
    safeCalibrationRef.current = buildSafeEightDotCalibration(calibration);
    gazeFilterRef.current = { x: 0, y: 0, initialized: false };
    headPoseBaselineRef.current = null;
    headPoseSince.current = null;
  }, [calibration]);

  const emit = useCallback((eventType: string, severity: Severity, metadata: Record<string, unknown> = {}) => {
    const now = Date.now();
    const payload: ProctoringPayload = { type: 'proctoring_event', sessionId, eventType, severity, metadata, timestamp: now };
    socket?.readyState === 1 && socket.send(JSON.stringify(payload));
    setEvents((current) => [{ eventType, severity, timestamp: now, metadata }, ...current].slice(0, 10));
  }, [sessionId, socket]);

  function emitWithCooldown(ref: { current: number }, eventType: string, severity: Severity, metadata: Record<string, unknown> = {}) {
    const now = Date.now();
    if (now - ref.current < ALERT_COOLDOWN_MS) return false;
    ref.current = now;
    emit(eventType, severity, metadata);
    return true;
  }

  function emitTabSwitchWithCooldown(metadata: Record<string, unknown> = {}) {
    const now = Date.now();
    if (now - tabSwitchAlertAt.current < TAB_SWITCH_ALERT_COOLDOWN_MS) return false;
    tabSwitchAlertAt.current = now;
    emit('TAB_SWITCH_DETECTED', 'HIGH', metadata);
    return true;
  }

  function emitFullscreenWithCooldown(eventType: string, severity: Severity, metadata: Record<string, unknown> = {}) {
    const now = Date.now();
    if (now - fullscreenAlertAt.current < FULLSCREEN_ALERT_COOLDOWN_MS) return false;
    fullscreenAlertAt.current = now;
    emit(eventType, severity, metadata);
    return true;
  }


  const handleViolationRecordingReady = useCallback(
    (clip: ViolationRecordingClip) => {
      setViolationRecordings((current) => [clip, ...current].slice(0, 20));
      emit('VIOLATION_SCREEN_RECORDING_READY', 'MEDIUM', {
        recordingId: clip.id,
        startedAt: clip.startedAt,
        endedAt: clip.endedAt,
        durationMs: clip.durationMs,
        violationTypes: clip.violationTypes,
        eventCount: clip.events.length,
        mimeType: clip.mimeType,
        sizeBytes: clip.blob.size,
      });
    },
    [emit],
  );

  const handleViolationRecordingError = useCallback(
    (error: Error) => {
      const now = Date.now();
      if (now - violationRecordingErrorAlertAt.current < ALERT_COOLDOWN_MS) return;

      violationRecordingErrorAlertAt.current = now;
      emit('VIOLATION_SCREEN_RECORDING_ERROR', 'MEDIUM', {
        message: error.message,
        screenShareRequired: true,
      });
    },
    [emit],
  );

  const handleScreenShareStopped = useCallback(() => {
    const now = Date.now();
    setState((current) => ({
      ...current,
      screenShareReadyBeforeInterview: false,
      screenSharePromptRequired: true,
      lastObservationAt: now,
      status: 'Screen sharing stopped',
    }));

    emit('SCREEN_SHARE_STOPPED', 'HIGH', {
      reason: 'candidate_stopped_screen_share',
      screenShareRequired: true,
      stoppedAt: now,
    });
  }, [emit]);

  const violationScreenRecorder = useViolationScreenRecorder({
    defaultClipMs: DEFAULT_VIOLATION_SCREEN_CLIP_MS,
    minClipMs: MIN_VIOLATION_SCREEN_CLIP_MS,
    maxClipMs: MAX_VIOLATION_SCREEN_CLIP_MS,
    timesliceMs: VIOLATION_SCREEN_RECORDING_TIMESLICE_MS,
    onClipReady: handleViolationRecordingReady,
    onError: handleViolationRecordingError,
    onScreenShareStopped: handleScreenShareStopped,
  });

  useEffect(() => {
    const now = Date.now();
    screenShareGrantedRef.current = violationScreenRecorder.hasScreenSharePermission;
    setState((current) => ({
      ...current,
      screenShareSupported: isScreenShareRecordingSupported(),
      screenShareReadyBeforeInterview: violationScreenRecorder.hasScreenSharePermission,
      screenSharePromptRequired: !violationScreenRecorder.hasScreenSharePermission,
      preInterviewScreenShareGrantedAt: violationScreenRecorder.hasScreenSharePermission
        ? current.preInterviewScreenShareGrantedAt ?? now
        : current.preInterviewScreenShareGrantedAt,
    }));
  }, [violationScreenRecorder.hasScreenSharePermission]);

  async function requestExamFullscreen(trigger = 'manual') {
    if (typeof window === 'undefined' || typeof document === 'undefined') return false;

    const supported = isFullscreenSupported();
    const now = Date.now();
    setState((current) => ({
      ...current,
      fullscreenSupported: supported,
      fullscreenPromptRequired: supported && !Boolean(getFullscreenElement()),
      lastObservationAt: now,
    }));

    if (!supported) {
      emitFullscreenWithCooldown('FULLSCREEN_UNAVAILABLE', 'HIGH', {
        trigger,
        reason: 'fullscreen_api_unavailable_or_disabled',
      });
      setState((current) => ({
        ...current,
        fullscreenActive: false,
        fullscreenExitDetected: true,
        fullscreenExitReason: 'fullscreen_api_unavailable_or_disabled',
        lastFullscreenExitAt: now,
        fullscreenSupported: false,
        fullscreenReadyBeforeInterview: false,
        fullscreenPromptRequired: false,
        lastObservationAt: now,
        status: 'Fullscreen unavailable',
      }));
      return false;
    }

    if (getFullscreenElement()) {
      fullscreenEverEnteredRef.current = true;
      setState((current) => ({
        ...current,
        fullscreenActive: true,
        fullscreenExitDetected: false,
        fullscreenExitReason: null,
        fullscreenSupported: true,
        fullscreenReadyBeforeInterview: true,
        fullscreenPromptRequired: false,
        preInterviewFullscreenEnteredAt: current.preInterviewFullscreenEnteredAt ?? now,
        lastObservationAt: now,
        status: current.tabSwitchDetected ? 'Tab/window switch detected' : 'Fullscreen ready',
      }));
      return true;
    }

    if (fullscreenRequestInFlightRef.current) return false;

    const requestFullscreen = getRequestFullscreen();
    if (!requestFullscreen) return false;

    fullscreenRequestInFlightRef.current = true;
    try {
      const result = requestFullscreen.call(document.documentElement);
      if (result && typeof (result as Promise<void>).then === 'function') {
        await result;
      }
      fullscreenEverEnteredRef.current = true;
      setState((current) => ({
        ...current,
        fullscreenActive: true,
        fullscreenExitDetected: false,
        fullscreenExitReason: null,
        fullscreenSupported: true,
        fullscreenReadyBeforeInterview: true,
        fullscreenPromptRequired: false,
        preInterviewFullscreenEnteredAt: current.preInterviewFullscreenEnteredAt ?? Date.now(),
        lastObservationAt: Date.now(),
        status: current.tabSwitchDetected ? 'Tab/window switch detected' : 'Fullscreen ready',
      }));
      return true;
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Fullscreen request failed';
      emitFullscreenWithCooldown('FULLSCREEN_REQUEST_FAILED', 'MEDIUM', {
        trigger,
        message,
        // Most browsers reject fullscreen requests that are not started by a user gesture.
        userGestureRequired: true,
      });
      setState((current) => ({
        ...current,
        fullscreenActive: Boolean(getFullscreenElement()),
        fullscreenSupported: true,
        fullscreenReadyBeforeInterview: Boolean(getFullscreenElement()),
        fullscreenPromptRequired: !Boolean(getFullscreenElement()),
        lastObservationAt: Date.now(),
        status: current.fullscreenExitDetected
          ? 'Fullscreen exited'
          : current.tabSwitchDetected
            ? 'Tab/window switch detected'
            : 'Click to enable fullscreen',
      }));
      return false;
    } finally {
      fullscreenRequestInFlightRef.current = false;
    }
  }

  async function requestScreenShareBeforeInterview(trigger = 'pre_interview_start_button') {
    const supported = isScreenShareRecordingSupported();
    const now = Date.now();
    const alreadyGranted = screenShareGrantedRef.current || violationScreenRecorder.hasScreenSharePermission;

    setState((current) => ({
      ...current,
      screenShareSupported: supported,
      screenSharePromptRequired: supported && !alreadyGranted,
      preInterviewScreenShareRequestedAt: current.preInterviewScreenShareRequestedAt ?? now,
      status: alreadyGranted
        ? current.status
        : supported
          ? 'Please share your screen to start the interview'
          : 'Screen recording unavailable',
      lastObservationAt: now,
    }));

    if (!supported) {
      emit('PRE_INTERVIEW_SCREEN_SHARE_UNAVAILABLE', 'HIGH', {
        trigger,
        reason: 'screen_recording_api_unavailable_or_disabled',
      });
      return false;
    }

    if (alreadyGranted) {
      setState((current) => ({
        ...current,
        screenShareSupported: true,
        screenShareReadyBeforeInterview: true,
        screenSharePromptRequired: false,
        preInterviewScreenShareGrantedAt: current.preInterviewScreenShareGrantedAt ?? now,
        lastObservationAt: now,
      }));
      return true;
    }

    if (screenShareRequestInFlightRef.current) {
      return false;
    }

    screenShareRequestInFlightRef.current = true;
    try {
      const granted = await violationScreenRecorder.requestScreenShare();
      const completedAt = Date.now();
      screenShareGrantedRef.current = granted;

      setState((current) => ({
        ...current,
        screenShareSupported: true,
        screenShareReadyBeforeInterview: granted,
        screenSharePromptRequired: !granted,
        preInterviewScreenShareGrantedAt: granted ? completedAt : current.preInterviewScreenShareGrantedAt,
        lastObservationAt: completedAt,
        status: granted
          ? current.fullscreenActive
            ? 'Screen sharing and fullscreen ready. You can start the interview.'
            : 'Screen sharing ready. Please enter fullscreen.'
          : 'Please share your screen before starting the interview',
      }));

      emit(granted ? 'PRE_INTERVIEW_SCREEN_SHARE_CONFIRMED' : 'PRE_INTERVIEW_SCREEN_SHARE_FAILED', granted ? 'MEDIUM' : 'HIGH', {
        trigger,
        requestedAt: now,
        completedAt,
        screenShareRequired: true,
        error: granted ? undefined : violationScreenRecorder.screenShareError,
      });

      return granted;
    } finally {
      screenShareRequestInFlightRef.current = false;
    }
  }

  async function enterFullscreenBeforeInterview(trigger = 'pre_interview_fullscreen_button') {
    const screenShareReady = await requestScreenShareBeforeInterview(trigger);
    if (!screenShareReady) return false;

    const now = Date.now();
    setState((current) => ({
      ...current,
      preInterviewFullscreenRequestedAt: now,
      fullscreenPromptRequired: !Boolean(getFullscreenElement()) && isFullscreenSupported(),
      status: Boolean(getFullscreenElement()) ? 'Fullscreen ready' : 'Entering fullscreen...',
      lastObservationAt: now,
    }));

    const entered = await requestExamFullscreen(trigger);
    const active = Boolean(getFullscreenElement());
    const enteredAt = Date.now();

    setState((current) => ({
      ...current,
      fullscreenActive: active,
      fullscreenReadyBeforeInterview: entered && active,
      fullscreenPromptRequired: !active && isFullscreenSupported(),
      preInterviewFullscreenEnteredAt: entered && active ? enteredAt : current.preInterviewFullscreenEnteredAt,
      lastObservationAt: enteredAt,
      status: entered && active
        ? 'Fullscreen ready. You can start the interview.'
        : current.fullscreenExitDetected
          ? 'Fullscreen exited'
          : 'Please enter fullscreen before starting the interview',
    }));

    if (entered && active) {
      emit('PRE_INTERVIEW_FULLSCREEN_CONFIRMED', 'MEDIUM', {
        trigger,
        requestedAt: now,
        enteredAt,
      });
    }

    return entered && active;
  }

  async function prepareInterviewStart() {
    return enterFullscreenBeforeInterview('start_interview_button');
  }

  function clearTabSwitchDurationTimer() {
    if (tabSwitchDurationTimerRef.current !== null && typeof window !== 'undefined') {
      window.clearInterval(tabSwitchDurationTimerRef.current);
      tabSwitchDurationTimerRef.current = null;
    }
  }

  function startTabSwitchDurationTimer() {
    if (typeof window === 'undefined' || tabSwitchDurationTimerRef.current !== null) return;

    tabSwitchDurationTimerRef.current = window.setInterval(() => {
      const startedAt = tabSwitchSince.current;
      if (!startedAt) return;

      const activeDurationMs = Date.now() - startedAt;
      setState((current) => ({
        ...current,
        activeTabSwitchDurationMs: activeDurationMs,
        lastObservationAt: Date.now(),
      }));
    }, 1000);
  }

  useEffect(() => {
    if (typeof window === 'undefined' || typeof document === 'undefined') return;

    const getTabAwayReason = (trigger: string) => {
      if (document.visibilityState === 'hidden') return 'document_hidden';
      if (!document.hasFocus()) return trigger === 'blur' ? 'window_blur' : 'window_not_focused';
      return null;
    };

    const markTabAway = (trigger: string) => {
      const reason = getTabAwayReason(trigger);
      if (!reason) return;

      const now = Date.now();
      if (!tabSwitchSince.current) tabSwitchSince.current = now;
      tabSwitchReasonRef.current = reason;
      startTabSwitchDurationTimer();

      setState((current) => ({
        ...current,
        tabSwitchDetected: true,
        tabSwitchReason: reason,
        lastTabSwitchAt: now,
        activeTabSwitchDurationMs: tabSwitchSince.current ? now - tabSwitchSince.current : 0,
        lastObservationAt: now,
        status: current.fullscreenExitDetected ? 'Fullscreen exited' : 'Tab/window switch detected',
      }));

      if (!tabSwitchIncidentActive.current) {
        tabSwitchIncidentActive.current = true;
        const metadata = {
          reason,
          trigger,
          startedAt: tabSwitchSince.current,
          activeDurationMs: tabSwitchSince.current ? now - tabSwitchSince.current : 0,
          visibilityState: document.visibilityState,
          documentHidden: document.hidden,
          hasFocus: document.hasFocus(),
        };

        if (emitTabSwitchWithCooldown(metadata)) {
          violationScreenRecorder.startViolationRecording('TAB_SWITCH', metadata);
        }
      }
    };

    const markTabReturned = (trigger: string) => {
      if (getTabAwayReason(trigger)) return;
      if (!tabSwitchIncidentActive.current) return;

      const now = Date.now();
      const startedAt = tabSwitchSince.current;
      const durationMs = startedAt ? now - startedAt : 0;
      const durationSeconds = Math.round((durationMs / 1000) * 10) / 10;
      const reason = tabSwitchReasonRef.current;

      clearTabSwitchDurationTimer();

      emit('TAB_RETURNED', 'MEDIUM', {
        reason,
        trigger,
        startedAt,
        returnedAt: now,
        durationMs,
        durationSeconds,
        visibilityState: document.visibilityState,
        documentHidden: document.hidden,
        hasFocus: document.hasFocus(),
      });
      violationScreenRecorder.stopViolationRecording();

      tabSwitchSince.current = null;
      tabSwitchIncidentActive.current = false;
      tabSwitchReasonRef.current = null;

      setState((current) => ({
        ...current,
        tabSwitchDetected: false,
        tabSwitchReason: null,
        lastTabSwitchDurationMs: durationMs,
        activeTabSwitchDurationMs: 0,
        totalTabSwitchDurationMs: current.totalTabSwitchDurationMs + durationMs,
        tabSwitchCount: current.tabSwitchCount + 1,
        lastObservationAt: now,
        status: current.fullscreenExitDetected
          ? 'Fullscreen exited'
          : current.cameraActive
            ? 'Detection active'
            : current.status,
      }));
    };

    const handleVisibilityChange = () => {
      if (document.visibilityState === 'hidden') {
        markTabAway('visibilitychange');
      } else {
        markTabReturned('visibilitychange');
      }
    };

    const handleBlur = () => markTabAway('blur');
    const handleFocus = () => markTabReturned('focus');
    const handlePageHide = () => markTabAway('pagehide');
    const handlePageShow = () => markTabReturned('pageshow');

    document.addEventListener('visibilitychange', handleVisibilityChange);
    window.addEventListener('blur', handleBlur);
    window.addEventListener('focus', handleFocus);
    window.addEventListener('pagehide', handlePageHide);
    window.addEventListener('pageshow', handlePageShow);

    // If the hook mounts while the exam page is already not focused/visible, flag it immediately.
    markTabAway('mount');

    return () => {
      document.removeEventListener('visibilitychange', handleVisibilityChange);
      window.removeEventListener('blur', handleBlur);
      window.removeEventListener('focus', handleFocus);
      window.removeEventListener('pagehide', handlePageHide);
      window.removeEventListener('pageshow', handlePageShow);
      tabSwitchSince.current = null;
      tabSwitchIncidentActive.current = false;
      tabSwitchReasonRef.current = null;
      clearTabSwitchDurationTimer();
    };
  }, [emit, violationScreenRecorder.startViolationRecording, violationScreenRecorder.stopViolationRecording]);

  useEffect(() => {
    if (typeof window === 'undefined' || typeof document === 'undefined') return;

    const markFullscreenEntered = (trigger: string) => {
      const now = Date.now();
      const durationMs = fullscreenExitSince.current ? now - fullscreenExitSince.current : 0;
      const previousReason = fullscreenExitReasonRef.current;

      fullscreenEverEnteredRef.current = true;
      fullscreenExitSince.current = null;
      fullscreenExitReasonRef.current = null;

      if (fullscreenIncidentActive.current) {
        fullscreenIncidentActive.current = false;
        emit('FULLSCREEN_RESTORED', 'MEDIUM', {
          trigger,
          previousReason,
          durationMs,
        });
        violationScreenRecorder.stopViolationRecording();
      } else {
        emit('FULLSCREEN_ENTERED', 'MEDIUM', { trigger });
      }

      setState((current) => ({
        ...current,
        fullscreenActive: true,
        fullscreenExitDetected: false,
        fullscreenExitReason: null,
        fullscreenSupported: true,
        fullscreenReadyBeforeInterview: true,
        fullscreenPromptRequired: false,
        preInterviewFullscreenEnteredAt: current.preInterviewFullscreenEnteredAt ?? now,
        lastObservationAt: now,
        status: current.tabSwitchDetected ? 'Tab/window switch detected' : 'Fullscreen ready',
      }));
    };

    const markFullscreenExited = (trigger: string, reason = 'fullscreen_exited') => {
      const now = Date.now();
      const shouldFlag = fullscreenEverEnteredRef.current || fullscreenIncidentActive.current;

      if (!shouldFlag) {
        setState((current) => ({
          ...current,
          fullscreenActive: false,
          fullscreenSupported: isFullscreenSupported(),
          fullscreenReadyBeforeInterview: false,
          fullscreenPromptRequired: isFullscreenSupported(),
          lastObservationAt: now,
        }));
        return;
      }

      if (!fullscreenExitSince.current) fullscreenExitSince.current = now;
      fullscreenExitReasonRef.current = reason;
      fullscreenIncidentActive.current = true;

      const metadata = {
        trigger,
        reason,
        fullscreenElementPresent: Boolean(getFullscreenElement()),
        visibilityState: document.visibilityState,
        documentHidden: document.hidden,
        hasFocus: document.hasFocus(),
      };

      if (emitFullscreenWithCooldown('FULLSCREEN_EXITED_DETECTED', 'HIGH', metadata)) {
        violationScreenRecorder.startViolationRecording('FULLSCREEN_EXIT', metadata);
      }

      setState((current) => ({
        ...current,
        fullscreenActive: false,
        fullscreenExitDetected: true,
        fullscreenExitReason: reason,
        lastFullscreenExitAt: now,
        fullscreenSupported: isFullscreenSupported(),
        fullscreenReadyBeforeInterview: false,
        fullscreenPromptRequired: isFullscreenSupported(),
        lastObservationAt: now,
        status: 'Fullscreen exited',
      }));
    };

    const handleFullscreenChange = () => {
      if (getFullscreenElement()) {
        markFullscreenEntered('fullscreenchange');
      } else {
        markFullscreenExited('fullscreenchange');
      }
    };

    const handleFullscreenError = () => {
      emitFullscreenWithCooldown('FULLSCREEN_REQUEST_FAILED', 'MEDIUM', {
        trigger: 'fullscreenerror',
        reason: 'browser_fullscreen_error',
      });
    };

    document.addEventListener('fullscreenchange', handleFullscreenChange);
    document.addEventListener('fullscreenerror', handleFullscreenError);
    document.addEventListener('webkitfullscreenchange', handleFullscreenChange as EventListener);
    document.addEventListener('webkitfullscreenerror', handleFullscreenError as EventListener);
    document.addEventListener('MSFullscreenChange', handleFullscreenChange as EventListener);
    document.addEventListener('MSFullscreenError', handleFullscreenError as EventListener);

    setState((current) => ({
      ...current,
      fullscreenActive: Boolean(getFullscreenElement()),
      fullscreenSupported: isFullscreenSupported(),
      fullscreenReadyBeforeInterview: Boolean(getFullscreenElement()),
      fullscreenPromptRequired: !Boolean(getFullscreenElement()) && isFullscreenSupported(),
    }));

    // Do not call getDisplayMedia() automatically here: browsers only show the
    // screen-share prompt from a real user gesture. The first click/key press
    // runs the full pre-interview permission flow instead of fullscreen alone.
    const requestOnUserGesture = () => {
      const screenShareMissing = !screenShareGrantedRef.current && !violationScreenRecorder.hasScreenSharePermission;
      const fullscreenMissing = !getFullscreenElement();

      if ((screenShareMissing || fullscreenMissing) && !fullscreenRequestInFlightRef.current && !screenShareRequestInFlightRef.current) {
        void enterFullscreenBeforeInterview('user_interaction');
      }
    };

    window.addEventListener('pointerdown', requestOnUserGesture, true);
    window.addEventListener('keydown', requestOnUserGesture, true);

    return () => {
      document.removeEventListener('fullscreenchange', handleFullscreenChange);
      document.removeEventListener('fullscreenerror', handleFullscreenError);
      document.removeEventListener('webkitfullscreenchange', handleFullscreenChange as EventListener);
      document.removeEventListener('webkitfullscreenerror', handleFullscreenError as EventListener);
      document.removeEventListener('MSFullscreenChange', handleFullscreenChange as EventListener);
      document.removeEventListener('MSFullscreenError', handleFullscreenError as EventListener);
      window.removeEventListener('pointerdown', requestOnUserGesture, true);
      window.removeEventListener('keydown', requestOnUserGesture, true);
      fullscreenExitSince.current = null;
      fullscreenIncidentActive.current = false;
      fullscreenExitReasonRef.current = null;
      fullscreenRequestInFlightRef.current = false;
      screenShareRequestInFlightRef.current = false;
      screenShareGrantedRef.current = false;
    };
  }, [emit, violationScreenRecorder.startViolationRecording, violationScreenRecorder.stopViolationRecording]);

  useEffect(() => {
    aliveRef.current = true;

    async function start() {
      setState((current) => ({
        initialized: false,
        status: current.fullscreenExitDetected
          ? 'Fullscreen exited'
          : current.tabSwitchDetected
            ? 'Tab/window switch detected'
            : 'Requesting camera access...',
        permissionDenied: false,
        cameraActive: false,
        faceDetectorActive: false,
        objectDetectorActive: false,
        faceCount: 0,
        phoneDetected: false,
        gazeAwayDetected: false,
        gazeDirection: 'center',
        headMovementDetected: false,
        headPoseDeviationDegrees: 0,
        tabSwitchDetected: current.tabSwitchDetected,
        tabSwitchReason: current.tabSwitchReason,
        lastTabSwitchAt: current.lastTabSwitchAt,
        lastTabSwitchDurationMs: current.lastTabSwitchDurationMs,
        activeTabSwitchDurationMs: current.activeTabSwitchDurationMs,
        totalTabSwitchDurationMs: current.totalTabSwitchDurationMs,
        tabSwitchCount: current.tabSwitchCount,
        fullscreenActive: current.fullscreenActive,
        fullscreenExitDetected: current.fullscreenExitDetected,
        fullscreenExitReason: current.fullscreenExitReason,
        lastFullscreenExitAt: current.lastFullscreenExitAt,
        fullscreenSupported: current.fullscreenSupported,
        fullscreenReadyBeforeInterview: current.fullscreenReadyBeforeInterview,
        fullscreenPromptRequired: current.fullscreenPromptRequired,
        preInterviewFullscreenRequestedAt: current.preInterviewFullscreenRequestedAt,
        preInterviewFullscreenEnteredAt: current.preInterviewFullscreenEnteredAt,
        screenShareSupported: current.screenShareSupported,
        screenShareReadyBeforeInterview: current.screenShareReadyBeforeInterview,
        screenSharePromptRequired: current.screenSharePromptRequired,
        preInterviewScreenShareRequestedAt: current.preInterviewScreenShareRequestedAt,
        preInterviewScreenShareGrantedAt: current.preInterviewScreenShareGrantedAt,
        lastObservationAt: current.lastObservationAt,
      }));

      const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
      streamRef.current = stream;

      if (!aliveRef.current) {
        stream.getTracks().forEach((track) => track.stop());
        return;
      }

      const video = videoRef.current;
      if (video) {
        video.srcObject = stream;
        await video.play();
      }

      void requestExamFullscreen('camera_started');

      setState((current) => ({
        ...current,
        initialized: true,
        status: current.fullscreenExitDetected
          ? 'Fullscreen exited'
          : current.tabSwitchDetected
            ? 'Tab/window switch detected'
            : 'Detection active',
        permissionDenied: false,
        cameraActive: true,
      }));

      const vision = await FilesetResolver.forVisionTasks('https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm');
      const [faceLandmarker, objectDetector] = await Promise.all([
        FaceLandmarker.createFromOptions(vision, {
          baseOptions: { modelAssetPath: FACE_MODEL_URL },
          runningMode: 'VIDEO',
          numFaces: 4,
          minFaceDetectionConfidence: 0.55,
          minFacePresenceConfidence: 0.55,
          minTrackingConfidence: 0.55,
          outputFaceBlendshapes: true,
          outputFacialTransformationMatrixes: true,
        }),
        ObjectDetector.createFromOptions(vision, {
          baseOptions: { modelAssetPath: OBJECT_MODEL_URL },
          runningMode: 'VIDEO',
          scoreThreshold: 0.5,
          maxResults: 5,
        }),
      ]);

      faceLandmarkerRef.current = faceLandmarker;
      objectDetectorRef.current = objectDetector;

      setState((current) => ({
        ...current,
        faceDetectorActive: true,
        objectDetectorActive: true,
      }));

      const tick = () => {
        if (!aliveRef.current) return;
        const currentVideo = videoRef.current;
        const faceTask = faceLandmarkerRef.current;
        const objectTask = objectDetectorRef.current;
        const cameraIssue = getCameraHealthIssue(currentVideo, streamRef.current);
        if (cameraIssue) {
          const now = Date.now();
          if (!cameraOffSince.current) cameraOffSince.current = now;

          const durationMs = now - (cameraOffSince.current || now);
          if (durationMs > CAMERA_OFF_CONFIRM_MS) {
            cameraOffIncidentActive.current = true;
            const metadata = {
              reason: cameraIssue,
              durationMs,
              trackReadyState: getVideoTrack(streamRef.current)?.readyState,
              trackMuted: getVideoTrack(streamRef.current)?.muted ?? false,
              trackEnabled: getVideoTrack(streamRef.current)?.enabled ?? false,
            };

            if (emitWithCooldown(cameraOffAlertAt, 'CAMERA_OFF_DETECTED', 'HIGH', metadata)) {
              violationScreenRecorder.recordViolationClip('CAMERA_OFF', metadata);
            }
          }

          missingSince.current = null;
          setState((current) => ({
            ...current,
            cameraActive: false,
            faceDetectorActive: !!faceTask,
            objectDetectorActive: !!objectTask,
            faceCount: 0,
            phoneDetected: false,
            gazeAwayDetected: false,
            gazeDirection: 'center',
            headMovementDetected: false,
            headPoseDeviationDegrees: 0,
            lastObservationAt: now,
            status: 'Camera is off or not sending video',
          }));
          frameTimerRef.current = window.setTimeout(tick, LIVE_INTERVAL_MS);
          return;
        }

        if (cameraOffSince.current !== null) {
          const now = Date.now();
          const durationMs = now - cameraOffSince.current;
          if (cameraOffIncidentActive.current) {
            emit('CAMERA_RESTORED', 'MEDIUM', { durationMs });
          }
          cameraOffSince.current = null;
          cameraOffIncidentActive.current = false;
        }

        if (!currentVideo || !faceTask || !objectTask || currentVideo.readyState < 2) {
          if (!missingSince.current) missingSince.current = Date.now();
          if (Date.now() - (missingSince.current || 0) > NO_FACE_THRESHOLD_MS) {
            const metadata = { durationMs: Date.now() - (missingSince.current || 0) };
            if (emitWithCooldown(faceAlertAt, 'FACE_NOT_DETECTED', 'HIGH', metadata)) {
              violationScreenRecorder.recordViolationClip('NO_FACE', metadata);
            }
          }
          setState((current) => ({
            ...current,
            cameraActive: !!currentVideo?.srcObject,
            faceDetectorActive: !!faceTask,
            objectDetectorActive: !!objectTask,
            faceCount: 0,
            phoneDetected: false,
            headMovementDetected: false,
            headPoseDeviationDegrees: 0,
            lastObservationAt: Date.now(),
          }));
          frameTimerRef.current = window.setTimeout(tick, LIVE_INTERVAL_MS);
          return;
        }

        missingSince.current = null;
        const timestamp = performance.now();

        let faceResult: FaceLandmarkerResult | null = null;
        let objectResult: ObjectDetectorResult | null = null;

        try {
          if (faceTask && typeof (faceTask as any).detectForVideo === 'function') {
            faceResult = (faceTask as any).detectForVideo(currentVideo, timestamp);
          } else {
            console.warn('Face task not ready or detectForVideo missing');
            setState((current) => ({ ...current, status: 'Face detector unavailable' }));
          }
        } catch (error) {
          console.error('Face detect error', error);
          setState((current) => ({ ...current, status: 'Face detector unavailable' }));
        }

        try {
          if (objectTask && typeof (objectTask as any).detectForVideo === 'function') {
            objectResult = (objectTask as any).detectForVideo(currentVideo, timestamp);
          } else {
            console.warn('Object task not ready or detectForVideo missing');
            setState((current) => ({ ...current, status: 'Object detector unavailable' }));
          }
        } catch (error) {
          console.error('Object detect error', error);
          setState((current) => ({ ...current, status: 'Object detector unavailable' }));
        }

        const faceCount = faceResult?.faceLandmarks?.length || 0;
        const detectedPhone = objectResult ? isPhoneDetection(objectResult) : false;

        const headPose = faceCount === 1 ? estimateHeadPose(faceResult) : null;
        if (headPose && !headPoseBaselineRef.current) {
          headPoseBaselineRef.current = headPose;
        }
        const headPoseDeviation = headPose && headPoseBaselineRef.current ? calculateHeadPoseDeviation(headPose, headPoseBaselineRef.current) : null;
        const headMovementDetected = Boolean(headPoseDeviation?.tooMuch);

        const safeCalibration = safeCalibrationRef.current;
        const gaze = detectGazeAway(
          faceResult,
          safeCalibration.thresholdX ?? DEFAULT_GAZE_THRESHOLD_X,
          safeCalibration.thresholdY ?? DEFAULT_GAZE_THRESHOLD_Y,
          safeCalibration.neutralX ?? 0,
          safeCalibration.neutralY ?? 0,
          headPoseDeviation,
          gazeFilterRef,
          0.18, // reduced from 0.28 for better filtering of false positives
        );

        setState((current) => ({
          ...current,
          cameraActive: true,
          faceDetectorActive: true,
          objectDetectorActive: true,
          faceCount,
          phoneDetected: detectedPhone,
          gazeAwayDetected: gaze.away,
          gazeDirection: gaze.direction,
          headMovementDetected,
          headPoseDeviationDegrees: Math.round(headPoseDeviation?.magnitude ?? 0),
          lastObservationAt: Date.now(),
          status: current.fullscreenExitDetected
            ? 'Fullscreen exited'
            : current.tabSwitchDetected
              ? 'Tab/window switch detected'
              : faceCount > 1
              ? 'Multiple faces detected'
              : detectedPhone
                ? 'Phone detected'
                : headMovementDetected
                  ? `Head moved too much (${Math.round(headPoseDeviation?.magnitude ?? 0)}°)`
                  : gaze.away
                    ? `Looking away (${gaze.direction})`
                    : 'Detection active',
        }));

        if (faceCount === 0) {
          if (!missingSince.current) missingSince.current = Date.now();
          if (Date.now() - (missingSince.current || 0) > NO_FACE_THRESHOLD_MS) {
            const metadata = { durationMs: Date.now() - (missingSince.current || 0) };
            if (emitWithCooldown(faceAlertAt, 'FACE_NOT_DETECTED', 'HIGH', metadata)) {
              violationScreenRecorder.recordViolationClip('NO_FACE', metadata);
            }
          }
        } else {
          missingSince.current = null;
        }

        // require a short confirmation window to avoid spurious multi-face / phone alerts
        if (faceCount > 1) {
          if (!multiFaceSince.current) multiFaceSince.current = Date.now();
          if (Date.now() - (multiFaceSince.current || 0) > MULTI_FACE_CONFIRM_MS) {
            const metadata = { faceCount, faces: faceCount };
            if (emitWithCooldown(multiFaceAlertAt, 'MULTIPLE_FACES_DETECTED', 'HIGH', metadata)) {
              violationScreenRecorder.recordViolationClip('MULTIPLE_FACES', metadata);
            }
            multiFaceSince.current = null;
          }
        } else {
          multiFaceSince.current = null;
        }

        if (detectedPhone) {
          if (!phoneSince.current) phoneSince.current = Date.now();
          if (Date.now() - (phoneSince.current || 0) > PHONE_CONFIRM_MS) {
            const metadata = {
              detections: objectResult?.detections?.map((detection) => ({
                label: detection.categories?.[0]?.categoryName || detection.categories?.[0]?.displayName || 'object',
                score: detection.categories?.[0]?.score ?? 0,
                box: detection.boundingBox,
              })),
              topScore: objectResult ? latestDetectionScore(objectResult) : 0,
            };

            if (emitWithCooldown(phoneAlertAt, 'MOBILE_PHONE_DETECTED', 'HIGH', metadata)) {
              violationScreenRecorder.recordViolationClip('MOBILE_PHONE', metadata);
            }
            phoneSince.current = null;
          }
        } else {
          phoneSince.current = null;
        }

        if (gaze.away) {
          if (!gazeSince.current) gazeSince.current = Date.now();
          if (Date.now() - (gazeSince.current || 0) > GAZE_CONFIRM_MS) {
            const metadata = {
              direction: gaze.direction,
              confidence: gaze.confidence,
              source: gaze.source,
              calibrationTrusted: safeCalibration.trusted,
              calibrationReason: safeCalibration.reason,
              thresholdX: safeCalibration.thresholdX,
              thresholdY: safeCalibration.thresholdY,
              headPoseCompensated: gaze.headPoseCompensated,
              uncompensatedX: gaze.uncompensatedX,
              uncompensatedY: gaze.uncompensatedY,
              compensatedX: gaze.compensatedX,
              compensatedY: gaze.compensatedY,
              yawDelta: gaze.yawDelta,
              pitchDelta: gaze.pitchDelta,
            };

            if (emitWithCooldown(gazeAlertAt, 'GAZE_AWAY_DETECTED', 'MEDIUM', metadata)) {
              violationScreenRecorder.recordViolationClip('GAZE_AWAY', metadata);
            }
            gazeSince.current = null;
          }
        } else {
          gazeSince.current = null;
        }

        if (headMovementDetected && headPoseDeviation && headPose) {
          if (!headPoseSince.current) headPoseSince.current = Date.now();
          if (Date.now() - (headPoseSince.current || 0) > HEAD_POSE_CONFIRM_MS) {
            const metadata = {
              deviationDegrees: Math.round(headPoseDeviation.magnitude),
              maxAxisDegrees: Math.round(headPoseDeviation.maxAxis),
              yawDelta: Math.round(headPoseDeviation.yaw),
              pitchDelta: Math.round(headPoseDeviation.pitch),
              rollDelta: Math.round(headPoseDeviation.roll),
              source: headPose.source,
            };

            if (emitWithCooldown(headPoseAlertAt, 'HEAD_MOVEMENT_DETECTED', 'MEDIUM', metadata)) {
              violationScreenRecorder.recordViolationClip('HEAD_MOVEMENT', metadata);
            }
            headPoseSince.current = null;
          }
        } else {
          headPoseSince.current = null;
        }

        frameTimerRef.current = window.setTimeout(tick, LIVE_INTERVAL_MS);
      };

      frameTimerRef.current = window.setTimeout(tick, LIVE_INTERVAL_MS);
    }

    start().catch((error) => {
      setState({
        initialized: false,
        status: 'Camera permission denied',
        permissionDenied: true,
        cameraActive: false,
        faceDetectorActive: false,
        objectDetectorActive: false,
        faceCount: 0,
        phoneDetected: false,
        gazeAwayDetected: false,
        gazeDirection: 'center',
        headMovementDetected: false,
        headPoseDeviationDegrees: 0,
        tabSwitchDetected: false,
        tabSwitchReason: null,
        lastTabSwitchAt: null,
        lastTabSwitchDurationMs: null,
        activeTabSwitchDurationMs: 0,
        totalTabSwitchDurationMs: 0,
        tabSwitchCount: 0,
        fullscreenActive: Boolean(getFullscreenElement()),
        fullscreenExitDetected: false,
        fullscreenExitReason: null,
        lastFullscreenExitAt: null,
        fullscreenSupported: typeof document === 'undefined' ? true : isFullscreenSupported(),
        fullscreenReadyBeforeInterview: Boolean(getFullscreenElement()),
        fullscreenPromptRequired: typeof document === 'undefined' ? false : !Boolean(getFullscreenElement()) && isFullscreenSupported(),
        preInterviewFullscreenRequestedAt: null,
        preInterviewFullscreenEnteredAt: Boolean(getFullscreenElement()) ? Date.now() : null,
        screenShareSupported: typeof navigator === 'undefined' ? true : isScreenShareRecordingSupported(),
        screenShareReadyBeforeInterview: violationScreenRecorder.hasScreenSharePermission,
        screenSharePromptRequired: !violationScreenRecorder.hasScreenSharePermission,
        preInterviewScreenShareRequestedAt: null,
        preInterviewScreenShareGrantedAt: violationScreenRecorder.hasScreenSharePermission ? Date.now() : null,
        lastObservationAt: null,
      });
      const message = error instanceof Error ? error.message : 'getUserMedia failed';
      const metadata = { reason: message, permissionDenied: true };
      emit('CAMERA_PERMISSION_DENIED', 'HIGH', { message });
      emit('CAMERA_OFF_DETECTED', 'HIGH', metadata);
      violationScreenRecorder.recordViolationClip('CAMERA_OFF', metadata);
    });

    return () => {
      aliveRef.current = false;
      if (frameTimerRef.current) {
        window.clearTimeout(frameTimerRef.current);
      }
      streamRef.current?.getTracks().forEach((track) => track.stop());
      cameraOffSince.current = null;
      cameraOffIncidentActive.current = false;
      tabSwitchSince.current = null;
      tabSwitchIncidentActive.current = false;
      tabSwitchReasonRef.current = null;
      clearTabSwitchDurationTimer();
      fullscreenExitSince.current = null;
      fullscreenIncidentActive.current = false;
      fullscreenExitReasonRef.current = null;
      fullscreenRequestInFlightRef.current = false;
      screenShareRequestInFlightRef.current = false;
      screenShareGrantedRef.current = false;
      faceLandmarkerRef.current?.close();
      objectDetectorRef.current?.close();
      faceLandmarkerRef.current = null;
      objectDetectorRef.current = null;
      streamRef.current = null;
    };
  }, [emit, violationScreenRecorder.recordViolationClip]);

  return {
    videoRef,
    events,
    emit,
    state,
    // Backward-compatible: existing UI buttons that call requestFullscreen() now also request screen sharing first.
    requestFullscreen: enterFullscreenBeforeInterview,
    requestExamFullscreen,
    enterFullscreenBeforeInterview,
    prepareInterviewStart,
    startInterviewSetup: prepareInterviewStart,
    requestProctoringPermissions: prepareInterviewStart,
    requestRequiredPermissions: prepareInterviewStart,
    requestScreenShareBeforeInterview,
    requestScreenShare: violationScreenRecorder.requestScreenShare,
    stopScreenShare: violationScreenRecorder.stopScreenShare,
    hasScreenSharePermission: violationScreenRecorder.hasScreenSharePermission,
    isRecordingViolation: violationScreenRecorder.isRecordingViolation,
    screenShareError: violationScreenRecorder.screenShareError,
    violationRecordings,
    canStartInterview:
      state.fullscreenActive &&
      state.fullscreenReadyBeforeInterview &&
      !state.fullscreenExitDetected &&
      state.screenShareReadyBeforeInterview &&
      violationScreenRecorder.hasScreenSharePermission,
    preInterviewFullscreenRequired: state.fullscreenSupported && !state.fullscreenActive,
    preInterviewScreenShareRequired: state.screenShareSupported && !violationScreenRecorder.hasScreenSharePermission,
  };
}
