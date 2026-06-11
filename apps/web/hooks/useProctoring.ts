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
  | 'COPY_SHORTCUT'
  | 'PASTE_SHORTCUT'
  | 'CAMERA_OFF'
  | 'MULTIPLE_FACES'
  | 'NO_FACE'
  | 'MOBILE_PHONE'
  | 'FOREIGN_OBJECT'
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
  phoneProximity: PhoneProximity;
  phoneOrientation: PhoneOrientation;
  foreignObjectDetected: boolean;
  foreignObjectLabel: string;
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
  microphoneSupported: boolean;
  microphoneReadyBeforeInterview: boolean;
  microphonePromptRequired: boolean;
  preInterviewMicrophoneRequestedAt: number | null;
  preInterviewMicrophoneGrantedAt: number | null;
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
const GAZE_CONFIRM_MS = 1000; // faster gaze confirmation
const HEAD_POSE_CONFIRM_MS = 1200;
const CAMERA_OFF_CONFIRM_MS = 2000;
const TAB_SWITCH_ALERT_COOLDOWN_MS = 500;
const TAB_SWITCH_CONFIRM_MS = 800;
const FULLSCREEN_ALERT_COOLDOWN_MS = 1000;
const CLIPBOARD_SHORTCUT_ALERT_COOLDOWN_MS = 1000;
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
  return typeof navigator.mediaDevices?.getDisplayMedia === 'function' && typeof MediaRecorder !== 'undefined';
}

function isMicrophoneCaptureSupported() {
  if (typeof navigator === 'undefined') return true;
  return typeof navigator.mediaDevices?.getUserMedia === 'function';
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

// ─────────────────────────────────────────────────────────────────────────────
// Enhanced phone-detection — types & constants
// ─────────────────────────────────────────────────────────────────────────────

export type PhoneProximity = 'onscreen' | 'close' | 'mid' | 'far';

export type PhoneOrientation = 'portrait' | 'landscape' | 'diagonal' | 'unknown';

export type PhoneSource = 'ml_label' | 'aspect_heuristic' | 'edge_touch' | 'combined';

export type PhoneAnalysis = {
  detected:          boolean;
  confidence:        number;
  rollingConfidence: number;
  proximity:         PhoneProximity;
  orientation:       PhoneOrientation;
  source:            PhoneSource;
  confirmMs:         number;
  detections: Array<{
    label:       string;
    score:       number;
    areaRatio:   number;
    aspectRatio: number;
  }>;
};

type PhoneState = {
  scoreBuffer:    number[];
  confirmedSince: number | null;
};

// ─────────────────────────────────────────────────────────────────────────────
// ← NEW: Foreign-object detection types & constants
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Result of the foreign-object edge-peek analysis for a single frame.
 *
 * A "foreign object" is any non-whitelisted item that:
 *   (a) partially enters the frame from any edge  (edge-peek), OR
 *   (b) occupies the lower-third of the frame between the candidate's face
 *       region and the camera (occlusion heuristic).
 *
 * Allowed objects that are explicitly NOT flagged:
 *   • body parts  (hand, arm, shoulder, finger, wrist, elbow, neck, face, head)
 *   • writing instruments (pen, pencil, marker, stylus)
 *
 * Everything else — phones, tablets, remotes, books, notepads, cups, bottles,
 * keys, wallets, earphones, glasses cases, etc. — should not appear between
 * the candidate and the camera during a proctored session.
 */
export type ForeignObjectAnalysis = {
  detected:       boolean;
  label:          string;
  confidence:     number;
  /** 'edge_peek'   — box clips frame boundary but covers < PEEK_MAX_AREA of frame */
  /** 'occluding'   — box sits in the lower third of the frame at significant size */
  /** 'partial_edge'— tiny sliver (< PEEK_MIN_AREA) sustained over multiple frames */
  trigger:        'edge_peek' | 'occluding' | 'partial_edge' | 'none';
  areaRatio:      number;
  edgeSides:      Array<'top' | 'bottom' | 'left' | 'right'>;
};

type ForeignObjectState = {
  /** Rolling frame buffer of per-frame foreign-object scores (0 or raw score). */
  scoreBuffer:    number[];
  confirmedSince: number | null;
  /** Label of the most recently sustained detection for metadata. */
  lastLabel:      string;
};

// ── Foreign-object tuning constants ──────────────────────────────────────────

/**
 * Objects that are explicitly ALLOWED between the candidate and the camera.
 * Body parts are listed exhaustively because EfficientDet may predict them
 * on clothing, reflected surfaces, or close-up camera views.
 */
const FOREIGN_OBJECT_ALLOWLIST: string[] = [
  // body parts
  'person', 'people', 'human', 'face', 'man', 'woman', 'boy', 'girl',
  'head', 'body', 'torso', 'hand', 'arm', 'finger', 'wrist', 'elbow',
  'shoulder', 'neck', 'chest', 'ear', 'eye', 'nose', 'mouth', 'lip',
  // writing instruments
  'pen', 'pencil', 'marker', 'stylus', 'crayon', 'chalk',
  // headphones / earphones are common during interviews — allow them
  'headphone', 'headset', 'earphone', 'earpiece',
  // glasses on the candidate's face are ok
  'glasses', 'sunglasses',
];

/**
 * Minimum fraction of the frame area a box must cover to trigger the
 * edge-peek strategy.  Below this the box is likely a detection artefact.
 * 0.003 ≈ a ~50×40 px box in a 640×480 frame — a sliver of a phone edge.
 */
const PEEK_MIN_AREA = 0.003;

/**
 * Maximum fraction for the edge-peek strategy.  Above this the object is
 * large enough to be caught by the phone-detection logic or the occlusion
 * heuristic — no need to double-count.
 */
const PEEK_MAX_AREA = 0.20;

/**
 * Fraction of frame width/height considered "touching" an edge.
 * Slightly wider than EDGE_MARGIN used in phone detection so that objects
 * peeking in from outside the frame are caught even when their centroid is
 * still inside by a small margin.
 */
const FOREIGN_EDGE_MARGIN = 0.06;

/**
 * Objects in the lower third of the frame (y > OCCLUDE_Y_THRESHOLD) that
 * cover more than OCCLUDE_MIN_AREA of the frame are treated as occluding
 * objects between the candidate and the camera.
 */
const OCCLUDE_Y_THRESHOLD = 0.60;   // normalised y of top edge of bounding box
const OCCLUDE_MIN_AREA    = 0.06;   // 6 % of frame area

/** Minimum ML score to consider a detection for foreign-object analysis. */
const FOREIGN_MIN_SCORE   = 0.25;

/** Rolling window length (frames) for foreign-object sustained detection. */
const FOREIGN_ROLLING_WINDOW = 12;   // ≈ 2.4 s at 200 ms/frame

/**
 * A very small edge-peeking object (PEEK_MIN_AREA … 0.02) requires more
 * frames of sustained presence before an alert fires.
 */
const FOREIGN_CONFIRM_MS_SMALL  = 2000;
const FOREIGN_CONFIRM_MS_NORMAL = 1000;

// ─────────────────────────────────────────────────────────────────────────────
// Phone-detection tuning constants (unchanged)
// ─────────────────────────────────────────────────────────────────────────────

const PHONE_SCORE_THRESHOLD = 0.2;
const CONFIRM_MS_HIGH       = 500;
const CONFIRM_MS_MED        = 1200;
const CONFIRM_MS_LOW        = 2500;

const AREA_ONSCREEN = 0.40;
const AREA_CLOSE    = 0.15;
const AREA_MID      = 0.04;

const PORTRAIT_AR  = { min: 0.38, max: 0.62 };
const LANDSCAPE_AR = { min: 1.70, max: 2.90 };
const DIAGONAL_AR  = { min: 0.63, max: 1.69 };

const MIN_HEURISTIC_AREA = 0.006;
const ROLLING_WINDOW     = 20;
const COMBINED_BONUS     = 0.15;

const EDGE_MARGIN          = 0.04;
const EDGE_TOUCH_BOOST     = 0.18;
const EDGE_STANDALONE_AREA = 0.25;

const PHONE_KEYWORDS = [
  'cell phone', 'mobile phone', 'smartphone', 'cellphone',
  'iphone', 'android', 'phone',
];

const ADJACENT_LABELS = [
  'remote', 'remote control', 'tablet', 'book', 'calculator', 'mouse',
];

const HEURISTIC_BLOCKLIST = [
  'person', 'people', 'human', 'face', 'man', 'woman', 'boy', 'girl',
  'head', 'body', 'torso', 'hand', 'arm',
  'door', 'window', 'picture frame', 'painting', 'poster',
  'screen', 'monitor', 'tv', 'television', 'laptop',
];

const PROXIMITY_RANK: Record<PhoneProximity, number> = {
  far: 0, mid: 1, close: 2, onscreen: 3,
};

// ─────────────────────────────────────────────────────────────────────────────
// Proctoring score types
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Raw per-frame counters accumulated throughout the interview session.
 * Stored in a ref so updates never trigger re-renders.
 */
type ProctoringData = {
  gaze: {
    framesLookingAway: number;
    totalFrames: number;
  };
  face: {
    framesNormal: number;        // exactly 1 face visible
    framesNoFace: number;        // 0 faces visible
    framesMultipleFaces: number; // 2+ faces visible
    multipleFaceEvents: number;  // distinct transitions from <=1 -> >1 faces
    totalFrames: number;
  };
  tabSwitch: {
    count: number;               // number of distinct tab/window-switch events
  };
  phone: {
    detectedSeconds: number;       // cumulative seconds phone was visible
    maxConsecutiveSeconds: number; // longest single detection streak
    consecutiveSeconds: number;    // current streak, resets when not detected
    events: number;                // number of distinct detection streaks
    totalSeconds: number;          // total interview duration in seconds
    wasDetected: boolean;          // whether phone was detected last tick
  };
  headPose: {
    framesNormal: number;
    framesDeviated: number;
    totalFrames: number;
  };
  sessionControl: {
    fullscreenExitCount: number;
    screenShareStoppedCount: number;
    cameraOffSeconds: number;
    cameraOffMaxConsecutiveSeconds: number;
    cameraOffConsecutiveSeconds: number;
    cameraOffIncidentCount: number;
    screenShareStoppedAndNotRestored: boolean;
  };
};

/**
 * Final computed proctoring score returned by computeProctoringScore().
 * Pass this to your report-card component after the interview ends.
 */
export type ProctoringScore = {
  final: number; // 0-100 weighted composite
  gazeScore: number;
  faceScore: number;
  tabScore: number;
  phoneScore: number;
  headPoseScore: number;
  sessionControlScore: number;
  band: 'Clean' | 'Minor Concerns' | 'Suspicious' | 'High Risk';
  flags: string[];
};

function createInitialProctoringData(): ProctoringData {
  return {
    gaze: { framesLookingAway: 0, totalFrames: 0 },
    face: {
      framesNormal: 0,
      framesNoFace: 0,
      framesMultipleFaces: 0,
      multipleFaceEvents: 0,
      totalFrames: 0,
    },
    tabSwitch: { count: 0 },
    phone: {
      detectedSeconds: 0,
      maxConsecutiveSeconds: 0,
      consecutiveSeconds: 0,
      events: 0,
      totalSeconds: 0,
      wasDetected: false,
    },
    headPose: { framesNormal: 0, framesDeviated: 0, totalFrames: 0 },
    sessionControl: {
      fullscreenExitCount: 0,
      screenShareStoppedCount: 0,
      cameraOffSeconds: 0,
      cameraOffMaxConsecutiveSeconds: 0,
      cameraOffConsecutiveSeconds: 0,
      cameraOffIncidentCount: 0,
      screenShareStoppedAndNotRestored: false,
    },
  };
}

// ─────────────────────────────────────────────────────────────────────────────
// Phone-detection helpers (unchanged)
// ─────────────────────────────────────────────────────────────────────────────

function getPhoneCategoryName(category: any): string {
  return (
    (category.categoryName || category.displayName || category.label || '')
      .toString()
      .toLowerCase()
  );
}

function getProximity(areaRatio: number): PhoneProximity {
  if (areaRatio >= AREA_ONSCREEN) return 'onscreen';
  if (areaRatio >= AREA_CLOSE)    return 'close';
  if (areaRatio >= AREA_MID)      return 'mid';
  return 'far';
}

function getPhoneOrientation(ar: number): PhoneOrientation {
  if (ar >= PORTRAIT_AR.min  && ar <= PORTRAIT_AR.max)  return 'portrait';
  if (ar >= LANDSCAPE_AR.min && ar <= LANDSCAPE_AR.max) return 'landscape';
  if (ar >= DIAGONAL_AR.min  && ar <= DIAGONAL_AR.max)  return 'diagonal';
  return 'unknown';
}

function rollingScore(buffer: number[]): number {
  if (!buffer.length) return 0;
  let wSum = 0, wTotal = 0;
  for (let i = 0; i < buffer.length; i++) {
    const w = i + 1;
    wSum   += buffer[i] * w;
    wTotal += w;
  }
  return wSum / wTotal;
}

function createPhoneState(): PhoneState {
  return { scoreBuffer: [], confirmedSince: null };
}

function getPhoneConfirmMs(confidence: number, proximity: PhoneProximity): number {
  if (confidence >= 0.55 || proximity === 'close' || proximity === 'onscreen') return CONFIRM_MS_HIGH;
  if (confidence >= 0.35 || proximity === 'mid') return CONFIRM_MS_MED;
  return CONFIRM_MS_LOW;
}

function isEdgeTouching(
  box: { originX: number; originY: number; width: number; height: number },
  videoWidth: number,
  videoHeight: number,
  margin = EDGE_MARGIN,
): boolean {
  const x1 = box.originX / videoWidth;
  const y1 = box.originY / videoHeight;
  const x2 = (box.originX + box.width)  / videoWidth;
  const y2 = (box.originY + box.height) / videoHeight;
  return x1 <= margin || y1 <= margin || x2 >= (1 - margin) || y2 >= (1 - margin);
}

function analyzePhoneDetection(
  result: ObjectDetectorResult,
  videoWidth: number,
  videoHeight: number,
  state: PhoneState,
): PhoneAnalysis {
  const frameArea = (videoWidth * videoHeight) || 1;

  let mlScore        = 0;
  let heuristicScore = 0;
  let edgeScore      = 0;
  let bestProximity:   PhoneProximity   = 'far';
  let bestOrientation: PhoneOrientation = 'unknown';
  const detections: PhoneAnalysis['detections'] = [];

  for (const detection of result.detections ?? []) {
    const box = detection.boundingBox;
    if (!box?.width || !box?.height) continue;

    const areaRatio   = (box.width * box.height) / frameArea;
    const aspectRatio = box.width / box.height;
    const orientation = getPhoneOrientation(aspectRatio);
    const proximity   = getProximity(areaRatio);
    const edgeTouching = isEdgeTouching(box, videoWidth, videoHeight);

    // Strategy 1: ML label
    for (const category of detection.categories ?? []) {
      const name  = getPhoneCategoryName(category);
      let   score = typeof category.score === 'number' ? category.score : 0;
      if (PHONE_KEYWORDS.some((kw) => name.includes(kw)) && score >= PHONE_SCORE_THRESHOLD) {
        if (edgeTouching) score = Math.min(score + EDGE_TOUCH_BOOST, 1.0);
        if (score > mlScore) {
          mlScore = score;
          if (PROXIMITY_RANK[proximity] > PROXIMITY_RANK[bestProximity]) {
            bestProximity   = proximity;
            bestOrientation = orientation !== 'unknown' ? orientation : bestOrientation;
          }
        }
        detections.push({ label: name, score, areaRatio, aspectRatio });
      }
    }

    // Strategy 2: Aspect-ratio heuristic
    if (areaRatio >= MIN_HEURISTIC_AREA && orientation !== 'unknown') {
      const alreadyScoredByML = detections.some(
        (d) => d.areaRatio === areaRatio && d.aspectRatio === aspectRatio && !d.label.startsWith('~'),
      );
      if (!alreadyScoredByML) {
        let bestCategoryScore = 0;
        let bestLabel         = 'object';
        for (const category of detection.categories ?? []) {
          const name  = getPhoneCategoryName(category);
          const score = typeof category.score === 'number' ? category.score : 0;
          if (score > bestCategoryScore) { bestCategoryScore = score; bestLabel = name; }
        }
        const isBlocked  = HEURISTIC_BLOCKLIST.some((kw) => bestLabel.includes(kw));
        if (isBlocked) continue;
        const isAdjacent = ADJACENT_LABELS.some((kw) => bestLabel.includes(kw));
        let computed = 0;
        if (orientation === 'portrait') {
          const sizeFactor    = Math.min(areaRatio / AREA_MID, 2.5);
          const adjacentBoost = isAdjacent ? 0.12 : 0;
          computed = 0.46 * sizeFactor + adjacentBoost;
        } else if (orientation === 'landscape') {
          if (!isAdjacent) continue;
          const sizeFactor = Math.min(areaRatio / AREA_MID, 2.5);
          computed = 0.36 * sizeFactor + 0.12;
        } else if (orientation === 'diagonal') {
          if (!isAdjacent && !edgeTouching) continue;
          if (areaRatio < MIN_HEURISTIC_AREA * 2.0) continue;
          const sizeFactor    = Math.min(areaRatio / AREA_MID, 2.5);
          const adjacentBoost = isAdjacent ? 0.10 : 0;
          computed = 0.32 * sizeFactor + adjacentBoost;
        }
        if (edgeTouching) computed += EDGE_TOUCH_BOOST;
        const capped = Math.min(computed, 0.72);
        if (capped >= PHONE_SCORE_THRESHOLD) {
          if (capped > heuristicScore) {
            heuristicScore = capped;
            if (PROXIMITY_RANK[proximity] > PROXIMITY_RANK[bestProximity]) {
              bestProximity = proximity; bestOrientation = orientation;
            }
          }
          detections.push({ label: `~${bestLabel}`, score: capped, areaRatio, aspectRatio });
        }
      }
    }

    // Strategy 4: Edge-touch standalone
    const alreadyScored = detections.some(
      (d) => d.areaRatio === areaRatio && d.aspectRatio === aspectRatio,
    );
    if (
      edgeTouching && !alreadyScored &&
      areaRatio >= EDGE_STANDALONE_AREA &&
      (orientation === 'portrait' || orientation === 'diagonal')
    ) {
      let bestCategoryScore = 0;
      let bestLabel         = 'object';
      for (const category of detection.categories ?? []) {
        const name  = getPhoneCategoryName(category);
        const score = typeof category.score === 'number' ? category.score : 0;
        if (score > bestCategoryScore) { bestCategoryScore = score; bestLabel = name; }
      }
      const isBlocked = HEURISTIC_BLOCKLIST.some((kw) => bestLabel.includes(kw));
      if (!isBlocked) {
        const standalone = 0.48;
        if (standalone > edgeScore) {
          edgeScore = standalone;
          if (PROXIMITY_RANK['onscreen'] > PROXIMITY_RANK[bestProximity]) {
            bestProximity = 'onscreen'; bestOrientation = orientation;
          }
        }
        detections.push({ label: `~edge:${bestLabel}`, score: standalone, areaRatio, aspectRatio });
      }
    }
  }

  // Strategy 3: Combined bonus
  const strategyCount = [mlScore, heuristicScore, edgeScore].filter((s) => s > 0).length;
  const rawConfidence =
    strategyCount >= 2
      ? Math.min(Math.max(mlScore, heuristicScore, edgeScore) + COMBINED_BONUS, 1.0)
      : Math.max(mlScore, heuristicScore, edgeScore);

  state.scoreBuffer.push(rawConfidence);
  if (state.scoreBuffer.length > ROLLING_WINDOW) state.scoreBuffer.shift();
  const rolling = rollingScore(state.scoreBuffer);

  const source: PhoneSource =
    strategyCount >= 2                       ? 'combined'
    : mlScore > 0                            ? 'ml_label'
    : edgeScore > 0 && heuristicScore === 0  ? 'edge_touch'
    :                                          'aspect_heuristic';

  return {
    detected:          rolling >= PHONE_SCORE_THRESHOLD,
    confidence:        rawConfidence,
    rollingConfidence: rolling,
    proximity:         bestProximity,
    orientation:       bestOrientation,
    source,
    confirmMs:         getPhoneConfirmMs(rolling, bestProximity),
    detections,
  };
}

function consumePhoneAlert(phone: PhoneAnalysis, state: PhoneState): boolean {
  if (!phone.detected) { state.confirmedSince = null; return false; }
  if (state.confirmedSince === null) state.confirmedSince = Date.now();
  const held = Date.now() - state.confirmedSince;
  if (held >= phone.confirmMs) { state.confirmedSince = null; return true; }
  return false;
}

// ─────────────────────────────────────────────────────────────────────────────
// ← NEW: Foreign-object detection helpers
// ─────────────────────────────────────────────────────────────────────────────

function createForeignObjectState(): ForeignObjectState {
  return { scoreBuffer: [], confirmedSince: null, lastLabel: '' };
}

/**
 * Returns true when the label is explicitly allowed between the candidate
 * and the camera (body parts and writing instruments).
 */
function isForeignObjectAllowed(label: string): boolean {
  return FOREIGN_OBJECT_ALLOWLIST.some((allowed) => label.includes(allowed));
}

/**
 * Returns which frame edges (top / bottom / left / right) the bounding box
 * clips, using a wider margin than the phone-detection edge check so that
 * objects peeking in from outside the frame are reliably caught.
 *
 * Coordinates are normalised to [0, 1].
 */
function getEdgeSides(
  box: { originX: number; originY: number; width: number; height: number },
  videoWidth: number,
  videoHeight: number,
): Array<'top' | 'bottom' | 'left' | 'right'> {
  const x1 = box.originX / videoWidth;
  const y1 = box.originY / videoHeight;
  const x2 = (box.originX + box.width)  / videoWidth;
  const y2 = (box.originY + box.height) / videoHeight;
  const m  = FOREIGN_EDGE_MARGIN;
  const sides: Array<'top' | 'bottom' | 'left' | 'right'> = [];
  if (x1 <= m)       sides.push('left');
  if (x2 >= 1 - m)   sides.push('right');
  if (y1 <= m)       sides.push('top');
  if (y2 >= 1 - m)   sides.push('bottom');
  return sides;
}

/**
 * Per-frame foreign-object analysis.
 *
 * Three triggers, evaluated in priority order:
 *
 *   1. partial_edge — a very small box (< PEEK_MIN_AREA × 3) that clips an edge.
 *                    Requires a longer sustained presence (handled by
 *                    FOREIGN_CONFIRM_MS_SMALL) to avoid noise from detection
 *                    artefacts at frame boundaries.
 *
 *   2. edge_peek   — a non-whitelisted object's box clips ≥ 1 frame edge AND
 *                    covers PEEK_MIN_AREA × 3 … PEEK_MAX_AREA of the frame.
 *                    This is the primary scenario the user described:
 *                    "phone edge barely visible at the side of the screen."
 *
 *   3. occluding   — a non-whitelisted object sits in the lower 40 % of the
 *                    frame (below OCCLUDE_Y_THRESHOLD) and covers ≥ OCCLUDE_MIN_AREA.
 *                    This catches objects placed on a desk between the candidate
 *                    and the camera that are not fully peeking from an edge.
 *
 * The rolling average over FOREIGN_ROLLING_WINDOW frames smooths out single
 * missed frames without delaying the confirmation clock significantly.
 */
function analyzeForeignObject(
  result: ObjectDetectorResult,
  videoWidth: number,
  videoHeight: number,
  state: ForeignObjectState,
): ForeignObjectAnalysis {
  const frameArea = (videoWidth * videoHeight) || 1;

  let bestScore:      number = 0;
  let bestLabel:      string = '';
  let bestTrigger:    ForeignObjectAnalysis['trigger'] = 'none';
  let bestAreaRatio:  number = 0;
  let bestEdgeSides:  Array<'top' | 'bottom' | 'left' | 'right'> = [];

  for (const detection of result.detections ?? []) {
    const box = detection.boundingBox;
    if (!box?.width || !box?.height) continue;

    const areaRatio = (box.width * box.height) / frameArea;

    // Resolve the best label and its score for this detection.
    let detLabel = 'object';
    let detScore = 0;
    for (const category of detection.categories ?? []) {
      const name  = getPhoneCategoryName(category);
      const score = typeof category.score === 'number' ? category.score : 0;
      if (score > detScore) { detScore = score; detLabel = name; }
    }

    // Skip detections with very low ML confidence — likely noise.
    if (detScore < FOREIGN_MIN_SCORE) continue;

    // Skip explicitly allowed objects.
    if (isForeignObjectAllowed(detLabel)) continue;

    const edgeSides = getEdgeSides(box, videoWidth, videoHeight);
    const clipsEdge = edgeSides.length > 0;

    // ── Trigger 1: partial_edge ───────────────────────────────────────────
    // Tiny edge-clipping boxes are treated conservatively and require a longer
    // sustained confirmation window through consumeForeignObjectAlert().
    if (clipsEdge && areaRatio >= PEEK_MIN_AREA && areaRatio < PEEK_MIN_AREA * 3) {
      const computed = Math.min(detScore * 0.75, 1.0);
      if (computed > bestScore) {
        bestScore     = computed;
        bestLabel     = detLabel;
        bestTrigger   = 'partial_edge';
        bestAreaRatio = areaRatio;
        bestEdgeSides = edgeSides;
      }
      continue;
    }

    // ── Trigger 2: edge_peek ──────────────────────────────────────────────
    if (clipsEdge && areaRatio >= PEEK_MIN_AREA * 3 && areaRatio < PEEK_MAX_AREA) {
      // The box partially enters from outside the frame — a "peeking" object.
      // Weight the score by how much of the object is outside the frame:
      // a box that is 90 % outside (very thin sliver) still gets a meaningful
      // score because the candidate is deliberately hiding the object.
      const edgeBoost = edgeSides.length >= 2 ? 0.10 : 0; // corner peek
      const computed  = Math.min(detScore + edgeBoost, 1.0);

      if (computed > bestScore) {
        bestScore     = computed;
        bestLabel     = detLabel;
        bestTrigger   = 'edge_peek';
        bestAreaRatio = areaRatio;
        bestEdgeSides = edgeSides;
      }
      continue; // do not fall through to lower-priority triggers
    }

    // ── Trigger 3: occluding ─────────────────────────────────────────────
    // Normalise the top-edge y coordinate of the box.
    const normY1 = box.originY / videoHeight;
    if (normY1 >= OCCLUDE_Y_THRESHOLD && areaRatio >= OCCLUDE_MIN_AREA) {
      // Object sits low in the frame (near the desk surface) at significant size.
      const computed = Math.min(detScore * 0.9, 1.0); // slight discount vs edge peek
      if (computed > bestScore) {
        bestScore     = computed;
        bestLabel     = detLabel;
        bestTrigger   = 'occluding';
        bestAreaRatio = areaRatio;
        bestEdgeSides = edgeSides;
      }
      continue;
    }
  }

  // ── Rolling average ───────────────────────────────────────────────────────
  state.scoreBuffer.push(bestScore);
  if (state.scoreBuffer.length > FOREIGN_ROLLING_WINDOW) state.scoreBuffer.shift();
  const rolling = rollingScore(state.scoreBuffer);

  if (bestScore > 0) state.lastLabel = bestLabel;

  return {
    detected:   rolling >= FOREIGN_MIN_SCORE,
    label:      bestScore > 0 ? bestLabel : state.lastLabel,
    confidence: rolling,
    trigger:    bestScore > 0 ? bestTrigger : 'none',
    areaRatio:  bestAreaRatio,
    edgeSides:  bestEdgeSides,
  };
}

/**
 * Returns true when the foreign-object rolling signal has been sustained
 * long enough to warrant an alert.  The confirmation window is longer for
 * very small partial-edge detections to suppress artefact noise.
 */
function consumeForeignObjectAlert(
  fo: ForeignObjectAnalysis,
  state: ForeignObjectState,
): boolean {
  if (!fo.detected) { state.confirmedSince = null; return false; }
  if (state.confirmedSince === null) state.confirmedSince = Date.now();
  const confirmMs =
    fo.trigger === 'partial_edge' || fo.areaRatio < PEEK_MIN_AREA * 3
      ? FOREIGN_CONFIRM_MS_SMALL
      : FOREIGN_CONFIRM_MS_NORMAL;
  const held = Date.now() - state.confirmedSince;
  if (held >= confirmMs) { state.confirmedSince = null; return true; }
  return false;
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
  const [sessionStarted, setSessionStarted] = useState(false);
  const [state, setState] = useState<DetectionState>({
    initialized: false,
    status: 'Initializing camera...',
    permissionDenied: false,
    cameraActive: false,
    faceDetectorActive: false,
    objectDetectorActive: false,
    faceCount: 0,
    phoneDetected: false,
    phoneProximity: 'far',
    phoneOrientation: 'unknown',
    foreignObjectDetected: false,
    foreignObjectLabel: '',
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
    microphoneSupported: typeof navigator === 'undefined' ? true : isMicrophoneCaptureSupported(),
    microphoneReadyBeforeInterview: false,
    microphonePromptRequired: true,
    preInterviewMicrophoneRequestedAt: null,
    preInterviewMicrophoneGrantedAt: null,
    screenShareSupported: typeof navigator === 'undefined' ? true : isScreenShareRecordingSupported(),
    screenShareReadyBeforeInterview: false,
    screenSharePromptRequired: true,
    preInterviewScreenShareRequestedAt: null,
    preInterviewScreenShareGrantedAt: null,
    lastObservationAt: null,
  });
  const [violationRecordings, setViolationRecordings] = useState<ViolationRecordingClip[]>([]);
  const [finalProctoringScore, setFinalProctoringScore] = useState<ProctoringScore | null>(null);
  const [proctoringSessionEnded, setProctoringSessionEnded] = useState(false);
  const violationRecordingErrorAlertAt = useRef<number>(0);
  const missingSince = useRef<number | null>(null);
  const faceAlertAt = useRef<number>(0);
  const phoneAlertAt = useRef<number>(0);
  const foreignObjectAlertAt = useRef<number>(0);
  const gazeAlertAt = useRef<number>(0);
  const headPoseAlertAt = useRef<number>(0);
  const cameraOffAlertAt = useRef<number>(0);
  const tabSwitchAlertAt = useRef<number>(0);
  const copyShortcutAlertAt = useRef<number>(0);
  const pasteShortcutAlertAt = useRef<number>(0);
  const multiFaceAlertAt = useRef<number>(0);
  const multiFaceSince = useRef<number | null>(null);
  const gazeSince = useRef<number | null>(null);
  const headPoseSince = useRef<number | null>(null);
  const cameraOffSince = useRef<number | null>(null);
  const cameraOffIncidentActive = useRef(false);
  const cameraOffConfirmTimerRef = useRef<number | null>(null);
  const tabSwitchSince = useRef<number | null>(null);
  const tabSwitchIncidentActive = useRef(false);
  const tabSwitchReasonRef = useRef<string | null>(null);
  const tabSwitchDurationTimerRef = useRef<number | null>(null);
  const tabSwitchConfirmTimerRef = useRef<number | null>(null);
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
  const phoneStateRef = useRef<PhoneState>(createPhoneState());
  const foreignObjectStateRef = useRef<ForeignObjectState>(createForeignObjectState());
  const headPoseBaselineRef = useRef<HeadPose | null>(null);
  const proctoringDataRef = useRef<ProctoringData>(createInitialProctoringData());
  const prevFaceCountRef = useRef<number>(1);
  const sessionStartedRef = useRef(false);

  useEffect(() => {
    sessionStartedRef.current = sessionStarted;
  }, [sessionStarted]);

  useEffect(() => {
    safeCalibrationRef.current = buildSafeEightDotCalibration(calibration);
    gazeFilterRef.current = { x: 0, y: 0, initialized: false };
    headPoseBaselineRef.current = null;
    headPoseSince.current = null;
  }, [calibration]);

  const emit = useCallback((eventType: string, severity: Severity, metadata: Record<string, unknown> = {}) => {
    if (!sessionStartedRef.current) return;

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
      if (!sessionStartedRef.current) {
        URL.revokeObjectURL(clip.url);
        return;
      }

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
    if (sessionStartedRef.current) {
      proctoringDataRef.current.sessionControl.screenShareStoppedCount += 1;
      proctoringDataRef.current.sessionControl.screenShareStoppedAndNotRestored = true;
    }

    setState((current) => ({
      ...current,
      screenShareReadyBeforeInterview: false,
      screenSharePromptRequired: true,
      lastObservationAt: now,
      status: 'Screen sharing stopped',
    }));

    if (sessionStartedRef.current) {
      emit('SCREEN_SHARE_STOPPED', 'HIGH', {
        reason: 'candidate_stopped_screen_share',
        screenShareRequired: true,
        stoppedAt: now,
      });
    }
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

  function computeProctoringScore(): ProctoringScore {
    const pd = proctoringDataRef.current;

    const gazeScore = pd.gaze.totalFrames > 0
      ? Math.max(0, 100 - (pd.gaze.framesLookingAway / pd.gaze.totalFrames) * 100)
      : 100;

    const faceScore = pd.face.totalFrames > 0
      ? Math.max(0, (pd.face.framesNormal / pd.face.totalFrames) * 100)
      : 100;

    const tabScore = Math.max(0, 100 - pd.tabSwitch.count * 15);

    let phoneScore = pd.phone.totalSeconds > 0
      ? Math.max(0, 100 - (pd.phone.detectedSeconds / pd.phone.totalSeconds) * 100)
      : 100;

    if (pd.phone.maxConsecutiveSeconds >= 5) {
      phoneScore = Math.min(phoneScore, 30);
    }

    const headPoseScore = pd.headPose.totalFrames > 0
      ? Math.max(0, (pd.headPose.framesNormal / pd.headPose.totalFrames) * 100)
      : 100;

    let sessionControlScore = Math.max(
      0,
      100 -
        pd.sessionControl.fullscreenExitCount * 20 -
        pd.sessionControl.screenShareStoppedCount * 35 -
        pd.sessionControl.cameraOffSeconds * 2,
    );

    if (pd.sessionControl.cameraOffMaxConsecutiveSeconds >= 10) {
      sessionControlScore = Math.min(sessionControlScore, 30);
    }

    if (pd.sessionControl.screenShareStoppedAndNotRestored) {
      sessionControlScore = Math.min(sessionControlScore, 20);
    }

    const final = Math.round(
      gazeScore * 0.15 +
      faceScore * 0.15 +
      tabScore * 0.20 +
      phoneScore * 0.20 +
      headPoseScore * 0.10 +
      sessionControlScore * 0.20,
    );

    const flags: string[] = [];
    if (pd.tabSwitch.count > 0) {
      flags.push(`Tab switching detected (${pd.tabSwitch.count}x)`);
    }
    if (pd.face.multipleFaceEvents > 0) {
      flags.push(`Multiple faces detected (${pd.face.multipleFaceEvents} occurrence${pd.face.multipleFaceEvents > 1 ? 's' : ''})`);
    }
    if (pd.phone.events > 0) {
      flags.push(`Phone detected (${pd.phone.events} occurrence${pd.phone.events > 1 ? 's' : ''})`);
    }
    if (pd.face.totalFrames > 0 && pd.face.framesNoFace / pd.face.totalFrames > 0.2) {
      flags.push('Prolonged absence from frame');
    }
    if (pd.headPose.totalFrames > 0 && pd.headPose.framesDeviated / pd.headPose.totalFrames > 0.2) {
      flags.push('Significant head movement detected');
    }
    if (pd.gaze.totalFrames > 0 && pd.gaze.framesLookingAway / pd.gaze.totalFrames > 0.2) {
      flags.push('Frequent gaze-away detected');
    }
    if (pd.sessionControl.fullscreenExitCount > 0) {
      flags.push(`Fullscreen exited (${pd.sessionControl.fullscreenExitCount}x)`);
    }
    if (pd.sessionControl.screenShareStoppedCount > 0) {
      flags.push(`Screen sharing stopped (${pd.sessionControl.screenShareStoppedCount}x)`);
    }
    if (pd.sessionControl.cameraOffIncidentCount > 0) {
      flags.push(`Camera off (${pd.sessionControl.cameraOffIncidentCount} occurrence${pd.sessionControl.cameraOffIncidentCount > 1 ? 's' : ''}, ${Math.round(pd.sessionControl.cameraOffSeconds)}s total)`);
    }

    const band: ProctoringScore['band'] =
      final >= 85 ? 'Clean'
      : final >= 65 ? 'Minor Concerns'
      : final >= 40 ? 'Suspicious'
      : 'High Risk';

    return {
      final,
      gazeScore: Math.round(gazeScore),
      faceScore: Math.round(faceScore),
      tabScore: Math.round(tabScore),
      phoneScore: Math.round(phoneScore),
      headPoseScore: Math.round(headPoseScore),
      sessionControlScore: Math.round(sessionControlScore),
      band,
      flags,
    };
  }

  function resetSessionTracking() {
    proctoringDataRef.current = createInitialProctoringData();
    prevFaceCountRef.current = 1;
    setEvents([]);
    setFinalProctoringScore(null);
    setProctoringSessionEnded(false);
    missingSince.current = null;
    multiFaceSince.current = null;
    gazeSince.current = null;
    headPoseSince.current = null;
    cameraOffSince.current = null;
    cameraOffIncidentActive.current = false;
    tabSwitchSince.current = null;
    tabSwitchIncidentActive.current = false;
    tabSwitchReasonRef.current = null;
    fullscreenExitSince.current = null;
    fullscreenIncidentActive.current = false;
    fullscreenExitReasonRef.current = null;
    phoneStateRef.current = createPhoneState();
    foreignObjectStateRef.current = createForeignObjectState();
    clearTabSwitchDurationTimer();

    setState((current) => ({
      ...current,
      status: current.cameraActive ? 'Proctoring session active' : current.status,
      phoneDetected: false,
      foreignObjectDetected: false,
      foreignObjectLabel: '',
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
      fullscreenExitDetected: false,
      fullscreenExitReason: null,
      lastFullscreenExitAt: null,
      lastObservationAt: Date.now(),
    }));
  }

  function startProctoringSession() {
    if (sessionStartedRef.current || proctoringSessionEnded) return false;

    resetSessionTracking();
    sessionStartedRef.current = true;
    setSessionStarted(true);
    emit('PROCTORING_SESSION_STARTED', 'MEDIUM', { startedAt: Date.now() });
    return true;
  }

  function stopLiveProctoringResources() {
    aliveRef.current = false;

    if (frameTimerRef.current !== null && typeof window !== 'undefined') {
      window.clearTimeout(frameTimerRef.current);
      frameTimerRef.current = null;
    }

    clearTabSwitchDurationTimer();

    try {
      violationScreenRecorder.stopViolationRecording();
    } catch {
      // Best-effort cleanup only. Do not block score display if the recorder is already stopped.
    }

    try {
      violationScreenRecorder.stopScreenShare();
    } catch {
      // Best-effort cleanup only. Screen sharing may already be stopped by the browser/user.
    }

    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;

    faceLandmarkerRef.current?.close();
    objectDetectorRef.current?.close();
    faceLandmarkerRef.current = null;
    objectDetectorRef.current = null;

    cameraOffSince.current = null;
    cameraOffIncidentActive.current = false;
    if (cameraOffConfirmTimerRef.current !== null && typeof window !== 'undefined') {
      window.clearTimeout(cameraOffConfirmTimerRef.current);
      cameraOffConfirmTimerRef.current = null;
    }
    tabSwitchSince.current = null;
    tabSwitchIncidentActive.current = false;
    tabSwitchReasonRef.current = null;
    fullscreenExitSince.current = null;
    fullscreenIncidentActive.current = false;
    fullscreenExitReasonRef.current = null;
    fullscreenRequestInFlightRef.current = false;
    screenShareRequestInFlightRef.current = false;
    screenShareGrantedRef.current = false;
  }

  function endProctoringSession(): ProctoringScore {
    const score = computeProctoringScore();
    const endedAt = Date.now();

    setFinalProctoringScore(score);
    setProctoringSessionEnded(true);
    stopLiveProctoringResources();

    emit('PROCTORING_SCORE_FINALIZED', 'MEDIUM', {
      endedAt,
      final: score.final,
      band: score.band,
      gazeScore: score.gazeScore,
      faceScore: score.faceScore,
      tabScore: score.tabScore,
      phoneScore: score.phoneScore,
      headPoseScore: score.headPoseScore,
      sessionControlScore: score.sessionControlScore,
      flags: score.flags,
    });

    sessionStartedRef.current = false;
    setSessionStarted(false);
    setState((current) => ({
      ...current,
      status: 'Session ended. Final proctoring score ready.',
      cameraActive: false,
      faceDetectorActive: false,
      objectDetectorActive: false,
      phoneDetected: false,
      phoneProximity: 'far',
      phoneOrientation: 'unknown',
      foreignObjectDetected: false,
      foreignObjectLabel: '',
      gazeAwayDetected: false,
      gazeDirection: 'center',
      headMovementDetected: false,
      headPoseDeviationDegrees: 0,
      screenShareReadyBeforeInterview: false,
      screenSharePromptRequired: current.screenShareSupported,
      lastObservationAt: endedAt,
    }));

    return score;
  }

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
      if (granted) {
        proctoringDataRef.current.sessionControl.screenShareStoppedAndNotRestored = false;
      }

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
    if (tabSwitchConfirmTimerRef.current !== null && typeof window !== 'undefined') {
      window.clearTimeout(tabSwitchConfirmTimerRef.current);
      tabSwitchConfirmTimerRef.current = null;
    }

    if (cameraOffConfirmTimerRef.current !== null && typeof window !== 'undefined') {
      window.clearTimeout(cameraOffConfirmTimerRef.current);
      cameraOffConfirmTimerRef.current = null;
    }
  }

  function startTabSwitchDurationTimer() {
    if (!sessionStartedRef.current || typeof window === 'undefined' || tabSwitchDurationTimerRef.current !== null) return;

    tabSwitchDurationTimerRef.current = window.setInterval(() => {
      if (!sessionStartedRef.current) return;
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

    const handleClipboardShortcut = (event: KeyboardEvent) => {
      if ((!event.ctrlKey && !event.metaKey) || event.altKey || event.repeat) return;

      const key = event.key.toLowerCase();
      if (key !== 'c' && key !== 'v') return;

      const now = Date.now();
      const alertAt = key === 'c' ? copyShortcutAlertAt : pasteShortcutAlertAt;
      if (now - alertAt.current < CLIPBOARD_SHORTCUT_ALERT_COOLDOWN_MS) return;
      alertAt.current = now;

      const target = event.target instanceof HTMLElement ? event.target : null;
      const metadata = {
        shortcut: `${event.metaKey ? 'Meta' : 'Ctrl'}+${key.toUpperCase()}`,
        key: key.toUpperCase(),
        targetTag: target?.tagName?.toLowerCase() ?? 'unknown',
        targetType: target instanceof HTMLInputElement ? target.type : undefined,
        targetEditable: Boolean(
          target?.isContentEditable
          || target instanceof HTMLInputElement
          || target instanceof HTMLTextAreaElement,
        ),
      };

      if (key === 'c') {
        emit('COPY_SHORTCUT_DETECTED', 'MEDIUM', metadata);
        violationScreenRecorder.recordViolationClip('COPY_SHORTCUT', metadata);
      } else {
        emit('PASTE_SHORTCUT_DETECTED', 'HIGH', metadata);
        violationScreenRecorder.recordViolationClip('PASTE_SHORTCUT', metadata);
      }
    };

    window.addEventListener('keydown', handleClipboardShortcut, true);

    return () => {
      window.removeEventListener('keydown', handleClipboardShortcut, true);
    };
  }, [emit, violationScreenRecorder.recordViolationClip]);

  useEffect(() => {
    if (typeof window === 'undefined' || typeof document === 'undefined') return;

    const getTabAwayReason = (trigger: string) => {
      if (document.visibilityState === 'hidden') return 'document_hidden';
      if (!document.hasFocus()) return trigger === 'blur' ? 'window_blur' : 'window_not_focused';
      return null;
    };

    const markTabAway = (trigger: string) => {
      if (!sessionStartedRef.current) return;

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

      // Do not treat the initial mount as a real tab switch event.
      if (trigger === 'mount') return;

      // Require a short confirmation window before treating transient focus/blur as a tab switch.
      if (tabSwitchConfirmTimerRef.current !== null && typeof window !== 'undefined') {
        window.clearTimeout(tabSwitchConfirmTimerRef.current);
        tabSwitchConfirmTimerRef.current = null;
      }

      tabSwitchConfirmTimerRef.current = typeof window !== 'undefined'
        ? window.setTimeout(() => {
          if (!sessionStartedRef.current) {
            tabSwitchConfirmTimerRef.current = null;
            return;
          }

          if (!tabSwitchIncidentActive.current) {
            tabSwitchIncidentActive.current = true;
            proctoringDataRef.current.tabSwitch.count += 1;
            const metadata = {
              reason,
              trigger,
              startedAt: tabSwitchSince.current,
              activeDurationMs: tabSwitchSince.current ? Date.now() - (tabSwitchSince.current as number) : 0,
              visibilityState: document.visibilityState,
              documentHidden: document.hidden,
              hasFocus: document.hasFocus(),
            };

            if (emitTabSwitchWithCooldown(metadata)) {
              violationScreenRecorder.startViolationRecording('TAB_SWITCH', metadata);
            }
          }
          tabSwitchConfirmTimerRef.current = null;
        }, TAB_SWITCH_CONFIRM_MS)
        : null;
    };

    const markTabReturned = (trigger: string) => {
      if (!sessionStartedRef.current) {
        tabSwitchSince.current = null;
        tabSwitchIncidentActive.current = false;
        tabSwitchReasonRef.current = null;
        clearTabSwitchDurationTimer();
        return;
      }

      if (getTabAwayReason(trigger)) return;

      // If a confirmation timer was pending (transient blur), cancel it and do not record an incident.
      if (tabSwitchConfirmTimerRef.current !== null && typeof window !== 'undefined') {
        window.clearTimeout(tabSwitchConfirmTimerRef.current);
        tabSwitchConfirmTimerRef.current = null;
      }

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
      if (!sessionStartedRef.current) {
        fullscreenExitSince.current = null;
        fullscreenIncidentActive.current = false;
        fullscreenExitReasonRef.current = null;
        setState((current) => ({
          ...current,
          fullscreenActive: false,
          fullscreenExitDetected: false,
          fullscreenExitReason: null,
          fullscreenSupported: isFullscreenSupported(),
          fullscreenReadyBeforeInterview: false,
          fullscreenPromptRequired: isFullscreenSupported(),
          lastObservationAt: now,
        }));
        return;
      }

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
      const isNewFullscreenIncident = !fullscreenIncidentActive.current;
      fullscreenIncidentActive.current = true;
      if (isNewFullscreenIncident) {
        proctoringDataRef.current.sessionControl.fullscreenExitCount += 1;
      }

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
      if (!sessionStartedRef.current) return;
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
        phoneProximity: 'far',
        phoneOrientation: 'unknown',
        foreignObjectDetected: false,
        foreignObjectLabel: '',
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
        microphoneSupported: current.microphoneSupported,
        microphoneReadyBeforeInterview: current.microphoneReadyBeforeInterview,
        microphonePromptRequired: current.microphonePromptRequired,
        preInterviewMicrophoneRequestedAt: current.preInterviewMicrophoneRequestedAt,
        preInterviewMicrophoneGrantedAt: current.preInterviewMicrophoneGrantedAt,
        screenShareSupported: current.screenShareSupported,
        screenShareReadyBeforeInterview: current.screenShareReadyBeforeInterview,
        screenSharePromptRequired: current.screenSharePromptRequired,
        preInterviewScreenShareRequestedAt: current.preInterviewScreenShareRequestedAt,
        preInterviewScreenShareGrantedAt: current.preInterviewScreenShareGrantedAt,
        lastObservationAt: current.lastObservationAt,
      }));

      const requestedAt = Date.now();
      setState((current) => ({
        ...current,
        microphonePromptRequired: false,
        status: 'Requesting camera...',
        lastObservationAt: requestedAt,
      }));


      setFinalProctoringScore(null);
      setProctoringSessionEnded(false);
      proctoringDataRef.current = createInitialProctoringData();
      prevFaceCountRef.current = 1;

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
        microphoneReadyBeforeInterview: false,
        microphonePromptRequired: false,
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
          scoreThreshold: 0.25,
          maxResults: 8,
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
          if (!sessionStartedRef.current) {
            cameraOffSince.current = null;
            cameraOffIncidentActive.current = false;
            if (cameraOffConfirmTimerRef.current !== null && typeof window !== 'undefined') {
              window.clearTimeout(cameraOffConfirmTimerRef.current);
              cameraOffConfirmTimerRef.current = null;
            }
            setState((current) => ({
              ...current,
              cameraActive: false,
              faceDetectorActive: !!faceTask,
              objectDetectorActive: !!objectTask,
              lastObservationAt: now,
              status: 'Camera is off or not sending video',
            }));
            frameTimerRef.current = window.setTimeout(tick, LIVE_INTERVAL_MS);
            return;
          }

          if (!cameraOffSince.current) cameraOffSince.current = now;

          // If we don't yet have a confirmed incident, start a confirm timer.
          if (!cameraOffIncidentActive.current && cameraOffConfirmTimerRef.current === null && typeof window !== 'undefined') {
            cameraOffConfirmTimerRef.current = window.setTimeout(() => {
              cameraOffConfirmTimerRef.current = null;
              if (!sessionStartedRef.current) return;

              // Mark incident active and capture initial metadata/recording
              const proctoringData = proctoringDataRef.current;
              if (!cameraOffIncidentActive.current) {
                proctoringData.sessionControl.cameraOffIncidentCount += 1;
              }
              cameraOffIncidentActive.current = true;

              const durationMs = Date.now() - (cameraOffSince.current || Date.now());
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
            }, CAMERA_OFF_CONFIRM_MS);
          }

          // If an incident is active, accumulate seconds.
          if (cameraOffIncidentActive.current) {
            const proctoringData = proctoringDataRef.current;
            proctoringData.sessionControl.cameraOffSeconds += LIVE_INTERVAL_MS / 1000;
            proctoringData.sessionControl.cameraOffConsecutiveSeconds += LIVE_INTERVAL_MS / 1000;
            proctoringData.sessionControl.cameraOffMaxConsecutiveSeconds = Math.max(
              proctoringData.sessionControl.cameraOffMaxConsecutiveSeconds,
              proctoringData.sessionControl.cameraOffConsecutiveSeconds,
            );
          }

          missingSince.current = null;
          setState((current) => ({
            ...current,
            cameraActive: false,
            faceDetectorActive: !!faceTask,
            objectDetectorActive: !!objectTask,
            faceCount: 0,
            phoneDetected: false,
            phoneProximity: 'far',
            phoneOrientation: 'unknown',
            foreignObjectDetected: false,
            foreignObjectLabel: '',
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
          // Clear any pending confirm timer when camera is restored.
          if (cameraOffConfirmTimerRef.current !== null && typeof window !== 'undefined') {
            window.clearTimeout(cameraOffConfirmTimerRef.current);
            cameraOffConfirmTimerRef.current = null;
          }
          cameraOffSince.current = null;
          cameraOffIncidentActive.current = false;
          proctoringDataRef.current.sessionControl.cameraOffConsecutiveSeconds = 0;
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
            phoneProximity: 'far',
            phoneOrientation: 'unknown',
            foreignObjectDetected: false,
            foreignObjectLabel: '',
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
        const videoWidth = currentVideo.videoWidth || 640;
        const videoHeight = currentVideo.videoHeight || 480;

        const phone = objectResult
          ? analyzePhoneDetection(objectResult, videoWidth, videoHeight, phoneStateRef.current)
          : null;
        const detectedPhone = phone?.detected ?? false;

        const foreignObject = objectResult
          ? analyzeForeignObject(objectResult, videoWidth, videoHeight, foreignObjectStateRef.current)
          : null;
        const detectedForeignObject = foreignObject?.detected ?? false;

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

        if (!sessionStartedRef.current) {
          missingSince.current = null;
          multiFaceSince.current = null;
          gazeSince.current = null;
          headPoseSince.current = null;
          phoneStateRef.current = createPhoneState();
          foreignObjectStateRef.current = createForeignObjectState();
          prevFaceCountRef.current = faceCount;
          setState((current) => ({
            ...current,
            cameraActive: true,
            faceDetectorActive: true,
            objectDetectorActive: true,
            faceCount,
            phoneDetected: false,
            foreignObjectDetected: false,
            foreignObjectLabel: '',
            gazeAwayDetected: false,
            gazeDirection: 'center',
            headMovementDetected: false,
            headPoseDeviationDegrees: 0,
            lastObservationAt: Date.now(),
            status: 'Ready to start session',
          }));
          frameTimerRef.current = window.setTimeout(tick, LIVE_INTERVAL_MS);
          return;
        }

        // Proctoring score counters: update once per detection tick without causing re-renders.
        const proctoringData = proctoringDataRef.current;
        const intervalSeconds = LIVE_INTERVAL_MS / 1000;

        proctoringData.gaze.totalFrames += 1;
        if (gaze.away) {
          proctoringData.gaze.framesLookingAway += 1;
        }

        proctoringData.face.totalFrames += 1;
        if (faceCount === 0) {
          proctoringData.face.framesNoFace += 1;
        } else if (faceCount === 1) {
          proctoringData.face.framesNormal += 1;
        } else {
          proctoringData.face.framesMultipleFaces += 1;
        }

        if (faceCount > 1 && prevFaceCountRef.current <= 1) {
          proctoringData.face.multipleFaceEvents += 1;
        }
        prevFaceCountRef.current = faceCount;

        if (faceCount === 1 && headPose) {
          proctoringData.headPose.totalFrames += 1;
          if (headMovementDetected) {
            proctoringData.headPose.framesDeviated += 1;
          } else {
            proctoringData.headPose.framesNormal += 1;
          }
        }

        proctoringData.phone.totalSeconds += intervalSeconds;
        if (detectedPhone) {
          proctoringData.phone.detectedSeconds += intervalSeconds;
          proctoringData.phone.consecutiveSeconds += intervalSeconds;
          proctoringData.phone.maxConsecutiveSeconds = Math.max(
            proctoringData.phone.maxConsecutiveSeconds,
            proctoringData.phone.consecutiveSeconds,
          );
          if (!proctoringData.phone.wasDetected) {
            proctoringData.phone.events += 1;
          }
          proctoringData.phone.wasDetected = true;
        } else {
          proctoringData.phone.consecutiveSeconds = 0;
          proctoringData.phone.wasDetected = false;
        }

        setState((current) => ({
          ...current,
          cameraActive: true,
          faceDetectorActive: true,
          objectDetectorActive: true,
          faceCount,
          phoneDetected: detectedPhone,
          phoneProximity: phone?.proximity ?? 'far',
          phoneOrientation: phone?.orientation ?? 'unknown',
          foreignObjectDetected: detectedForeignObject,
          foreignObjectLabel: foreignObject?.label ?? '',
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
                ? `Phone detected (${phone?.proximity ?? 'unknown'}, ${phone?.orientation ?? 'unknown'})`
                : detectedForeignObject
                  ? `Foreign object detected (${foreignObject?.trigger ?? 'unknown'}: ${foreignObject?.label ?? 'object'})`
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

        if (phone && consumePhoneAlert(phone, phoneStateRef.current)) {
          const metadata = {
            proximity: phone.proximity,
            orientation: phone.orientation,
            source: phone.source,
            confidence: phone.rollingConfidence,
            detections: phone.detections,
          };

          if (emitWithCooldown(phoneAlertAt, 'MOBILE_PHONE_DETECTED', 'HIGH', metadata)) {
            violationScreenRecorder.recordViolationClip('MOBILE_PHONE', metadata);
          }
        }

        if (foreignObject && consumeForeignObjectAlert(foreignObject, foreignObjectStateRef.current)) {
          const metadata = {
            label: foreignObject.label,
            trigger: foreignObject.trigger,
            confidence: foreignObject.confidence,
            areaRatio: foreignObject.areaRatio,
            edgeSides: foreignObject.edgeSides,
          };

          if (emitWithCooldown(foreignObjectAlertAt, 'FOREIGN_OBJECT_DETECTED', 'HIGH', metadata)) {
            violationScreenRecorder.recordViolationClip('FOREIGN_OBJECT', metadata);
          }
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
        phoneProximity: 'far',
        phoneOrientation: 'unknown',
        foreignObjectDetected: false,
        foreignObjectLabel: '',
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
        microphoneSupported: typeof navigator === 'undefined' ? true : isMicrophoneCaptureSupported(),
        microphoneReadyBeforeInterview: false,
        microphonePromptRequired: true,
        preInterviewMicrophoneRequestedAt: null,
        preInterviewMicrophoneGrantedAt: null,
        screenShareSupported: typeof navigator === 'undefined' ? true : isScreenShareRecordingSupported(),
        screenShareReadyBeforeInterview: violationScreenRecorder.hasScreenSharePermission,
        screenSharePromptRequired: !violationScreenRecorder.hasScreenSharePermission,
        preInterviewScreenShareRequestedAt: null,
        preInterviewScreenShareGrantedAt: violationScreenRecorder.hasScreenSharePermission ? Date.now() : null,
        lastObservationAt: null,
      });
      const message = error instanceof Error ? error.message : 'getUserMedia failed';
      const metadata = { reason: message, permissionDenied: true };
      if (sessionStartedRef.current) {
        proctoringDataRef.current.sessionControl.cameraOffIncidentCount += 1;
        proctoringDataRef.current.sessionControl.cameraOffMaxConsecutiveSeconds = Math.max(
          proctoringDataRef.current.sessionControl.cameraOffMaxConsecutiveSeconds,
          10,
        );
        emit('CAMERA_PERMISSION_DENIED', 'HIGH', { message });
        emit('CAMERA_OFF_DETECTED', 'HIGH', metadata);
        violationScreenRecorder.recordViolationClip('CAMERA_OFF', metadata);
      }
    });

    return () => {
      aliveRef.current = false;
      if (frameTimerRef.current) {
        window.clearTimeout(frameTimerRef.current);
      }
      streamRef.current?.getTracks().forEach((track) => track.stop());
      cameraOffSince.current = null;
      cameraOffIncidentActive.current = false;
      if (cameraOffConfirmTimerRef.current !== null && typeof window !== 'undefined') {
        window.clearTimeout(cameraOffConfirmTimerRef.current);
        cameraOffConfirmTimerRef.current = null;
      }
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
    sessionStarted,
    startProctoringSession,
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
    /** Raw per-frame counters. Read via .current — do not mutate directly. */
    proctoringRef: proctoringDataRef,
    /** Final score is null while the interview is running and becomes available after endProctoringSession(). */
    finalProctoringScore,
    proctoringScore: finalProctoringScore,
    proctoringSessionEnded,
    sessionEnded: proctoringSessionEnded,
    showProctoringScore: proctoringSessionEnded && finalProctoringScore !== null,
    /** Call this from the End Interview button. It freezes counters, stops monitoring, stores, emits, and returns the final score. */
    endProctoringSession,
    completeSession: endProctoringSession,
    handleCompleteSession: endProctoringSession,
    onCompleteSession: endProctoringSession,
    endSession: endProctoringSession,
    finishProctoringSession: endProctoringSession,
    /** Still available if you only want to compute a preview without ending/stopping the session. */
    computeProctoringScore,
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
