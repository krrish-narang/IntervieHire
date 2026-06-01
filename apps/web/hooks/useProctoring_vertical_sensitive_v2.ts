'use client';

import { useEffect, useRef, useState } from 'react';
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
const HEAD_POSE_ROTATION_THRESHOLD_DEG = 23;
const HEAD_POSE_AXIS_THRESHOLD_DEG = 23;
const GAZE_BLENDSHAPE_THRESHOLD = 0.8; // more sensitive blendshape threshold
// Blendshapes are used only to correct vertical direction when geometry already says gaze is away.
// Looking down can make the iris partially occluded by the eyelid, which sometimes makes raw iris geometry look like "up".
const VERTICAL_BLENDSHAPE_DIRECTION_THRESHOLD = 0.33;
const VERTICAL_BLENDSHAPE_MARGIN = 0.08;
// Downward gaze is usually weaker in iris geometry because the eyelids partially cover the iris.
// Keep normal up/left/right sensitivity unchanged, but make positive-Y/downward movement easier to trigger.
const DOWNWARD_GAZE_THRESHOLD_FACTOR = 1.7;
const MIN_DOWNWARD_GAZE_THRESHOLD = 0.065;
const DOWNWARD_BLENDSHAPE_AWAY_THRESHOLD = 0.4;
const DOWNWARD_BLENDSHAPE_MARGIN = 0.1;
const DOWNWARD_GEOMETRY_SUPPORT_FACTOR = 0.4;
const MIN_DOWNWARD_GEOMETRY_SUPPORT = 0.03;
// Fallback geometry thresholds used when no calibration has been run
const DEFAULT_GAZE_THRESHOLD_X = FALLBACK_GAZE_THRESHOLD_X;
const DEFAULT_GAZE_THRESHOLD_Y = FALLBACK_GAZE_THRESHOLD_Y;

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

function detectGazeAway(
  result: FaceLandmarkerResult | null,
  thresholdX = DEFAULT_GAZE_THRESHOLD_X,
  thresholdY = DEFAULT_GAZE_THRESHOLD_Y,
  neutralX = 0,
  neutralY = 0,
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
      };
    }
    return { away: false, direction: 'center', confidence: 0, source: 'geometry' as const };
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

  const upBlendshapeScore = Math.min(upLeft, upRight);
  const downBlendshapeScore = Math.min(downLeft, downRight);
  const horizontalGazeAway = Math.abs(useX) >= effectiveThresholdX;
  const upwardGazeAway = useY <= -effectiveThresholdUp;
  const downwardGazeAway = useY >= effectiveThresholdDown;
  const downwardGeometrySupport = Math.max(
    effectiveThresholdDown * DOWNWARD_GEOMETRY_SUPPORT_FACTOR,
    MIN_DOWNWARD_GEOMETRY_SUPPORT,
  );
  const downwardBlendshapeAway =
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
      source: (direction === vertical ? verticalSource : 'geometry') as const,
    };
  }

  return { away: false, direction: 'center', confidence: 0, source: 'geometry' as const };
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
    lastObservationAt: null,
  });
  const missingSince = useRef<number | null>(null);
  const faceAlertAt = useRef<number>(0);
  const phoneAlertAt = useRef<number>(0);
  const gazeAlertAt = useRef<number>(0);
  const headPoseAlertAt = useRef<number>(0);
  const multiFaceAlertAt = useRef<number>(0);
  const multiFaceSince = useRef<number | null>(null);
  const phoneSince = useRef<number | null>(null);
  const gazeSince = useRef<number | null>(null);
  const headPoseSince = useRef<number | null>(null);
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

  function emit(eventType: string, severity: Severity, metadata: Record<string, unknown> = {}) {
    const payload: ProctoringPayload = { type: 'proctoring_event', sessionId, eventType, severity, metadata, timestamp: Date.now() };
    socket?.readyState === 1 && socket.send(JSON.stringify(payload));
    setEvents((current) => [{ eventType, severity, timestamp: Date.now(), metadata }, ...current].slice(0, 10));
  }

  function emitWithCooldown(ref: { current: number }, eventType: string, severity: Severity, metadata: Record<string, unknown> = {}) {
    const now = Date.now();
    if (now - ref.current < ALERT_COOLDOWN_MS) return;
    ref.current = now;
    emit(eventType, severity, metadata);
  }

  useEffect(() => {
    aliveRef.current = true;

    async function start() {
      setState({
        initialized: false,
        status: 'Requesting camera access...',
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
        lastObservationAt: null,
      });

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

      setState((current) => ({
        ...current,
        initialized: true,
        status: 'Detection active',
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
        if (!currentVideo || !faceTask || !objectTask || currentVideo.readyState < 2) {
          if (!missingSince.current) missingSince.current = Date.now();
          if (Date.now() - (missingSince.current || 0) > NO_FACE_THRESHOLD_MS) {
            emitWithCooldown(faceAlertAt, 'FACE_NOT_DETECTED', 'HIGH', { durationMs: Date.now() - (missingSince.current || 0) });
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
        const safeCalibration = safeCalibrationRef.current;
        const gaze = detectGazeAway(
          faceResult,
          safeCalibration.thresholdX ?? DEFAULT_GAZE_THRESHOLD_X,
          safeCalibration.thresholdY ?? DEFAULT_GAZE_THRESHOLD_Y,
          safeCalibration.neutralX ?? 0,
          safeCalibration.neutralY ?? 0,
          gazeFilterRef,
          0.18, // reduced from 0.28 for better filtering of false positives
        );

        const headPose = faceCount === 1 ? estimateHeadPose(faceResult) : null;
        if (headPose && !headPoseBaselineRef.current) {
          headPoseBaselineRef.current = headPose;
        }
        const headPoseDeviation = headPose && headPoseBaselineRef.current ? calculateHeadPoseDeviation(headPose, headPoseBaselineRef.current) : null;
        const headMovementDetected = Boolean(headPoseDeviation?.tooMuch);

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
          status: faceCount > 1
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
            emitWithCooldown(faceAlertAt, 'FACE_NOT_DETECTED', 'HIGH', { durationMs: Date.now() - (missingSince.current || 0) });
          }
        } else {
          missingSince.current = null;
        }

        // require a short confirmation window to avoid spurious multi-face / phone alerts
        if (faceCount > 1) {
          if (!multiFaceSince.current) multiFaceSince.current = Date.now();
          if (Date.now() - (multiFaceSince.current || 0) > MULTI_FACE_CONFIRM_MS) {
            emitWithCooldown(multiFaceAlertAt, 'MULTIPLE_FACES_DETECTED', 'HIGH', { faceCount, faces: faceCount });
            multiFaceSince.current = null;
          }
        } else {
          multiFaceSince.current = null;
        }

        if (detectedPhone) {
          if (!phoneSince.current) phoneSince.current = Date.now();
          if (Date.now() - (phoneSince.current || 0) > PHONE_CONFIRM_MS) {
            emitWithCooldown(phoneAlertAt, 'MOBILE_PHONE_DETECTED', 'HIGH', {
              detections: objectResult?.detections?.map((detection) => ({
                label: detection.categories?.[0]?.categoryName || detection.categories?.[0]?.displayName || 'object',
                score: detection.categories?.[0]?.score ?? 0,
                box: detection.boundingBox,
              })),
              topScore: objectResult ? latestDetectionScore(objectResult) : 0,
            });
            phoneSince.current = null;
          }
        } else {
          phoneSince.current = null;
        }

        if (gaze.away) {
          if (!gazeSince.current) gazeSince.current = Date.now();
          if (Date.now() - (gazeSince.current || 0) > GAZE_CONFIRM_MS) {
            emitWithCooldown(gazeAlertAt, 'GAZE_AWAY_DETECTED', 'MEDIUM', {
              direction: gaze.direction,
              confidence: gaze.confidence,
              source: gaze.source,
              calibrationTrusted: safeCalibration.trusted,
              calibrationReason: safeCalibration.reason,
              thresholdX: safeCalibration.thresholdX,
              thresholdY: safeCalibration.thresholdY,
            });
            gazeSince.current = null;
          }
        } else {
          gazeSince.current = null;
        }

        if (headMovementDetected && headPoseDeviation && headPose) {
          if (!headPoseSince.current) headPoseSince.current = Date.now();
          if (Date.now() - (headPoseSince.current || 0) > HEAD_POSE_CONFIRM_MS) {
            emitWithCooldown(headPoseAlertAt, 'HEAD_MOVEMENT_DETECTED', 'MEDIUM', {
              deviationDegrees: Math.round(headPoseDeviation.magnitude),
              maxAxisDegrees: Math.round(headPoseDeviation.maxAxis),
              yawDelta: Math.round(headPoseDeviation.yaw),
              pitchDelta: Math.round(headPoseDeviation.pitch),
              rollDelta: Math.round(headPoseDeviation.roll),
              source: headPose.source,
            });
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
        lastObservationAt: null,
      });
      emit('CAMERA_PERMISSION_DENIED', 'HIGH', { message: error instanceof Error ? error.message : 'getUserMedia failed' });
    });

    return () => {
      aliveRef.current = false;
      if (frameTimerRef.current) {
        window.clearTimeout(frameTimerRef.current);
      }
      streamRef.current?.getTracks().forEach((track) => track.stop());
      faceLandmarkerRef.current?.close();
      objectDetectorRef.current?.close();
      faceLandmarkerRef.current = null;
      objectDetectorRef.current = null;
      streamRef.current = null;
    };
  }, [sessionId, socket]);

  return { videoRef, events, emit, state };
}
