'use client';

import { useCallback, useRef, useState } from 'react';
import { FaceLandmarker, FilesetResolver, type FaceLandmarkerResult } from '@mediapipe/tasks-vision';

const FACE_MODEL_URL =
  'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task';

// How many frames to sample per calibration point
const SAMPLES_PER_POINT = 20;
// ms between each sample
const SAMPLE_INTERVAL_MS = 80;

export type CalibrationPoint = {
  id: string;
  label: string;
  // 0..1 fractions of screen
  xFrac: number;
  yFrac: number;
};

// 9-point grid: corners, edges, center
export const CALIBRATION_POINTS: CalibrationPoint[] = [
  { id: 'tl',  label: 'Top-left',     xFrac: 0.1,  yFrac: 0.1  },
  { id: 'tc',  label: 'Top-center',   xFrac: 0.5,  yFrac: 0.1  },
  { id: 'tr',  label: 'Top-right',    xFrac: 0.9,  yFrac: 0.1  },
  { id: 'ml',  label: 'Middle-left',  xFrac: 0.1,  yFrac: 0.5  },
  { id: 'mc',  label: 'Center',       xFrac: 0.5,  yFrac: 0.5  },
  { id: 'mr',  label: 'Middle-right', xFrac: 0.9,  yFrac: 0.5  },
  { id: 'bl',  label: 'Bottom-left',  xFrac: 0.1,  yFrac: 0.9  },
  { id: 'bc',  label: 'Bottom-center',xFrac: 0.5,  yFrac: 0.9  },
  { id: 'br',  label: 'Bottom-right', xFrac: 0.9,  yFrac: 0.9  },
];

type IrisSample = { offsetX: number; offsetY: number };

type PointSamples = {
  pointId: string;
  samples: IrisSample[];
};

export type CalibrationResult = {
  // Personalised iris-offset thresholds
  thresholdX: number;
  thresholdY: number;
  // Neutral center offsets (subtracted before threshold test)
  neutralX: number;
  neutralY: number;
  // Per-point raw data (useful for debugging)
  pointData: PointSamples[];
  // Sanity score 0-1: how much the edge points actually differ from center
  qualityScore: number;
};

export type CalibrationPhase =
  | 'idle'
  | 'intro'          // explaining what's about to happen
  | 'waiting'        // countdown before a point activates
  | 'sampling'       // actively collecting iris data for this point
  | 'between'        // brief rest between points
  | 'done'
  | 'error';

export type CalibrationState = {
  phase: CalibrationPhase;
  currentPointIndex: number;
  samplesCollected: number;
  result: CalibrationResult | null;
  error: string | null;
};

function getFacePoint(
  landmarks: FaceLandmarkerResult['faceLandmarks'][number] | undefined,
  index: number,
) {
  return landmarks?.[index];
}

function sampleIrisOffset(
  faceTask: FaceLandmarker,
  video: HTMLVideoElement,
): IrisSample | null {
  let result: FaceLandmarkerResult | null = null;
  try {
    result = (faceTask as any).detectForVideo(video, performance.now());
  } catch {
    return null;
  }
  const lm = result?.faceLandmarks?.[0];
  if (!lm) return null;

  const leftOuter  = getFacePoint(lm, 33);
  const leftInner  = getFacePoint(lm, 133);
  const leftIris   = getFacePoint(lm, 468);
  const rightInner = getFacePoint(lm, 362);
  const rightOuter = getFacePoint(lm, 263);
  const rightIris  = getFacePoint(lm, 473);

  if (!leftOuter || !leftInner || !leftIris || !rightInner || !rightOuter || !rightIris)
    return null;

  const leftW  = Math.max(Math.abs(leftOuter.x  - leftInner.x),  0.0001);
  const rightW = Math.max(Math.abs(rightOuter.x - rightInner.x), 0.0001);

  // upper/lower lids for vertical normalisation
  const lUp   = getFacePoint(lm, 159);
  const lDown = getFacePoint(lm, 145);
  const rUp   = getFacePoint(lm, 386);
  const rDown = getFacePoint(lm, 374);
  const leftH  = Math.max(Math.abs((lUp?.y ?? leftIris.y) - (lDown?.y ?? leftIris.y)), 0.0001);
  const rightH = Math.max(Math.abs((rUp?.y ?? rightIris.y) - (rDown?.y ?? rightIris.y)), 0.0001);

  const leftMidX  = (leftOuter.x  + leftInner.x)  / 2;
  const rightMidX = (rightOuter.x + rightInner.x) / 2;
  const leftMidY  = ((lUp?.y ?? leftIris.y) + (lDown?.y ?? leftIris.y)) / 2;
  const rightMidY = ((rUp?.y ?? rightIris.y) + (rDown?.y ?? rightIris.y)) / 2;

  const lOffX = (leftIris.x  - leftMidX)  / (leftW  / 2);
  const rOffX = (rightIris.x - rightMidX) / (rightW / 2);
  const lOffY = (leftIris.y  - leftMidY)  / (leftH  / 2);
  const rOffY = (rightIris.y - rightMidY) / (rightH / 2);

  return {
    offsetX: (lOffX + rOffX) / 2,
    offsetY: (lOffY + rOffY) / 2,
  };
}

function median(values: number[]): number {
  if (values.length === 0) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 0
    ? (sorted[mid - 1]! + sorted[mid]!) / 2
    : sorted[mid]!;
}

function computeResult(allPointSamples: PointSamples[]): CalibrationResult {
  // Center point gives us the neutral baseline
  const centerData = allPointSamples.find((p) => p.pointId === 'mc');
  const neutralX = centerData ? median(centerData.samples.map((s) => s.offsetX)) : 0;
  const neutralY = centerData ? median(centerData.samples.map((s) => s.offsetY)) : 0;

  // For each non-center point, compute how far from neutral
  // Threshold = 70% of the smallest observed deviation across edge points
  // (conservative so normal head-with-eyes-center doesn't false-trigger)
  const edgeDeviationsX: number[] = [];
  const edgeDeviationsY: number[] = [];

  for (const pd of allPointSamples) {
    if (pd.pointId === 'mc') continue;
    const medX = median(pd.samples.map((s) => s.offsetX));
    const medY = median(pd.samples.map((s) => s.offsetY));
    edgeDeviationsX.push(Math.abs(medX - neutralX));
    edgeDeviationsY.push(Math.abs(medY - neutralY));
  }

  // Use 60th-percentile deviation as the threshold (not min, not max)
  edgeDeviationsX.sort((a, b) => a - b);
  edgeDeviationsY.sort((a, b) => a - b);
  const p60idx = Math.floor(edgeDeviationsX.length * 0.6);
  const rawThreshX = edgeDeviationsX[p60idx] ?? 0.18;
  const rawThreshY = edgeDeviationsY[p60idx] ?? 0.22;

  // Scale down slightly so off-screen gaze still triggers but on-screen edge points remain valid.
  const thresholdX = Math.max(rawThreshX * 0.92, 0.05);
  const thresholdY = Math.max(rawThreshY * 0.92, 0.05);

  // Quality: how separable is the center from the edges?
  // A score near 1 means the person's eyes moved a lot between center and corners.
  const avgEdgeX = edgeDeviationsX.reduce((a, b) => a + b, 0) / (edgeDeviationsX.length || 1);
  const avgEdgeY = edgeDeviationsY.reduce((a, b) => a + b, 0) / (edgeDeviationsY.length || 1);
  const qualityScore = Math.min(
    ((avgEdgeX / Math.max(thresholdX, 0.001)) + (avgEdgeY / Math.max(thresholdY, 0.001))) / 4,
    1,
  );

  return { thresholdX, thresholdY, neutralX, neutralY, pointData: allPointSamples, qualityScore };
}

export function useGazeCalibration(videoRef: React.RefObject<HTMLVideoElement>) {
  const [calState, setCalState] = useState<CalibrationState>({
    phase: 'idle',
    currentPointIndex: -1,
    samplesCollected: 0,
    result: null,
    error: null,
  });

  const faceLandmarkerRef = useRef<FaceLandmarker | null>(null);
  const abortRef = useRef(false);
  const allPointSamplesRef = useRef<PointSamples[]>([]);

  // Call this from the intro screen's "Start" button
  const startCalibration = useCallback(async () => {
    abortRef.current = false;
    allPointSamplesRef.current = [];

    setCalState({ phase: 'intro', currentPointIndex: -1, samplesCollected: 0, result: null, error: null });

    // Initialise MediaPipe if not already done
    if (!faceLandmarkerRef.current) {
      try {
        const vision = await FilesetResolver.forVisionTasks(
          'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm',
        );
        faceLandmarkerRef.current = await FaceLandmarker.createFromOptions(vision, {
          baseOptions: { modelAssetPath: FACE_MODEL_URL },
          runningMode: 'VIDEO',
          numFaces: 1,
          minFaceDetectionConfidence: 0.55,
          minFacePresenceConfidence: 0.55,
          minTrackingConfidence: 0.55,
          outputFaceBlendshapes: false,
        });
      } catch (err) {
        setCalState((s) => ({ ...s, phase: 'error', error: 'Failed to load face model' }));
        return;
      }
    }
  }, []);

  // Call this when user clicks "Begin" after reading the intro
  const beginPoints = useCallback(async () => {
    const video = videoRef.current;
    const faceTask = faceLandmarkerRef.current;
    if (!video || !faceTask) return;

    for (let i = 0; i < CALIBRATION_POINTS.length; i++) {
      if (abortRef.current) return;

      const point = CALIBRATION_POINTS[i]!;

      // "waiting" phase: show the dot, give user 1.5s to move eyes to it
      setCalState((s) => ({
        ...s,
        phase: 'waiting',
        currentPointIndex: i,
        samplesCollected: 0,
      }));
      await sleep(1500);
      if (abortRef.current) return;

      // "sampling" phase: collect frames
      setCalState((s) => ({ ...s, phase: 'sampling' }));
      const samples: IrisSample[] = [];

      for (let j = 0; j < SAMPLES_PER_POINT; j++) {
        if (abortRef.current) return;
        const sample = sampleIrisOffset(faceTask, video);
        if (sample) samples.push(sample);
        setCalState((s) => ({ ...s, samplesCollected: samples.length }));
        await sleep(SAMPLE_INTERVAL_MS);
      }

      allPointSamplesRef.current.push({ pointId: point.id, samples });

      // "between" phase (skip after last point)
      if (i < CALIBRATION_POINTS.length - 1) {
        setCalState((s) => ({ ...s, phase: 'between' }));
        await sleep(600);
      }
    }

    if (abortRef.current) return;

    const result = computeResult(allPointSamplesRef.current);
    setCalState((s) => ({ ...s, phase: 'done', result }));
  }, [videoRef]);

  const abort = useCallback(() => {
    abortRef.current = true;
    setCalState({ phase: 'idle', currentPointIndex: -1, samplesCollected: 0, result: null, error: null });
  }, []);

  return { calState, startCalibration, beginPoints, abort };
}

function sleep(ms: number) {
  return new Promise<void>((resolve) => setTimeout(resolve, ms));
}
