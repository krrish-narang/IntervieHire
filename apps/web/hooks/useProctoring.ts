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

// ─────────────────────────────────────────────────────────────────────────────
// General proctoring types
// ─────────────────────────────────────────────────────────────────────────────

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
  phoneProximity: PhoneProximity;
  phoneOrientation: PhoneOrientation;
  foreignObjectDetected: boolean;       // ← NEW
  foreignObjectLabel: string;           // ← NEW
  gazeAwayDetected: boolean;
  gazeDirection: string;
  lastObservationAt: number | null;
};

// ─────────────────────────────────────────────────────────────────────────────
// MediaPipe / detection constants
// ─────────────────────────────────────────────────────────────────────────────

const FACE_MODEL_URL =
  'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task';
const OBJECT_MODEL_URL =
  'https://storage.googleapis.com/mediapipe-tasks/object_detector/efficientdet_lite0_uint8.tflite';

const ALERT_COOLDOWN_MS     = 15000;
const NO_FACE_THRESHOLD_MS  = 4000;
const LIVE_INTERVAL_MS      = 200;
const MULTI_FACE_CONFIRM_MS = 1500;
const GAZE_CONFIRM_MS       = 1000;
const GAZE_BLENDSHAPE_THRESHOLD = 0.8;

const DEFAULT_GAZE_THRESHOLD_X = 0.32;
const DEFAULT_GAZE_THRESHOLD_Y = 0.34;

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
 *   1. edge_peek   — a non-whitelisted object's box clips ≥ 1 frame edge AND
 *                    covers PEEK_MIN_AREA … PEEK_MAX_AREA of the frame.
 *                    This is the primary scenario the user described:
 *                    "phone edge barely visible at the side of the screen."
 *
 *   2. occluding   — a non-whitelisted object sits in the lower 40 % of the
 *                    frame (below OCCLUDE_Y_THRESHOLD) and covers ≥ OCCLUDE_MIN_AREA.
 *                    This catches objects placed on a desk between the candidate
 *                    and the camera that are not fully peeking from an edge.
 *
 *   3. partial_edge — a very small box (< PEEK_MIN_AREA × 3) that clips an edge
 *                    with no recognisable label.  Requires a longer sustained
 *                    presence (handled by FOREIGN_CONFIRM_MS_SMALL) to avoid
 *                    noise from detection artefacts at frame boundaries.
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

    // ── Trigger 1: edge_peek ──────────────────────────────────────────────
    if (clipsEdge && areaRatio >= PEEK_MIN_AREA && areaRatio < PEEK_MAX_AREA) {
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

    // ── Trigger 2: occluding ─────────────────────────────────────────────
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

    // ── Trigger 3: partial_edge ───────────────────────────────────────────
    // Very small box that clips an edge — could be a sliver of a hidden phone.
    // We require an edge clip; without it the box is too ambiguous.
    if (clipsEdge && areaRatio >= PEEK_MIN_AREA && areaRatio < PEEK_MIN_AREA * 3) {
      const computed = Math.min(detScore * 0.75, 1.0); // conservative
      if (computed > bestScore) {
        bestScore     = computed;
        bestLabel     = detLabel;
        bestTrigger   = 'partial_edge';
        bestAreaRatio = areaRatio;
        bestEdgeSides = edgeSides;
      }
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

// ─────────────────────────────────────────────────────────────────────────────
// Gaze helpers (unchanged)
// ─────────────────────────────────────────────────────────────────────────────

function getFacePoint(
  landmarks: FaceLandmarkerResult['faceLandmarks'][number] | undefined,
  index: number,
) {
  return landmarks?.[index];
}

function detectGazeAway(
  result: FaceLandmarkerResult | null,
  thresholdX = DEFAULT_GAZE_THRESHOLD_X,
  thresholdY = DEFAULT_GAZE_THRESHOLD_Y,
  neutralX = 0,
  neutralY = 0,
  calibration?: CalibrationResult | null,
  filterRef?: { current: { x: number; y: number; initialized: boolean } } | null,
  smoothingAlpha = 0.25,
) {
  const faceLandmarks   = result?.faceLandmarks?.[0];
  const faceBlendshapes = result?.faceBlendshapes?.[0]?.categories ?? [];

  const getBlendshapeScore = (name: string) =>
    faceBlendshapes
      .find(
        (category) =>
          (category.categoryName || category.displayName || '').toLowerCase() === name.toLowerCase(),
      )
      ?.score ?? 0;

  const upLeft    = getBlendshapeScore('eyeLookUpLeft');
  const upRight   = getBlendshapeScore('eyeLookUpRight');
  const downLeft  = getBlendshapeScore('eyeLookDownLeft');
  const downRight = getBlendshapeScore('eyeLookDownRight');
  const outLeft   = getBlendshapeScore('eyeLookOutLeft');
  const outRight  = getBlendshapeScore('eyeLookOutRight');
  const inLeft    = getBlendshapeScore('eyeLookInLeft');
  const inRight   = getBlendshapeScore('eyeLookInRight');

  const directionScores = [
    { direction: 'up',    score: Math.min(upLeft,   upRight) },
    { direction: 'down',  score: Math.min(downLeft, downRight) },
    { direction: 'left',  score: Math.min(outLeft,  inRight) },
    { direction: 'right', score: Math.min(inLeft,   outRight) },
  ];

  const bestBlendshape = directionScores
    .slice()
    .sort((a, b) => b.score - a.score)[0];

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
  const leftH  = Math.max(Math.abs((lUp?.y  ?? leftIris.y)  - (lDown?.y ?? leftIris.y)),  0.0001);
  const rightH = Math.max(Math.abs((rUp?.y  ?? rightIris.y) - (rDown?.y ?? rightIris.y)), 0.0001);

  const leftMidX  = (leftEyeOuter.x  + leftEyeInner.x)  / 2;
  const rightMidX = (rightEyeOuter.x + rightEyeInner.x) / 2;
  const leftMidY  = ((lUp?.y  ?? leftIris.y)  + (lDown?.y ?? leftIris.y))  / 2;
  const rightMidY = ((rUp?.y  ?? rightIris.y) + (rDown?.y ?? rightIris.y)) / 2;

  const rawOffsetX = ((leftIris.x - leftMidX) / (leftW / 2) + (rightIris.x - rightMidX) / (rightW / 2)) / 2;
  const rawOffsetY = ((leftIris.y - leftMidY) / (leftH / 2) + (rightIris.y - rightMidY) / (rightH / 2)) / 2;

  const adjOffsetX = rawOffsetX - neutralX;
  const adjOffsetY = rawOffsetY - neutralY;

  let effectiveThresholdX = thresholdX;
  let effectiveThresholdY = thresholdY;

  if (calibration?.pointData && calibration.pointData.length > 0) {
    const centerPoint = calibration.pointData.find((p) => p.id === 'mc');
    if (centerPoint && centerPoint.samples.length > 0) {
      const centerFactor  = 0.95;
      effectiveThresholdX = calibration.thresholdX * centerFactor;
      effectiveThresholdY = calibration.thresholdY * centerFactor;
    }
  }

  let useX = adjOffsetX;
  let useY = adjOffsetY;
  if (filterRef) {
    const f = filterRef.current;
    if (!f.initialized) {
      f.x = adjOffsetX; f.y = adjOffsetY; f.initialized = true;
      useX = adjOffsetX; useY = adjOffsetY;
    } else {
      f.x  = f.x * (1 - smoothingAlpha) + adjOffsetX * smoothingAlpha;
      f.y  = f.y * (1 - smoothingAlpha) + adjOffsetY * smoothingAlpha;
      useX = f.x; useY = f.y;
    }
  }

  if (Math.abs(useX) >= effectiveThresholdX || Math.abs(useY) >= effectiveThresholdY) {
    const horizontal = useX > 0 ? 'left' : 'right';
    const vertical   = useY < 0 ? 'up'   : 'down';
    const direction  = Math.abs(useX) > Math.abs(useY) ? horizontal : vertical;
    return {
      away: true,
      direction,
      confidence: Math.max(Math.abs(useX), Math.abs(useY)),
      source: 'geometry' as const,
    };
  }

  return { away: false, direction: 'center', confidence: 0, source: 'geometry' as const };
}

// ─────────────────────────────────────────────────────────────────────────────
// Hook
// ─────────────────────────────────────────────────────────────────────────────

export function useProctoring(
  sessionId: string,
  socket?: WebSocket | null,
  calibration?: CalibrationResult | null,
) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [events, setEvents] = useState<ProctoringEvent[]>([]);
  const [state, setState] = useState<DetectionState>({
    initialized:           false,
    status:                'Initializing camera...',
    permissionDenied:      false,
    cameraActive:          false,
    faceDetectorActive:    false,
    objectDetectorActive:  false,
    faceCount:             0,
    phoneDetected:         false,
    phoneProximity:        'far',
    phoneOrientation:      'unknown',
    foreignObjectDetected: false,        // ← NEW
    foreignObjectLabel:    '',           // ← NEW
    gazeAwayDetected:      false,
    gazeDirection:         'center',
    lastObservationAt:     null,
  });

  // ── Refs ──────────────────────────────────────────────────────────────────
  const missingSince          = useRef<number | null>(null);
  const faceAlertAt           = useRef<number>(0);
  const phoneAlertAt          = useRef<number>(0);
  const gazeAlertAt           = useRef<number>(0);
  const multiFaceAlertAt      = useRef<number>(0);
  const foreignObjectAlertAt  = useRef<number>(0);   // ← NEW
  const multiFaceSince        = useRef<number | null>(null);
  const gazeSince             = useRef<number | null>(null);
  const faceLandmarkerRef     = useRef<FaceLandmarker | null>(null);
  const objectDetectorRef     = useRef<ObjectDetector | null>(null);
  const streamRef             = useRef<MediaStream | null>(null);
  const frameTimerRef         = useRef<number | null>(null);
  const aliveRef              = useRef(true);
  const calibrationRef        = useRef<CalibrationResult | null>(calibration ?? null);
  const gazeFilterRef         = useRef<{ x: number; y: number; initialized: boolean }>({
    x: 0, y: 0, initialized: false,
  });
  const phoneStateRef         = useRef<PhoneState>(createPhoneState());
  const foreignObjectStateRef = useRef<ForeignObjectState>(createForeignObjectState()); // ← NEW

  useEffect(() => {
    calibrationRef.current = calibration ?? null;
  }, [calibration]);

  // ── Event helpers ─────────────────────────────────────────────────────────

  function emit(eventType: string, severity: Severity, metadata: Record<string, unknown> = {}) {
    const payload: ProctoringPayload = {
      type: 'proctoring_event',
      sessionId,
      eventType,
      severity,
      metadata,
      timestamp: Date.now(),
    };
    socket?.readyState === 1 && socket.send(JSON.stringify(payload));
    setEvents((current) =>
      [{ eventType, severity, timestamp: Date.now(), metadata }, ...current].slice(0, 10),
    );
  }

  function emitWithCooldown(
    ref: { current: number },
    eventType: string,
    severity: Severity,
    metadata: Record<string, unknown> = {},
  ) {
    const now = Date.now();
    if (now - ref.current < ALERT_COOLDOWN_MS) return;
    ref.current = now;
    emit(eventType, severity, metadata);
  }

  // ── Main effect ───────────────────────────────────────────────────────────

  useEffect(() => {
    aliveRef.current = true;

    async function start() {
      setState({
        initialized:           false,
        status:                'Requesting camera access...',
        permissionDenied:      false,
        cameraActive:          false,
        faceDetectorActive:    false,
        objectDetectorActive:  false,
        faceCount:             0,
        phoneDetected:         false,
        phoneProximity:        'far',
        phoneOrientation:      'unknown',
        foreignObjectDetected: false,   // ← NEW
        foreignObjectLabel:    '',      // ← NEW
        gazeAwayDetected:      false,
        gazeDirection:         'center',
        lastObservationAt:     null,
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
        initialized:      true,
        status:           'Detection active',
        permissionDenied: false,
        cameraActive:     true,
      }));

      const vision = await FilesetResolver.forVisionTasks(
        'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm',
      );

      const [faceLandmarker, objectDetector] = await Promise.all([
        FaceLandmarker.createFromOptions(vision, {
          baseOptions:                { modelAssetPath: FACE_MODEL_URL },
          runningMode:                'VIDEO',
          numFaces:                   4,
          minFaceDetectionConfidence: 0.55,
          minFacePresenceConfidence:  0.55,
          minTrackingConfidence:      0.55,
          outputFaceBlendshapes:      true,
        }),
        ObjectDetector.createFromOptions(vision, {
          baseOptions:    { modelAssetPath: OBJECT_MODEL_URL },
          runningMode:    'VIDEO',
          scoreThreshold: 0.25,   // ← CHANGED: 0.30 → 0.25 to catch low-confidence peek detections
          maxResults:     8,      // ← CHANGED: 5 → 8 to surface more partial/occluded detections
        }),
      ]);

      faceLandmarkerRef.current  = faceLandmarker;
      objectDetectorRef.current  = objectDetector;

      setState((current) => ({
        ...current,
        faceDetectorActive:   true,
        objectDetectorActive: true,
      }));

      const tick = () => {
        if (!aliveRef.current) return;

        const currentVideo = videoRef.current;
        const faceTask     = faceLandmarkerRef.current;
        const objectTask   = objectDetectorRef.current;

        if (!currentVideo || !faceTask || !objectTask || currentVideo.readyState < 2) {
          if (!missingSince.current) missingSince.current = Date.now();
          if (Date.now() - (missingSince.current || 0) > NO_FACE_THRESHOLD_MS) {
            emitWithCooldown(faceAlertAt, 'FACE_NOT_DETECTED', 'HIGH', {
              durationMs: Date.now() - (missingSince.current || 0),
            });
          }
          setState((current) => ({
            ...current,
            cameraActive:         !!currentVideo?.srcObject,
            faceDetectorActive:   !!faceTask,
            objectDetectorActive: !!objectTask,
            faceCount:            0,
            phoneDetected:        false,
            phoneProximity:       'far',
            phoneOrientation:     'unknown',
            foreignObjectDetected: false,   // ← NEW
            foreignObjectLabel:    '',      // ← NEW
            lastObservationAt:    Date.now(),
          }));
          frameTimerRef.current = window.setTimeout(tick, LIVE_INTERVAL_MS);
          return;
        }

        missingSince.current = null;
        const timestamp = performance.now();

        // ── Run detectors ─────────────────────────────────────────────────

        let faceResult:   FaceLandmarkerResult  | null = null;
        let objectResult: ObjectDetectorResult  | null = null;

        try {
          if (typeof (faceTask as any).detectForVideo === 'function') {
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
          if (typeof (objectTask as any).detectForVideo === 'function') {
            objectResult = (objectTask as any).detectForVideo(currentVideo, timestamp);
          } else {
            console.warn('Object task not ready or detectForVideo missing');
            setState((current) => ({ ...current, status: 'Object detector unavailable' }));
          }
        } catch (error) {
          console.error('Object detect error', error);
          setState((current) => ({ ...current, status: 'Object detector unavailable' }));
        }

        // ── Analyse results ───────────────────────────────────────────────

        const faceCount = faceResult?.faceLandmarks?.length || 0;

        const phone = objectResult
          ? analyzePhoneDetection(
              objectResult,
              currentVideo.videoWidth  || 640,
              currentVideo.videoHeight || 480,
              phoneStateRef.current,
            )
          : null;
        const detectedPhone = phone?.detected ?? false;

        // ← NEW: foreign-object analysis (runs even when phone analysis fires
        // so that non-phone foreign objects are independently reported)
        const foreignObject = objectResult
          ? analyzeForeignObject(
              objectResult,
              currentVideo.videoWidth  || 640,
              currentVideo.videoHeight || 480,
              foreignObjectStateRef.current,
            )
          : null;
        const detectedForeign = foreignObject?.detected ?? false;

        const gaze = detectGazeAway(
          faceResult,
          calibrationRef.current?.thresholdX ?? DEFAULT_GAZE_THRESHOLD_X,
          calibrationRef.current?.thresholdY ?? DEFAULT_GAZE_THRESHOLD_Y,
          calibrationRef.current?.neutralX   ?? 0,
          calibrationRef.current?.neutralY   ?? 0,
          calibrationRef.current,
          gazeFilterRef,
          0.28,
        );

        // ── Update UI state ───────────────────────────────────────────────

        setState((current) => ({
          ...current,
          cameraActive:         true,
          faceDetectorActive:   true,
          objectDetectorActive: true,
          faceCount,
          phoneDetected:        detectedPhone,
          phoneProximity:       phone?.proximity   ?? 'far',
          phoneOrientation:     phone?.orientation ?? 'unknown',
          foreignObjectDetected: detectedForeign,                      // ← NEW
          foreignObjectLabel:    foreignObject?.label ?? '',           // ← NEW
          gazeAwayDetected:     gaze.away,
          gazeDirection:        gaze.direction,
          lastObservationAt:    Date.now(),
          status:
            faceCount > 1
              ? 'Multiple faces detected'
              : detectedPhone
              ? `Phone detected (${phone?.proximity ?? ''}, ${phone?.orientation ?? ''})`
              : detectedForeign
              ? `Foreign object detected (${foreignObject?.trigger ?? ''}: ${foreignObject?.label ?? ''})`  // ← NEW
              : gaze.away
              ? `Looking away (${gaze.direction})`
              : 'Detection active',
        }));

        // ── Emit alerts ───────────────────────────────────────────────────

        // Face missing
        if (faceCount === 0) {
          if (!missingSince.current) missingSince.current = Date.now();
          if (Date.now() - (missingSince.current || 0) > NO_FACE_THRESHOLD_MS) {
            emitWithCooldown(faceAlertAt, 'FACE_NOT_DETECTED', 'HIGH', {
              durationMs: Date.now() - (missingSince.current || 0),
            });
          }
        } else {
          missingSince.current = null;
        }

        // Multiple faces
        if (faceCount > 1) {
          if (!multiFaceSince.current) multiFaceSince.current = Date.now();
          if (Date.now() - (multiFaceSince.current || 0) > MULTI_FACE_CONFIRM_MS) {
            emitWithCooldown(multiFaceAlertAt, 'MULTIPLE_FACES_DETECTED', 'HIGH', {
              faceCount,
              faces: faceCount,
            });
            multiFaceSince.current = null;
          }
        } else {
          multiFaceSince.current = null;
        }

        // Phone — adaptive confirmation via consumePhoneAlert
        if (phone && consumePhoneAlert(phone, phoneStateRef.current)) {
          emitWithCooldown(phoneAlertAt, 'MOBILE_PHONE_DETECTED', 'HIGH', {
            proximity:   phone.proximity,
            orientation: phone.orientation,
            source:      phone.source,
            confidence:  phone.rollingConfidence,
            detections:  phone.detections,
          });
        }

        // ← NEW: Foreign object — adaptive confirmation via consumeForeignObjectAlert
        if (
          foreignObject &&
          consumeForeignObjectAlert(foreignObject, foreignObjectStateRef.current)
        ) {
          emitWithCooldown(foreignObjectAlertAt, 'FOREIGN_OBJECT_DETECTED', 'HIGH', {
            label:      foreignObject.label,
            trigger:    foreignObject.trigger,
            confidence: foreignObject.confidence,
            areaRatio:  foreignObject.areaRatio,
            edgeSides:  foreignObject.edgeSides,
          });
        }

        // Gaze away
        if (gaze.away) {
          if (!gazeSince.current) gazeSince.current = Date.now();
          if (Date.now() - (gazeSince.current || 0) > GAZE_CONFIRM_MS) {
            emitWithCooldown(gazeAlertAt, 'GAZE_AWAY_DETECTED', 'MEDIUM', {
              direction:  gaze.direction,
              confidence: gaze.confidence,
              source:     gaze.source,
            });
            gazeSince.current = null;
          }
        } else {
          gazeSince.current = null;
        }

        frameTimerRef.current = window.setTimeout(tick, LIVE_INTERVAL_MS);
      };

      frameTimerRef.current = window.setTimeout(tick, LIVE_INTERVAL_MS);
    }

    start().catch((error) => {
      setState({
        initialized:           false,
        status:                'Camera permission denied',
        permissionDenied:      true,
        cameraActive:          false,
        faceDetectorActive:    false,
        objectDetectorActive:  false,
        faceCount:             0,
        phoneDetected:         false,
        phoneProximity:        'far',
        phoneOrientation:      'unknown',
        foreignObjectDetected: false,   // ← NEW
        foreignObjectLabel:    '',      // ← NEW
        gazeAwayDetected:      false,
        gazeDirection:         'center',
        lastObservationAt:     null,
      });
      emit('CAMERA_PERMISSION_DENIED', 'HIGH', {
        message: error instanceof Error ? error.message : 'getUserMedia failed',
      });
    });

    return () => {
      aliveRef.current = false;
      if (frameTimerRef.current) window.clearTimeout(frameTimerRef.current);
      streamRef.current?.getTracks().forEach((track) => track.stop());
      faceLandmarkerRef.current?.close(); 
      objectDetectorRef.current?.close();
      faceLandmarkerRef.current  = null;
      objectDetectorRef.current  = null;
      streamRef.current          = null;
    };
  }, [sessionId, socket]);

  return { videoRef, events, emit, state };
}