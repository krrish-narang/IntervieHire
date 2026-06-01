import {
  CALIBRATION_THRESHOLD_SAFETY_FACTOR,
  FALLBACK_GAZE_THRESHOLD_X,
  FALLBACK_GAZE_THRESHOLD_Y,
  MAX_CALIBRATION_DOT_STD_X,
  MAX_CALIBRATION_DOT_STD_Y,
  MAX_SAFE_GAZE_THRESHOLD_X,
  MAX_SAFE_GAZE_THRESHOLD_Y,
  MAX_SAFE_NEUTRAL_X,
  MAX_SAFE_NEUTRAL_Y,
  MIN_DOT_DIRECTION_SEPARATION_X,
  MIN_DOT_DIRECTION_SEPARATION_Y,
  MIN_OPPOSITE_EDGE_GAP_X,
  MIN_OPPOSITE_EDGE_GAP_Y,
  MIN_SAFE_GAZE_THRESHOLD_X,
  MIN_SAFE_GAZE_THRESHOLD_Y,
  MIN_SAMPLES_PER_CALIBRATION_DOT,
} from './proctoringGazeThresholdsV3';

type UnknownRecord = Record<string, unknown>;

type CalibrationSampleLike = UnknownRecord;

type CalibrationPointLike = UnknownRecord & {
  id?: string;
  samples?: CalibrationSampleLike[];
};

type CalibrationLike = UnknownRecord & {
  thresholdX?: number;
  thresholdY?: number;
  neutralX?: number;
  neutralY?: number;
  pointData?: CalibrationPointLike[];
};

type DotAxisDirection = 'left' | 'center' | 'right' | 'up' | 'down';

type DotExpectation = {
  id: string;
  x: DotAxisDirection;
  y: DotAxisDirection;
};

type DotStats = {
  id: string;
  sampleCount: number;
  meanX: number;
  meanY: number;
  stdX: number;
  stdY: number;
};

export type EightDotCalibrationValidationResult = {
  accepted: boolean;
  reason: string;
  missingPointIds: string[];
  dotStats: DotStats[];
};

export type SafeEightDotCalibration = {
  thresholdX: number;
  thresholdY: number;
  neutralX: number;
  neutralY: number;
  trusted: boolean;
  reason: string;
  missingPointIds: string[];
  validation: EightDotCalibrationValidationResult;
};

// 8 fixed perimeter dots. No randomized dots are required.
// Center ('mc') is optional. If present, it is used as a stronger neutral check.
const EIGHT_DOT_EXPECTATIONS: DotExpectation[] = [
  { id: 'tl', x: 'left', y: 'up' },
  { id: 'tc', x: 'center', y: 'up' },
  { id: 'tr', x: 'right', y: 'up' },
  { id: 'ml', x: 'left', y: 'center' },
  { id: 'mr', x: 'right', y: 'center' },
  { id: 'bl', x: 'left', y: 'down' },
  { id: 'bc', x: 'center', y: 'down' },
  { id: 'br', x: 'right', y: 'down' },
];

const POINT_ID_ALIASES: Record<string, string> = {
  'top-left': 'tl',
  topleft: 'tl',
  top_left: 'tl',
  tl: 'tl',

  'top-center': 'tc',
  topcenter: 'tc',
  top_center: 'tc',
  tm: 'tc',
  tc: 'tc',

  'top-right': 'tr',
  topright: 'tr',
  top_right: 'tr',
  tr: 'tr',

  'middle-left': 'ml',
  middleleft: 'ml',
  middle_left: 'ml',
  'mid-left': 'ml',
  midleft: 'ml',
  ml: 'ml',

  'middle-center': 'mc',
  middlecenter: 'mc',
  middle_center: 'mc',
  'mid-center': 'mc',
  midcenter: 'mc',
  center: 'mc',
  c: 'mc',
  mc: 'mc',

  'middle-right': 'mr',
  middleright: 'mr',
  middle_right: 'mr',
  'mid-right': 'mr',
  midright: 'mr',
  mr: 'mr',

  'bottom-left': 'bl',
  bottomleft: 'bl',
  bottom_left: 'bl',
  bl: 'bl',

  'bottom-center': 'bc',
  bottomcenter: 'bc',
  bottom_center: 'bc',
  bm: 'bc',
  bc: 'bc',

  'bottom-right': 'br',
  bottomright: 'br',
  bottom_right: 'br',
  br: 'br',
};

const SAMPLE_X_KEYS = [
  'offsetX',
  'gazeX',
  'rawOffsetX',
  'adjOffsetX',
  'irisOffsetX',
  'eyeOffsetX',
  'x',
];

const SAMPLE_Y_KEYS = [
  'offsetY',
  'gazeY',
  'rawOffsetY',
  'adjOffsetY',
  'irisOffsetY',
  'eyeOffsetY',
  'y',
];

const POINT_MEAN_X_KEYS = ['meanX', 'avgX', 'averageX', 'medianX', 'offsetX', 'gazeX', 'rawOffsetX'];
const POINT_MEAN_Y_KEYS = ['meanY', 'avgY', 'averageY', 'medianY', 'offsetY', 'gazeY', 'rawOffsetY'];

function canonicalPointId(id: unknown) {
  if (typeof id !== 'string') return '';
  const normalized = id.trim().toLowerCase().replace(/\s+/g, '-');
  return POINT_ID_ALIASES[normalized] ?? POINT_ID_ALIASES[normalized.replace(/-/g, '')] ?? normalized;
}

function finiteNumber(value: unknown): number | null {
  return typeof value === 'number' && Number.isFinite(value) ? value : null;
}

function firstFiniteNumber(record: UnknownRecord, keys: string[]): number | null {
  for (const key of keys) {
    const value = finiteNumber(record[key]);
    if (value !== null) return value;
  }
  return null;
}

function mean(values: number[]) {
  return values.reduce((sum, value) => sum + value, 0) / Math.max(values.length, 1);
}

function standardDeviation(values: number[], average: number) {
  if (values.length <= 1) return 0;
  const variance = values.reduce((sum, value) => sum + Math.pow(value - average, 2), 0) / values.length;
  return Math.sqrt(variance);
}

function clamp(value: number, min: number, max: number) {
  return Math.min(Math.max(value, min), max);
}

function buildPointLookup(points: CalibrationPointLike[] = []) {
  const lookup = new Map<string, CalibrationPointLike>();
  for (const point of points) {
    const id = canonicalPointId(point.id);
    if (id && !lookup.has(id)) lookup.set(id, point);
  }
  return lookup;
}

function getDotStats(point: CalibrationPointLike | undefined, id: string): DotStats | null {
  if (!point) return null;

  const samples = Array.isArray(point.samples) ? point.samples : [];
  const xs: number[] = [];
  const ys: number[] = [];

  for (const sample of samples) {
    const sampleX = firstFiniteNumber(sample, SAMPLE_X_KEYS);
    const sampleY = firstFiniteNumber(sample, SAMPLE_Y_KEYS);

    if (sampleX !== null && sampleY !== null) {
      xs.push(sampleX);
      ys.push(sampleY);
    }
  }

  if (xs.length > 0 && ys.length > 0) {
    const meanX = mean(xs);
    const meanY = mean(ys);
    return {
      id,
      sampleCount: Math.min(xs.length, ys.length),
      meanX,
      meanY,
      stdX: standardDeviation(xs, meanX),
      stdY: standardDeviation(ys, meanY),
    };
  }

  // Fallback for calibration implementations that store aggregate values on the point itself.
  // We intentionally do not read generic point.x / point.y here because those are often screen coordinates.
  const pointMeanX = firstFiniteNumber(point, POINT_MEAN_X_KEYS);
  const pointMeanY = firstFiniteNumber(point, POINT_MEAN_Y_KEYS);
  if (pointMeanX !== null && pointMeanY !== null) {
    return {
      id,
      sampleCount: samples.length || MIN_SAMPLES_PER_CALIBRATION_DOT,
      meanX: pointMeanX,
      meanY: pointMeanY,
      stdX: 0,
      stdY: 0,
    };
  }

  return null;
}

function validateDotQuality(stats: DotStats[]) {
  for (const stat of stats) {
    if (stat.sampleCount < MIN_SAMPLES_PER_CALIBRATION_DOT) {
      return `Calibration dot ${stat.id} has too few valid samples`;
    }

    if (stat.stdX > MAX_CALIBRATION_DOT_STD_X || stat.stdY > MAX_CALIBRATION_DOT_STD_Y) {
      return `Calibration dot ${stat.id} was unstable`;
    }
  }

  return null;
}

function validateNeutral(neutralX: number, neutralY: number, centerStats: DotStats | null) {
  if (Math.abs(neutralX) > MAX_SAFE_NEUTRAL_X || Math.abs(neutralY) > MAX_SAFE_NEUTRAL_Y) {
    return 'Neutral gaze is too far from center';
  }

  if (!centerStats) return null;

  if (Math.abs(centerStats.meanX - neutralX) > MAX_SAFE_NEUTRAL_X || Math.abs(centerStats.meanY - neutralY) > MAX_SAFE_NEUTRAL_Y) {
    return 'Center calibration does not match neutral gaze';
  }

  return null;
}

function validateDotGeometry(statsById: Map<string, DotStats>, centerX: number, centerY: number) {
  for (const expected of EIGHT_DOT_EXPECTATIONS) {
    const stat = statsById.get(expected.id);
    if (!stat) return `Missing calibration dot ${expected.id}`;

    if (expected.x === 'left' && stat.meanX <= centerX + MIN_DOT_DIRECTION_SEPARATION_X) {
      return `Calibration dot ${expected.id} does not look left enough`;
    }

    if (expected.x === 'right' && stat.meanX >= centerX - MIN_DOT_DIRECTION_SEPARATION_X) {
      return `Calibration dot ${expected.id} does not look right enough`;
    }

    if (expected.y === 'up' && stat.meanY >= centerY - MIN_DOT_DIRECTION_SEPARATION_Y) {
      return `Calibration dot ${expected.id} does not look up enough`;
    }

    if (expected.y === 'down' && stat.meanY <= centerY + MIN_DOT_DIRECTION_SEPARATION_Y) {
      return `Calibration dot ${expected.id} does not look down enough`;
    }
  }

  const leftAverage = mean(['tl', 'ml', 'bl'].map((id) => statsById.get(id)?.meanX ?? centerX));
  const rightAverage = mean(['tr', 'mr', 'br'].map((id) => statsById.get(id)?.meanX ?? centerX));
  const topAverage = mean(['tl', 'tc', 'tr'].map((id) => statsById.get(id)?.meanY ?? centerY));
  const bottomAverage = mean(['bl', 'bc', 'br'].map((id) => statsById.get(id)?.meanY ?? centerY));

  // In this hook's geometry convention: positive X = looking left, positive Y = looking down.
  if (leftAverage - rightAverage < MIN_OPPOSITE_EDGE_GAP_X) {
    return 'Left and right calibration dots are not separated enough';
  }

  if (bottomAverage - topAverage < MIN_OPPOSITE_EDGE_GAP_Y) {
    return 'Top and bottom calibration dots are not separated enough';
  }

  return null;
}

export function validateEightDotCalibration(calibration: CalibrationLike | null | undefined): EightDotCalibrationValidationResult {
  if (!calibration?.pointData || calibration.pointData.length === 0) {
    return {
      accepted: false,
      reason: 'No calibration point data available',
      missingPointIds: EIGHT_DOT_EXPECTATIONS.map((point) => point.id),
      dotStats: [],
    };
  }

  const pointLookup = buildPointLookup(calibration.pointData);
  const missingPointIds = EIGHT_DOT_EXPECTATIONS
    .map((point) => point.id)
    .filter((id) => !pointLookup.has(id));

  const dotStats = EIGHT_DOT_EXPECTATIONS
    .map((point) => getDotStats(pointLookup.get(point.id), point.id))
    .filter((stats): stats is DotStats => Boolean(stats));

  if (missingPointIds.length > 0) {
    return {
      accepted: false,
      reason: `Missing required 8-dot calibration points: ${missingPointIds.join(', ')}`,
      missingPointIds,
      dotStats,
    };
  }

  const qualityFailure = validateDotQuality(dotStats);
  if (qualityFailure) {
    return {
      accepted: false,
      reason: qualityFailure,
      missingPointIds: [],
      dotStats,
    };
  }

  const centerStats = getDotStats(pointLookup.get('mc'), 'mc');
  const neutralX = finiteNumber(calibration.neutralX) ?? centerStats?.meanX ?? 0;
  const neutralY = finiteNumber(calibration.neutralY) ?? centerStats?.meanY ?? 0;

  const neutralFailure = validateNeutral(neutralX, neutralY, centerStats);
  if (neutralFailure) {
    return {
      accepted: false,
      reason: neutralFailure,
      missingPointIds: [],
      dotStats,
    };
  }

  const statsById = new Map(dotStats.map((stat) => [stat.id, stat]));
  const centerX = centerStats?.meanX ?? neutralX;
  const centerY = centerStats?.meanY ?? neutralY;
  const geometryFailure = validateDotGeometry(statsById, centerX, centerY);

  if (geometryFailure) {
    return {
      accepted: false,
      reason: geometryFailure,
      missingPointIds: [],
      dotStats,
    };
  }

  return {
    accepted: true,
    reason: '8-dot calibration accepted',
    missingPointIds: [],
    dotStats,
  };
}

export function buildSafeEightDotCalibration(calibration: CalibrationLike | null | undefined): SafeEightDotCalibration {
  const validation = validateEightDotCalibration(calibration);

  if (!calibration) {
    return {
      thresholdX: FALLBACK_GAZE_THRESHOLD_X,
      thresholdY: FALLBACK_GAZE_THRESHOLD_Y,
      neutralX: 0,
      neutralY: 0,
      trusted: false,
      reason: validation.reason,
      missingPointIds: validation.missingPointIds,
      validation,
    };
  }

  const rawThresholdX = finiteNumber(calibration.thresholdX) ?? FALLBACK_GAZE_THRESHOLD_X;
  const rawThresholdY = finiteNumber(calibration.thresholdY) ?? FALLBACK_GAZE_THRESHOLD_Y;
  const rawNeutralX = finiteNumber(calibration.neutralX) ?? 0;
  const rawNeutralY = finiteNumber(calibration.neutralY) ?? 0;

  // Critical fix:
  // Do NOT disable live gaze detection just because the 8-dot validation failed.
  // The previous version fell back to broad default thresholds when validation failed,
  // which made gaze-away detection too insensitive. Instead, always use the current
  // calibration values after sanitizing them with hard caps. The validation result is
  // only exposed as metadata through `trusted` and `reason`.
  return {
    thresholdX: clamp(
      rawThresholdX * CALIBRATION_THRESHOLD_SAFETY_FACTOR,
      MIN_SAFE_GAZE_THRESHOLD_X,
      MAX_SAFE_GAZE_THRESHOLD_X,
    ),
    thresholdY: clamp(
      rawThresholdY * CALIBRATION_THRESHOLD_SAFETY_FACTOR,
      MIN_SAFE_GAZE_THRESHOLD_Y,
      MAX_SAFE_GAZE_THRESHOLD_Y,
    ),
    neutralX: clamp(rawNeutralX, -MAX_SAFE_NEUTRAL_X, MAX_SAFE_NEUTRAL_X),
    neutralY: clamp(rawNeutralY, -MAX_SAFE_NEUTRAL_Y, MAX_SAFE_NEUTRAL_Y),
    trusted: validation.accepted,
    reason: validation.accepted ? validation.reason : `Calibration sanitized: ${validation.reason}`,
    missingPointIds: validation.missingPointIds,
    validation,
  };
}
