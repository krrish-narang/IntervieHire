// Anti-spoofing thresholds for 8-dot gaze calibration and live gaze detection.
// Important: these values cap unsafe calibration expansion, but they should not make
// live gaze detection insensitive. Calibration can make thresholds smaller, but not huge.

export const FALLBACK_GAZE_THRESHOLD_X = 0.32;
export const FALLBACK_GAZE_THRESHOLD_Y = 0.34;

// Keep the minimum low. A high minimum makes gaze-away impossible for users whose
// calibrated eye-offset range is naturally small.
export const MIN_SAFE_GAZE_THRESHOLD_X = 0.08;
export const MIN_SAFE_GAZE_THRESHOLD_Y = 0.08;

// Hard maximums: these are the anti-cheat caps. Even if a user looks outside the
// screen during calibration, the live valid region cannot grow past these values.
export const MAX_SAFE_GAZE_THRESHOLD_X = 0.34;
export const MAX_SAFE_GAZE_THRESHOLD_Y = 0.36;

// Use almost the calibrated threshold. The anti-cheat protection is mainly the
// maximum cap above, not shrinking the threshold so aggressively that detection breaks.
export const CALIBRATION_THRESHOLD_SAFETY_FACTOR = 0.95;

// If neutral is too far from raw eye center, clamp it so fake neutral calibration
// cannot hide large gaze offsets.
export const MAX_SAFE_NEUTRAL_X = 0.22;
export const MAX_SAFE_NEUTRAL_Y = 0.22;

// Per-dot quality checks. These validate the 8-dot calibration but do not disable
// live gaze detection. Live detection still runs with sanitized thresholds.
export const MIN_SAMPLES_PER_CALIBRATION_DOT = 8;
export const MAX_CALIBRATION_DOT_STD_X = 0.24;
export const MAX_CALIBRATION_DOT_STD_Y = 0.26;

// Expected 8-dot geometry separation from neutral/center.
export const MIN_DOT_DIRECTION_SEPARATION_X = 0.06;
export const MIN_DOT_DIRECTION_SEPARATION_Y = 0.06;
export const MIN_OPPOSITE_EDGE_GAP_X = 0.14;
export const MIN_OPPOSITE_EDGE_GAP_Y = 0.14;
