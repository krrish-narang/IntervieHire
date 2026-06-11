import type {
  CoverageStatus,
  EvalConfidence,
  EvalPointCoverage,
  EvalRedFlagSeverity,
  EvalResponseEvaluation,
} from "./types";
import { clampEvalScore } from "./scoring";

export interface EvalValidationResult {
  valid: boolean;
  issues: string[];
  normalized: EvalResponseEvaluation;
}

const CONFIDENCE_VALUES: EvalConfidence[] = ["high", "medium", "low"];
const SEVERITY_VALUES: EvalRedFlagSeverity[] = ["low", "medium", "high", "critical"];
const COVERAGE_STATUS_VALUES: CoverageStatus[] = ["full", "partial", "missing", "contradicted"];

export function validateEvalResponseEvaluation(
  evaluation: EvalResponseEvaluation,
): EvalValidationResult {
  const normalized = normalizeEvalResponseEvaluation(evaluation);
  const issues: string[] = [];

  if (!normalized.answerId) {
    issues.push("Missing answerId.");
  }

  if (!normalized.questionId) {
    issues.push("Missing questionId.");
  }

  if (!normalized.modelAnswerComparison) {
    issues.push("Missing modelAnswerComparison.");
  }

  if (Object.keys(normalized.dimensionScores).length === 0) {
    issues.push("At least one dimension score is required.");
  }

  if (!CONFIDENCE_VALUES.includes(normalized.evaluationConfidence)) {
    issues.push("evaluationConfidence must be high, medium, or low.");
  }

  for (const [dimension, score] of Object.entries(normalized.dimensionScores)) {
    if (!score.reason) {
      issues.push(`Dimension ${dimension} is missing a reason.`);
    }
  }

  for (const flag of normalized.redFlags) {
    if (!SEVERITY_VALUES.includes(flag.severity)) {
      issues.push(`Red flag ${flag.label} has invalid severity.`);
    }
  }

  for (const claim of normalized.modelAnswerComparison.incorrectClaims) {
    if (!SEVERITY_VALUES.includes(claim.severity)) {
      issues.push(`Incorrect claim ${claim.claim} has invalid severity.`);
    }
  }

  for (const point of [
    ...normalized.modelAnswerComparison.requiredPointCoverage,
    ...normalized.modelAnswerComparison.secondaryPointCoverage,
    ...normalized.modelAnswerComparison.excellentSignalCoverage,
  ]) {
    if (!COVERAGE_STATUS_VALUES.includes(point.status)) {
      issues.push(`Rubric point ${point.pointId} has invalid coverage status.`);
    }
  }

  return {
    valid: issues.length === 0,
    issues,
    normalized,
  };
}

export function normalizeEvalResponseEvaluation(
  evaluation: EvalResponseEvaluation,
): EvalResponseEvaluation {
  const modelAnswerComparison = evaluation.modelAnswerComparison ?? {
    requiredPointCoverage: [],
    secondaryPointCoverage: [],
    excellentSignalCoverage: [],
    incorrectClaims: [],
    coverageScore: 0,
  };

  return {
    ...evaluation,
    overallScore: clampEvalScore(evaluation.overallScore),
    modelAnswerComparison: {
      ...modelAnswerComparison,
      coverageScore: clampEvalScore(modelAnswerComparison.coverageScore),
      requiredPointCoverage: normalizeCoveragePoints(modelAnswerComparison.requiredPointCoverage),
      secondaryPointCoverage: normalizeCoveragePoints(modelAnswerComparison.secondaryPointCoverage),
      excellentSignalCoverage: normalizeCoveragePoints(modelAnswerComparison.excellentSignalCoverage),
      incorrectClaims: (modelAnswerComparison.incorrectClaims ?? []).map((claim) => ({
        claim: normalizeString(claim.claim),
        severity: normalizeSeverity(claim.severity),
        correction: normalizeString(claim.correction),
      })),
    },
    dimensionScores: Object.fromEntries(
      Object.entries(evaluation.dimensionScores ?? {}).map(([dimension, score]) => [
        dimension,
        {
          ...score,
          score: clampEvalScore(score?.score),
          evidence: normalizeStringArray(score.evidence),
          missing: normalizeStringArray(score.missing),
          reason: normalizeString(score.reason),
        },
      ]),
    ),
    aiAuthorshipAssessment: evaluation.aiAuthorshipAssessment
      ? {
          ...evaluation.aiAuthorshipAssessment,
          probability: clampEvalScore(evaluation.aiAuthorshipAssessment.probability),
          confidence: normalizeConfidence(evaluation.aiAuthorshipAssessment.confidence),
          provider: evaluation.aiAuthorshipAssessment.provider === "gemini" ? "gemini" : "deepseek",
          reasons: normalizeStringArray(evaluation.aiAuthorshipAssessment.reasons).slice(0, 3),
          disclaimer: normalizeString(evaluation.aiAuthorshipAssessment.disclaimer),
        }
      : undefined,
    transcriptConfidence: evaluation.transcriptConfidence
      ? {
          ...evaluation.transcriptConfidence,
          fillerCount: normalizeCount(evaluation.transcriptConfidence.fillerCount),
          hedgeCount: normalizeCount(evaluation.transcriptConfidence.hedgeCount),
          strongUncertaintyCount: normalizeCount(evaluation.transcriptConfidence.strongUncertaintyCount),
          repeatedWordCount: normalizeCount(evaluation.transcriptConfidence.repeatedWordCount),
          totalWords: normalizeCount(evaluation.transcriptConfidence.totalWords),
          fillerRate: normalizeRate(evaluation.transcriptConfidence.fillerRate),
          confidenceScore: clampEvalScore(evaluation.transcriptConfidence.confidenceScore),
          confidenceLevel: normalizeConfidence(evaluation.transcriptConfidence.confidenceLevel),
          reliability: normalizeConfidence(evaluation.transcriptConfidence.reliability),
          confidencePenalty: normalizeCount(evaluation.transcriptConfidence.confidencePenalty),
          notes: normalizeStringArray(evaluation.transcriptConfidence.notes),
        }
      : undefined,
    strengths: normalizeStringArray(evaluation.strengths ?? []),
    weaknesses: normalizeStringArray(evaluation.weaknesses ?? []),
    redFlags: (evaluation.redFlags ?? []).map((flag) => ({
      label: normalizeString(flag.label),
      severity: normalizeSeverity(flag.severity),
      reason: normalizeString(flag.reason),
    })),
    followUpRecommendations: normalizeStringArray(evaluation.followUpRecommendations ?? []),
    evaluationConfidence: normalizeConfidence(evaluation.evaluationConfidence),
    summary: normalizeString(evaluation.summary),
    transcriptOnly: true,
  };
}

function normalizeCoveragePoints(points: EvalPointCoverage[] | undefined): EvalPointCoverage[] {
  return (points ?? []).map((point) => {
    const status = normalizeCoverageStatus(point.status);

    return {
      ...point,
      pointId: normalizeString(point.pointId),
      description: normalizeString(point.description),
      weight: Number.isFinite(point.weight) && Number(point.weight) > 0 ? Number(point.weight) : undefined,
      status,
      score: clampEvalScore(point.score),
      evidence: normalizeStringArray(point.evidence),
      comment: normalizeString(point.comment),
    };
  });
}

function normalizeCoverageStatus(status: CoverageStatus | undefined): CoverageStatus {
  return COVERAGE_STATUS_VALUES.includes(status as CoverageStatus) ? status as CoverageStatus : "missing";
}

function normalizeConfidence(confidence: EvalConfidence | undefined): EvalConfidence {
  return CONFIDENCE_VALUES.includes(confidence as EvalConfidence) ? confidence as EvalConfidence : "low";
}

function normalizeSeverity(severity: EvalRedFlagSeverity | undefined): EvalRedFlagSeverity {
  return SEVERITY_VALUES.includes(severity as EvalRedFlagSeverity) ? severity as EvalRedFlagSeverity : "medium";
}

function normalizeCount(value: unknown): number {
  const numberValue = Number(value);
  return Number.isFinite(numberValue) ? Math.max(0, Math.round(numberValue)) : 0;
}

function normalizeRate(value: unknown): number {
  const numberValue = Number(value);
  return Number.isFinite(numberValue) ? Math.max(0, Math.min(1, numberValue)) : 0;
}

function normalizeString(value: unknown): string {
  return typeof value === "string" ? value.trim() : "";
}

function normalizeStringArray(values: unknown[] | undefined): string[] {
  return (values ?? []).map(normalizeString).filter(Boolean);
}
