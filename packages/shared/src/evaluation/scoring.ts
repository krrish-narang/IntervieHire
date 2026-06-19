import type {
  EvalCandidateFacingReport,
  EvalCandidateReport,
  EvalConfidence,
  EvalInterviewContext,
  EvalMode,
  EvalQuestionConfig,
  EvalRecommendation,
  EvalResponseEvaluation,
  EvalSkillScore,
} from "./types";
import { getEvalDimensionWeights } from "./metrics";

export function inferEvalMode(question: EvalQuestionConfig): EvalMode {
  return question.questionOrigin === "generated_followup"
    ? "followup_model_answer_contextual"
    : "model_answer_based";
}

export function clampEvalScore(score: number): number {
  if (!Number.isFinite(score)) {
    return 0;
  }

  return Math.max(0, Math.min(100, Math.round(score)));
}

export function weightedEvalAverage(scores: Array<{ score: number; weight: number }>): number {
  const validScores = scores.filter((item) => item.weight > 0);
  const totalWeight = validScores.reduce((sum, item) => sum + item.weight, 0);

  if (totalWeight === 0) {
    return 0;
  }

  const totalScore = validScores.reduce(
    (sum, item) => sum + clampEvalScore(item.score) * item.weight,
    0,
  );

  return clampEvalScore(totalScore / totalWeight);
}

export function calculateDimensionScore(
  evaluation: Pick<EvalResponseEvaluation, "dimensionScores">,
  context: Pick<EvalInterviewContext, "interviewType">,
  question: EvalQuestionConfig,
): number {
  const weights = getEvalDimensionWeights(context.interviewType, question.questionType);
  const weightedScores = Object.entries(weights)
    .filter(([dimension]) => evaluation.dimensionScores[dimension])
    .map(([dimension, weight]) => ({
      score: evaluation.dimensionScores[dimension].score,
      weight,
    }));

  if (weightedScores.length === 0) {
    return weightedEvalAverage(
      Object.values(evaluation.dimensionScores).map((dimension) => ({
        score: dimension.score,
        weight: 1,
      })),
    );
  }

  return weightedEvalAverage(weightedScores);
}

export function calculateCoverageScore(
  evaluation: Pick<EvalResponseEvaluation, "modelAnswerComparison">,
): number {
  const comparison = evaluation.modelAnswerComparison;
  const required = comparison.requiredPointCoverage.map((point) => ({
    score: point.score,
    weight: point.weight ?? 3,
  }));
  const secondary = comparison.secondaryPointCoverage.map((point) => ({
    score: point.score,
    weight: point.weight ?? 1.5,
  }));
  const excellent = comparison.excellentSignalCoverage.map((point) => ({
    score: point.score,
    weight: point.weight ?? 1,
  }));

  return weightedEvalAverage([...required, ...secondary, ...excellent]);
}

export function calculateFinalAnswerScore(params: {
  evaluation: EvalResponseEvaluation;
  context: Pick<EvalInterviewContext, "interviewType">;
  question: EvalQuestionConfig;
}): number {
  const coverageScore = calculateCoverageScore(params.evaluation);
  const dimensionScore = calculateDimensionScore(
    params.evaluation,
    params.context,
    params.question,
  );
  const redFlagPenalty = calculateRedFlagPenalty(params.evaluation);

  return clampEvalScore(
    weightedEvalAverage([
      { score: coverageScore, weight: 45 },
      { score: dimensionScore, weight: 55 },
    ]) - redFlagPenalty,
  );
}

export function aggregateEvalCandidateReport(
  context: EvalInterviewContext,
  evaluations: EvalResponseEvaluation[],
): EvalCandidateReport {
  const overallScore = weightedEvalAverage(
    evaluations.map((evaluation) => ({
      score: evaluation.overallScore,
      weight: getQuestionAggregationWeight(evaluation),
    })),
  );
  const redFlags = evaluations.flatMap((evaluation) => evaluation.redFlags);
  const recommendation = getRecommendation(overallScore, redFlags);
  // Hard recommendation cutoffs are unfair on a noisy LLM score: 87.9 vs 88.0 should not flip a
  // decision with full confidence. Near a cutoff, force low confidence and route to human review.
  const borderline = isBorderlineRecommendation(overallScore);
  const recommendationConfidence = borderline ? "low" : getRecommendationConfidence(evaluations);
  const skillScores = buildSkillScores(evaluations);
  const competencyInsights = buildCompetencyInsights(skillScores);
  const candidateConfidence = buildCandidateConfidence(evaluations);
  const baseNextSteps = topUniqueStrings(
    evaluations.flatMap((evaluation) => evaluation.followUpRecommendations),
    5,
  );
  const suggestedNextSteps = borderline
    ? topUniqueStrings(
        [
          "Overall score is near a decision threshold; a human reviewer should confirm this recommendation before acting.",
          ...baseNextSteps,
        ],
        5,
      )
    : baseNextSteps;

  return {
    interviewId: context.interviewId,
    candidateId: context.candidateId,
    roleTitle: context.roleTitle,
    interviewType: context.interviewType,
    overallScore,
    recommendation,
    recommendationConfidence,
    candidateConfidence,
    summary: buildReportSummary(
      overallScore,
      recommendation,
      recommendationConfidence,
      evaluations.length,
    ),
    strengths: competencyInsights.strengths,
    weaknesses: competencyInsights.weaknesses,
    redFlags,
    skillScores,
    questionBreakdown: evaluations,
    suggestedNextSteps,
    transcriptOnly: true,
    futureSignalPlaceholders: {
      audioAnalysisEnabled: false,
      videoAnalysisEnabled: false,
    },
  };
}

function buildCandidateConfidence(
  evaluations: EvalResponseEvaluation[],
): EvalCandidateReport["candidateConfidence"] {
  const analyses = evaluations
    .map((evaluation) => evaluation.transcriptConfidence)
    .filter((analysis): analysis is NonNullable<EvalResponseEvaluation["transcriptConfidence"]> => Boolean(analysis));

  if (analyses.length === 0 || analyses.every((analysis) => analysis.totalWords === 0)) {
    return {
      score: 0,
      level: "low",
      reliability: "low",
      summary: "Expressed confidence could not be assessed because no usable answer transcript was available.",
    };
  }

  const score = weightedEvalAverage(
    analyses
      .filter((analysis) => analysis.totalWords > 0)
      .map((analysis) => ({
        score: analysis.confidenceScore,
        weight: Math.min(analysis.totalWords, 80),
      })),
  );
  const totalWords = analyses.reduce((sum, analysis) => sum + analysis.totalWords, 0);
  const reliability: EvalConfidence = totalWords >= 100 ? "high" : totalWords >= 35 ? "medium" : "low";
  const level: EvalConfidence = score >= 75 ? "high" : score >= 50 ? "medium" : "low";

  return {
    score,
    level,
    reliability,
    summary: `Textual confidence was ${level} (${score}/100) with ${reliability} reliability, based on explicit uncertainty, hedging, fillers, and repeated-word patterns. This does not assess vocal tone or body language.`,
  };
}

// Total red-flag penalty is capped so that stacked flags cannot collapse every weak answer to the
// same floor. A 30/100 and a 5/100 answer must remain distinguishable.
export const MAX_RED_FLAG_PENALTY = 45;

function calculateRedFlagPenalty(evaluation: EvalResponseEvaluation): number {
  const rawPenalty = evaluation.redFlags.reduce((penalty, flag) => {
    if (flag.severity === "critical") return penalty + 35;
    if (flag.severity === "high") return penalty + 20;
    if (flag.severity === "medium") return penalty + 10;
    return penalty + 3;
  }, 0);

  return Math.min(rawPenalty, MAX_RED_FLAG_PENALTY);
}

function getQuestionAggregationWeight(evaluation: EvalResponseEvaluation): number {
  const baseWeight = evaluation.questionOrigin === "predetermined" ? 1 : 0.85;
  const confidenceMultiplier: Record<EvalConfidence, number> = {
    high: 1,
    medium: 0.85,
    low: 0.6,
  };

  return baseWeight * confidenceMultiplier[evaluation.evaluationConfidence];
}

function getRecommendation(
  overallScore: number,
  redFlags: EvalResponseEvaluation["redFlags"],
): EvalRecommendation {
  if (redFlags.some((flag) => flag.severity === "critical")) {
    return "needs_human_review";
  }

  if (redFlags.some((flag) => flag.severity === "high") && overallScore < 82) {
    return "hold";
  }

  if (overallScore >= 88) {
    return "strong_proceed";
  }

  if (overallScore >= 72) {
    return "proceed";
  }

  if (overallScore >= 55) {
    return "hold";
  }

  return "reject";
}

function getRecommendationConfidence(evaluations: EvalResponseEvaluation[]): EvalConfidence {
  if (evaluations.length === 0) {
    return "low";
  }

  const lowConfidenceCount = evaluations.filter(
    (evaluation) => evaluation.evaluationConfidence === "low",
  ).length;
  const highConfidenceCount = evaluations.filter(
    (evaluation) => evaluation.evaluationConfidence === "high",
  ).length;

  if (lowConfidenceCount / evaluations.length >= 0.35) {
    return "low";
  }

  if (highConfidenceCount / evaluations.length >= 0.6) {
    return "high";
  }

  return "medium";
}

function buildSkillScores(evaluations: EvalResponseEvaluation[]): EvalSkillScore[] {
  const dimensionMap = new Map<string, Array<{ score: number; answerId: string }>>();

  for (const evaluation of evaluations) {
    for (const [dimension, dimensionScore] of Object.entries(evaluation.dimensionScores)) {
      const existing = dimensionMap.get(dimension) ?? [];
      existing.push({ score: dimensionScore.score, answerId: evaluation.answerId });
      dimensionMap.set(dimension, existing);
    }
  }

  return Array.from(dimensionMap.entries()).map(([skill, values]) => ({
    skill,
    score: weightedEvalAverage(values.map((value) => ({ score: value.score, weight: 1 }))),
    evidenceAnswerIds: values.map((value) => value.answerId),
  }));
}

function buildReportSummary(
  score: number,
  recommendation: EvalRecommendation,
  confidence: EvalConfidence,
  answeredQuestionCount: number,
): string {
  const questionLabel = answeredQuestionCount === 1 ? "question" : "questions";
  return `Candidate scored ${score}/100 with a ${recommendation.replace(/_/g, " ")} recommendation and ${confidence} confidence based on ${answeredQuestionCount} answered ${questionLabel}.`;
}

function buildCompetencyInsights(skillScores: EvalSkillScore[]): {
  strengths: string[];
  weaknesses: string[];
} {
  if (skillScores.length === 0) {
    return {
      strengths: ["No competency strength could be established because no answers were evaluated."],
      weaknesses: ["Insufficient answered questions to identify a reliable development area."],
    };
  }

  const rankedHigh = [...skillScores].sort((a, b) => b.score - a.score);
  const rankedLow = [...skillScores].sort((a, b) => a.score - b.score);
  const clearStrengths = rankedHigh.filter((skill) => skill.score >= 75).slice(0, 4);
  const developmentAreas = rankedLow.filter((skill) => skill.score < 75).slice(0, 4);
  const strengthSource = clearStrengths.length ? clearStrengths : rankedHigh.slice(0, 2);
  const weaknessSource = developmentAreas.length ? developmentAreas : rankedLow.slice(0, 2);

  return {
    strengths: strengthSource.map((skill) => formatCompetencyStrength(
      skill,
      clearStrengths.length === 0,
    )),
    weaknesses: weaknessSource.map((skill) => formatCompetencyWeakness(
      skill,
      developmentAreas.length === 0,
    )),
  };
}

function formatCompetencyStrength(skill: EvalSkillScore, isRelative: boolean): string {
  const label = formatSkillName(skill.skill);
  const evidence = formatResponseCount(skill.evidenceAnswerIds.length);
  const prefix = isRelative ? "Relative strength" : "Strength";

  return `${prefix} in ${label}: ${skill.score}/100 across ${evidence}, indicating ${strengthDescription(skill.score)}.`;
}

function formatCompetencyWeakness(skill: EvalSkillScore, isDevelopmentOpportunity: boolean): string {
  const label = formatSkillName(skill.skill);
  const evidence = formatResponseCount(skill.evidenceAnswerIds.length);
  const prefix = isDevelopmentOpportunity ? "Development opportunity" : "Needs improvement";

  return `${prefix} in ${label}: ${skill.score}/100 across ${evidence}; ${weaknessDescription(skill.score)}.`;
}

function formatSkillName(skill: string): string {
  const labels: Record<string, string> = {
    model_answer_alignment: "answer relevance and expected-concept alignment",
    factual_correctness: "technical accuracy",
    completeness: "answer completeness",
    concept_coverage: "concept coverage",
    reasoning_quality: "reasoning and explanation",
    technical_specificity: "technical depth and specificity",
    clarity_structure: "clarity and structure",
    communication_quality: "communication",
    role_level_alignment: "role-level readiness",
    problem_understanding: "problem understanding",
    algorithm_correctness: "algorithmic correctness",
    edge_cases: "edge-case awareness",
    complexity_analysis: "complexity analysis",
    code_quality: "solution quality",
    requirements_understanding: "requirements understanding",
    architecture_quality: "architecture design",
    tradeoff_analysis: "trade-off analysis",
    scalability_reliability: "scalability and reliability",
  };

  return labels[skill] ?? skill.replace(/_/g, " ");
}

function formatResponseCount(count: number): string {
  return `${count} ${count === 1 ? "response" : "responses"}`;
}

function strengthDescription(score: number): string {
  if (score >= 90) return "exceptional and consistent capability";
  if (score >= 80) return "strong, dependable capability";
  if (score >= 75) return "solid capability with only minor gaps";
  return "the strongest demonstrated area in this interview";
}

function weaknessDescription(score: number): string {
  if (score < 40) return "substantial improvement is needed before this competency meets role expectations";
  if (score < 60) return "important gaps should be addressed with focused practice";
  if (score < 75) return "the foundation is present, but greater depth and consistency are needed";
  return "this was the least demonstrated area and is the best target for further development";
}

function topUniqueStrings(values: string[], limit: number): string[] {
  const seen = new Set<string>();
  const output: string[] = [];

  for (const value of values) {
    const normalized = value.trim();

    if (!normalized || seen.has(normalized.toLowerCase())) {
      continue;
    }

    seen.add(normalized.toLowerCase());
    output.push(normalized);

    if (output.length >= limit) {
      break;
    }
  }

  return output;
}

const RECOMMENDATION_CUTOFFS = [55, 72, 88];
const BORDERLINE_MARGIN = 3;

export function isBorderlineRecommendation(overallScore: number): boolean {
  return RECOMMENDATION_CUTOFFS.some((cutoff) => Math.abs(overallScore - cutoff) <= BORDERLINE_MARGIN);
}

// Score at/above which a skill reads as a strength in candidate-facing feedback.
const CANDIDATE_STRENGTH_THRESHOLD = 70;

/**
 * Builds the candidate-safe projection of a report. Derived from competency rankings but stripped
 * of every number, question, answer, recommendation, and rubric/dimension mechanic. Deterministic
 * (no LLM) so it cannot fabricate feedback or leak internals.
 */
export function buildCandidateFacingReport(report: EvalCandidateReport): EvalCandidateFacingReport {
  const ranked = [...report.skillScores].sort((a, b) => b.score - a.score);
  const strongSkills = ranked.filter((skill) => skill.score >= CANDIDATE_STRENGTH_THRESHOLD).slice(0, 4);
  const weakSkills = [...ranked].reverse().filter((skill) => skill.score < CANDIDATE_STRENGTH_THRESHOLD).slice(0, 4);

  const strengthSource = strongSkills.length ? strongSkills : ranked.slice(0, 2);
  const strengthKeys = new Set(strengthSource.map((skill) => skill.skill));
  const growthSource = (weakSkills.length ? weakSkills : [...ranked].reverse().slice(0, 2)).filter(
    (skill) => !strengthKeys.has(skill.skill),
  );

  const strengths = strengthSource.map((skill) => candidateStrengthPhrase(formatSkillName(skill.skill)));
  const growthAreas = growthSource.map((skill) => candidateGrowthPhrase(formatSkillName(skill.skill)));

  return {
    roleTitle: report.roleTitle,
    strengths: strengths.length
      ? strengths
      : ["You engaged with the interview and shared relevant experience."],
    growthAreas: growthAreas.length
      ? growthAreas
      : ["Keep practising structured, specific answers backed by concrete examples."],
    encouragementSummary: candidateEncouragement(report.overallScore),
  };
}

function candidateStrengthPhrase(label: string): string {
  return `You came across well on ${label}.`;
}

function candidateGrowthPhrase(label: string): string {
  return `There is room to grow in ${label} — practise going deeper with concrete, specific examples.`;
}

function candidateEncouragement(overallScore: number): string {
  if (overallScore >= 72) {
    return "You demonstrated strong, well-rounded performance in this interview. Keep building on the areas below to grow even further.";
  }

  if (overallScore >= 55) {
    return "You showed a solid foundation with clear room to grow. Focusing on the areas below will strengthen your future interviews.";
  }

  return "Thank you for completing the interview. The areas below are the best places to focus your practice going forward.";
}
