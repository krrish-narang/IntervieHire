export type EvalInterviewType =
  | "technical"
  | "behavioral"
  | "system_design"
  | "case_study"
  | "sales"
  | "hr_screening"
  | "mixed"
  | "custom";

export type EvalQuestionType =
  | "technical_theory"
  | "coding"
  | "system_design"
  | "behavioral"
  | "case_study"
  | "sales_roleplay"
  | "hr_screening"
  | "general"
  | "followup"
  | "custom";

export type EvalQuestionOrigin = "predetermined" | "generated_followup";

export type EvalMode = "model_answer_based" | "followup_model_answer_contextual";

export type EvalConfidence = "high" | "medium" | "low";

export type EvalRecommendation =
  | "strong_proceed"
  | "proceed"
  | "hold"
  | "reject"
  | "needs_human_review";

export type EvalRedFlagSeverity = "low" | "medium" | "high" | "critical";

export type CoverageStatus = "full" | "partial" | "missing" | "contradicted";

export interface EvalInterviewContext {
  interviewId: string;
  candidateId: string;
  companyId?: string;
  roleTitle: string;
  roleLevel?: string;
  interviewType: EvalInterviewType;
  mustHaveSkills?: string[];
  niceToHaveSkills?: string[];
  companyEvaluationNotes?: string;
}

export interface EvalRubricPoint {
  id: string;
  description: string;
  weight: number;
  keywords?: string[];
}

export interface EvalExpectedRedFlag {
  id: string;
  description: string;
  severity: EvalRedFlagSeverity;
}

export interface EvalModelAnswerRubric {
  requiredPoints: EvalRubricPoint[];
  secondaryPoints: EvalRubricPoint[];
  excellentAnswerSignals: EvalRubricPoint[];
  redFlags: EvalExpectedRedFlag[];
  notes?: string;
}

export interface EvalQuestionConfig {
  questionId: string;
  questionText: string;
  questionType: EvalQuestionType;
  questionOrigin: EvalQuestionOrigin;
  modelAnswer: string;
  difficulty?: "easy" | "medium" | "hard" | "custom";
  skillTags?: string[];
  importanceWeight?: number;
  modelAnswerRubric?: EvalModelAnswerRubric;
}

export interface EvalFollowupContext {
  originalQuestionId: string;
  originalAnswerId?: string;
  originalQuestionText: string;
  originalTranscript: string;
  followupQuestionId?: string;
  generatedFollowupQuestion: string;
}

export interface EvalTranscriptInput {
  source: "transcript";
  transcript: string;
  language?: string;
}

export interface EvalFutureAudioAnalysis {
  source: "audio";
  pace?: number | null;
  hesitation?: number | null;
  interruptionCount?: number | null;
  confidenceSignal?: number | null;
  notes?: string | null;
}

export interface EvalFutureVideoAnalysis {
  source: "video";
  eyeContact?: number | null;
  engagement?: number | null;
  facialExpressionNotes?: string | null;
  postureNotes?: string | null;
}

export interface EvalResponseInput {
  answerId: string;
  question: EvalQuestionConfig;
  response: EvalTranscriptInput;
  followupContext?: EvalFollowupContext;
  futureSignals?: {
    audio?: EvalFutureAudioAnalysis;
    video?: EvalFutureVideoAnalysis;
  };
}

export interface EvalDimensionScore {
  score: number;
  reason: string;
  evidence: string[];
  missing: string[];
}

export interface EvalPointCoverage {
  pointId: string;
  description: string;
  weight?: number;
  status: CoverageStatus;
  score: number;
  evidence: string[];
  comment: string;
}

export interface EvalModelAnswerComparison {
  requiredPointCoverage: EvalPointCoverage[];
  secondaryPointCoverage: EvalPointCoverage[];
  excellentSignalCoverage: EvalPointCoverage[];
  incorrectClaims: Array<{
    claim: string;
    severity: EvalRedFlagSeverity;
    correction: string;
  }>;
  coverageScore: number;
}

export interface EvalFollowupAnalysis {
  addressedFollowup: boolean;
  improvedPreviousAnswer: boolean;
  contradictedPreviousAnswer: boolean;
  handledProbeWell: boolean;
  followupValue: "high" | "medium" | "low";
  reason: string;
}

export interface EvalAiAuthorshipAssessment {
  probability: number;
  confidence: EvalConfidence;
  reasons: string[];
  provider: "deepseek" | "gemini";
  disclaimer: string;
}

export interface EvalResponseEvaluation {
  answerId: string;
  questionId: string;
  questionText: string;
  questionOrigin: EvalQuestionOrigin;
  evaluationMode: EvalMode;
  overallScore: number;
  modelAnswerComparison: EvalModelAnswerComparison;
  dimensionScores: Record<string, EvalDimensionScore>;
  followupAnalysis?: EvalFollowupAnalysis;
  aiAuthorshipAssessment?: EvalAiAuthorshipAssessment;
  transcriptConfidence?: {
    fillerCount: number;
    hedgeCount: number;
    strongUncertaintyCount: number;
    repeatedWordCount: number;
    totalWords: number;
    fillerRate: number;
    confidenceScore: number;
    confidenceLevel: EvalConfidence;
    reliability: EvalConfidence;
    confidencePenalty: number;
    notes: string[];
  };
  strengths: string[];
  weaknesses: string[];
  redFlags: Array<{
    label: string;
    severity: EvalRedFlagSeverity;
    reason: string;
  }>;
  followUpRecommendations: string[];
  evaluationConfidence: EvalConfidence;
  summary: string;
  transcriptOnly: true;
}

export interface EvalSkillScore {
  skill: string;
  score: number;
  evidenceAnswerIds: string[];
}

export interface EvalCandidateReport {
  interviewId: string;
  candidateId: string;
  roleTitle: string;
  interviewType: EvalInterviewType;
  overallScore: number;
  recommendation: EvalRecommendation;
  recommendationConfidence: EvalConfidence;
  candidateConfidence: {
    score: number;
    level: EvalConfidence;
    reliability: EvalConfidence;
    summary: string;
  };
  summary: string;
  strengths: string[];
  weaknesses: string[];
  redFlags: EvalResponseEvaluation["redFlags"];
  skillScores: EvalSkillScore[];
  questionBreakdown: EvalResponseEvaluation[];
  suggestedNextSteps: string[];
  transcriptOnly: true;
  futureSignalPlaceholders: {
    audioAnalysisEnabled: false;
    videoAnalysisEnabled: false;
  };
}

export interface EvalTurnLink {
  originalQuestionId: string;
  originalAnswerId: string;
  followupQuestionId: string;
  followupAnswerId?: string;
}

/**
 * Candidate-safe projection of the evaluation. Intentionally contains NO numeric scores, no
 * questions or answers, no recommendation, and no rubric/dimension mechanics — only qualitative
 * growth feedback the candidate is allowed to see. The full EvalCandidateReport is company-only.
 */
export interface EvalCandidateFacingReport {
  roleTitle: string;
  strengths: string[];
  growthAreas: string[];
  encouragementSummary: string;
}
