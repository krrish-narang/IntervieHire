import type { EvalInterviewType, EvalQuestionType } from "./types";

export type EvalDimensionWeights = Record<string, number>;

export const EVAL_UNIVERSAL_WEIGHTS: EvalDimensionWeights = {
  model_answer_alignment: 25,
  factual_correctness: 20,
  completeness: 15,
  reasoning_quality: 15,
  clarity_structure: 10,
  role_level_alignment: 10,
  communication_quality: 5,
};

export const EVAL_INTERVIEW_TYPE_WEIGHTS: Record<EvalInterviewType, EvalDimensionWeights> = {
  technical: {
    model_answer_alignment: 25,
    factual_correctness: 25,
    completeness: 15,
    reasoning_quality: 15,
    technical_specificity: 10,
    clarity_structure: 5,
    role_level_alignment: 5,
  },
  behavioral: {
    model_answer_alignment: 15,
    relevance: 15,
    ownership: 20,
    impact: 20,
    reflection: 10,
    clarity_structure: 10,
    role_level_alignment: 10,
  },
  system_design: {
    model_answer_alignment: 15,
    requirements_understanding: 15,
    architecture_quality: 20,
    tradeoff_analysis: 20,
    scalability_reliability: 15,
    clarity_structure: 10,
    role_level_alignment: 5,
  },
  case_study: {
    model_answer_alignment: 20,
    problem_framing: 15,
    analysis_quality: 20,
    business_judgment: 20,
    recommendation_quality: 15,
    clarity_structure: 10,
  },
  sales: {
    model_answer_alignment: 15,
    discovery_quality: 20,
    objection_handling: 20,
    customer_empathy: 15,
    persuasion: 15,
    communication_quality: 15,
  },
  hr_screening: {
    model_answer_alignment: 20,
    motivation: 20,
    role_alignment: 25,
    professionalism: 15,
    clarity_structure: 10,
    risk_awareness: 10,
  },
  mixed: EVAL_UNIVERSAL_WEIGHTS,
  custom: EVAL_UNIVERSAL_WEIGHTS,
};

export const EVAL_QUESTION_TYPE_WEIGHTS: Partial<
  Record<EvalQuestionType, EvalDimensionWeights>
> = {
  technical_theory: {
    model_answer_alignment: 25,
    factual_correctness: 25,
    concept_coverage: 20,
    reasoning_quality: 15,
    technical_specificity: 10,
    clarity_structure: 5,
  },
  coding: {
    model_answer_alignment: 15,
    problem_understanding: 15,
    algorithm_correctness: 25,
    edge_cases: 15,
    complexity_analysis: 15,
    code_quality: 10,
    communication_quality: 5,
  },
  system_design: EVAL_INTERVIEW_TYPE_WEIGHTS.system_design,
  behavioral: EVAL_INTERVIEW_TYPE_WEIGHTS.behavioral,
  case_study: EVAL_INTERVIEW_TYPE_WEIGHTS.case_study,
  sales_roleplay: EVAL_INTERVIEW_TYPE_WEIGHTS.sales,
  hr_screening: EVAL_INTERVIEW_TYPE_WEIGHTS.hr_screening,
  followup: {
    model_answer_alignment: 20,
    addressed_followup: 20,
    factual_correctness: 15,
    depth_expansion: 15,
    consistency_with_previous_answer: 15,
    adaptability: 10,
    communication_quality: 5,
  },
};

export function getEvalDimensionWeights(
  interviewType: EvalInterviewType,
  questionType?: EvalQuestionType,
): EvalDimensionWeights {
  if (questionType && EVAL_QUESTION_TYPE_WEIGHTS[questionType]) {
    return EVAL_QUESTION_TYPE_WEIGHTS[questionType];
  }

  return EVAL_INTERVIEW_TYPE_WEIGHTS[interviewType] ?? EVAL_UNIVERSAL_WEIGHTS;
}

/**
 * Maps a question type to the interview type it most strongly implies. Types that do not
 * point to a single interview style (general/followup/custom) intentionally have no mapping
 * so they do not skew inference.
 */
const QUESTION_TYPE_TO_INTERVIEW_TYPE: Partial<Record<EvalQuestionType, EvalInterviewType>> = {
  technical_theory: "technical",
  coding: "technical",
  system_design: "system_design",
  behavioral: "behavioral",
  case_study: "case_study",
  sales_roleplay: "sales",
  hr_screening: "hr_screening",
};

/**
 * Infers the interview type from the mix of question types actually used in an interview.
 * Returns the single implied type when all signal-bearing questions agree, "mixed" when they
 * conflict, and null when no question type carries a signal (caller should fall back to role).
 */
export function inferEvalInterviewType(
  questionTypes: Array<EvalQuestionType | undefined | null>,
): EvalInterviewType | null {
  const votes = new Set<EvalInterviewType>();

  for (const questionType of questionTypes) {
    const mapped = questionType ? QUESTION_TYPE_TO_INTERVIEW_TYPE[questionType] : undefined;
    if (mapped) {
      votes.add(mapped);
    }
  }

  if (votes.size === 0) {
    return null;
  }

  if (votes.size === 1) {
    return [...votes][0];
  }

  return "mixed";
}

/**
 * Dev-time guard: returns a list of weight tables whose weights do not sum to ~100.
 * Weighting is normalized at scoring time, so this is a correctness aid, not a runtime gate.
 */
export function validateEvalWeightTables(): string[] {
  const problems: string[] = [];
  const check = (label: string, weights: EvalDimensionWeights) => {
    const total = Object.values(weights).reduce((sum, weight) => sum + weight, 0);
    if (Math.abs(total - 100) > 0.01) {
      problems.push(`${label} weights sum to ${total}, expected 100`);
    }
  };

  check("universal", EVAL_UNIVERSAL_WEIGHTS);
  for (const [type, weights] of Object.entries(EVAL_INTERVIEW_TYPE_WEIGHTS)) {
    check(`interviewType:${type}`, weights);
  }
  for (const [type, weights] of Object.entries(EVAL_QUESTION_TYPE_WEIGHTS)) {
    if (weights) {
      check(`questionType:${type}`, weights);
    }
  }

  return problems;
}

export function normalizeEvalWeights(weights: EvalDimensionWeights): EvalDimensionWeights {
  const total = Object.values(weights).reduce((sum, weight) => sum + weight, 0);

  if (total === 0) {
    return weights;
  }

  return Object.fromEntries(
    Object.entries(weights).map(([dimension, weight]) => [
      dimension,
      Number(((weight / total) * 100).toFixed(2)),
    ]),
  );
}

export const EVAL_SCORE_BANDS = {
  exceptional: { min: 90, description: "Exceeds the model answer and role expectations." },
  strong: { min: 75, description: "Covers expected answer with minor gaps." },
  acceptable: { min: 60, description: "Shows partial understanding with notable gaps." },
  weak: { min: 40, description: "Misses important concepts or has shallow reasoning." },
  poor: { min: 0, description: "Incorrect, evasive, or mostly non-responsive." },
} as const;
