import type {
  EvalInterviewContext,
  EvalModelAnswerRubric,
  EvalQuestionConfig,
  EvalResponseInput,
} from "./types";
import { getEvalDimensionWeights, normalizeEvalWeights } from "./metrics";
import { inferEvalMode } from "./scoring";

export function buildEvalRubricExtractionPrompt(question: EvalQuestionConfig): string {
  return [
    "Convert the model answer into a structured evaluation rubric.",
    "The model answer is the reference answer, but do not make exact wording mandatory.",
    "Extract concepts, expected depth, strong-answer signals, and incorrect claims to watch for.",
    "Return strict JSON only.",
    "",
    `Question: ${question.questionText}`,
    `Question type: ${question.questionType}`,
    `Difficulty: ${question.difficulty ?? "not specified"}`,
    `Model answer: ${question.modelAnswer}`,
    "",
    "Required JSON shape:",
    JSON.stringify(
      {
        requiredPoints: [
          {
            id: "short_snake_case_id",
            description: "Concept the candidate must cover to show core understanding.",
            weight: 25,
          },
        ],
        secondaryPoints: [
          {
            id: "short_snake_case_id",
            description: "Useful supporting concept that improves completeness.",
            weight: 10,
          },
        ],
        excellentAnswerSignals: [
          {
            id: "short_snake_case_id",
            description: "Signal that the candidate exceeds the expected answer.",
            weight: 10,
          },
        ],
        redFlags: [
          {
            id: "short_snake_case_id",
            description: "Clearly incorrect, evasive, unsafe, or role-inappropriate claim.",
            severity: "low | medium | high | critical",
          },
        ],
        notes: "Any special scoring guidance.",
      } satisfies Record<keyof EvalModelAnswerRubric, unknown>,
      null,
      2,
    ),
  ].join("\n");
}

export function buildEvalAnswerPrompt(
  context: EvalInterviewContext,
  input: EvalResponseInput,
): string {
  const mode = inferEvalMode(input.question);
  const weights = normalizeEvalWeights(
    getEvalDimensionWeights(context.interviewType, input.question.questionType),
  );

  return [
    "Evaluate this interview answer for a hiring report.",
    "Use transcript content only. Do not infer tone, confidence, audio, video, facial expression, or body language.",
    "Every answer has a model answer. Compare the candidate against the model answer by concept, not exact phrasing.",
    "Give credit for equivalent correct ideas and valid alternative approaches.",
    "Penalize factual errors, contradictions, vague buzzwords, unsupported claims, and non-answers.",
    "Do not invent evidence. Evidence must come from the transcript and should be short.",
    "Return strict JSON only.",
    "",
    "Evaluation process:",
    "1. Identify the candidate's concrete claims.",
    "2. Compare those claims to the model answer and extracted rubric.",
    "3. Return exactly one coverage row for every rubric point, preserving its id, description, and weight.",
    "4. Mark a point full only when the transcript clearly demonstrates the complete concept; use partial for incomplete but correct understanding, missing when absent, and contradicted when the candidate states an incompatible claim.",
    "5. Use point scores consistently: full=100, partial=50, missing=0, contradicted=0.",
    "6. Do not award credit merely because the candidate repeats terms without explaining the concept.",
    "7. Treat the model answer as the expected substance, not a script. A technically valid alternative may receive equal credit when it answers the same requirement.",
    "8. Score each dimension using the provided dimension weights.",
    "9. List strengths, weaknesses, red flags, and useful follow-up probes.",
    "10. Set evaluationConfidence lower if the transcript is too short, unclear, or internally inconsistent.",
    "",
    `Role: ${context.roleTitle}`,
    `Role level: ${context.roleLevel ?? "not specified"}`,
    `Interview type: ${context.interviewType}`,
    `Must-have skills: ${(context.mustHaveSkills ?? []).join(", ") || "not specified"}`,
    `Nice-to-have skills: ${(context.niceToHaveSkills ?? []).join(", ") || "not specified"}`,
    `Company evaluation notes: ${context.companyEvaluationNotes ?? "not specified"}`,
    "",
    `Question id: ${input.question.questionId}`,
    `Answer id: ${input.answerId}`,
    `Question origin: ${input.question.questionOrigin}`,
    `Question type: ${input.question.questionType}`,
    `Evaluation mode: ${mode}`,
    `Difficulty: ${input.question.difficulty ?? "not specified"}`,
    `Skill tags: ${(input.question.skillTags ?? []).join(", ") || "not specified"}`,
    `Dimension weights: ${JSON.stringify(weights)}`,
    "",
    `Question: ${input.question.questionText}`,
    `Model answer: ${input.question.modelAnswer}`,
    `Structured model rubric: ${JSON.stringify(input.question.modelAnswerRubric ?? null)}`,
    "",
    input.followupContext
      ? [
          "Follow-up context:",
          `Original question: ${input.followupContext.originalQuestionText}`,
          `Original answer transcript: ${input.followupContext.originalTranscript}`,
          `Generated follow-up question: ${input.followupContext.generatedFollowupQuestion}`,
        ].join("\n")
      : "Follow-up context: not applicable",
    "",
    `Candidate transcript: ${input.response.transcript}`,
    "",
    "Required JSON shape:",
    getEvalResponseJsonShape(mode),
  ].join("\n");
}

function getEvalResponseJsonShape(mode: string): string {
  return JSON.stringify(
    {
      answerId: "answer id from input",
      questionId: "question id from input",
      questionOrigin: "predetermined | generated_followup",
      evaluationMode: mode,
      overallScore: 0,
      modelAnswerComparison: {
        requiredPointCoverage: [
          {
            pointId: "rubric point id",
            description: "Expected concept",
            weight: 25,
            status: "full | partial | missing | contradicted",
            score: "100 for full, 50 for partial, 0 for missing or contradicted",
            evidence: ["Short transcript evidence"],
            comment: "Why this status was assigned",
          },
        ],
        secondaryPointCoverage: [],
        excellentSignalCoverage: [],
        incorrectClaims: [
          {
            claim: "Incorrect candidate claim",
            severity: "low | medium | high | critical",
            correction: "Correct version",
          },
        ],
        coverageScore: 0,
      },
      dimensionScores: {
        dimension_name: {
          score: 0,
          reason: "Specific reason for the score.",
          evidence: ["Short transcript evidence"],
          missing: ["Missing or weak element"],
        },
      },
      followupAnalysis:
        mode === "followup_model_answer_contextual"
          ? {
              addressedFollowup: true,
              improvedPreviousAnswer: true,
              contradictedPreviousAnswer: false,
              handledProbeWell: true,
              followupValue: "high | medium | low",
              reason: "What additional signal the follow-up provided.",
            }
          : undefined,
      strengths: ["Concrete strength"],
      weaknesses: ["Concrete weakness"],
      redFlags: [
        {
          label: "Red flag label",
          severity: "low | medium | high | critical",
          reason: "Why this matters",
        },
      ],
      followUpRecommendations: ["Suggested probe for the next round"],
      evaluationConfidence: "high | medium | low",
      summary: "One-paragraph answer summary",
      transcriptOnly: true,
    },
    null,
    2,
  );
}
