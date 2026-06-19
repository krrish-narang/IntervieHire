const assert = require("node:assert/strict");
const test = require("node:test");

const {
  inferEvalInterviewType,
  validateEvalWeightTables,
} = require("../dist/evaluation/metrics.js");
const { calculateFinalAnswerScore, MAX_RED_FLAG_PENALTY } = require("../dist/evaluation/scoring.js");

test("every dimension weight table sums to 100", () => {
  assert.deepEqual(validateEvalWeightTables(), []);
});

test("infers a single interview type when question types agree", () => {
  assert.equal(inferEvalInterviewType(["technical_theory", "coding"]), "technical");
  assert.equal(inferEvalInterviewType(["behavioral", "behavioral"]), "behavioral");
});

test("infers mixed on conflict and null when no type carries signal", () => {
  assert.equal(inferEvalInterviewType(["behavioral", "coding"]), "mixed");
  assert.equal(inferEvalInterviewType(["general", "custom", undefined]), null);
});

test("stacked red flags cannot exceed the penalty cap", () => {
  const baseEvaluation = {
    answerId: "a1",
    questionId: "q1",
    questionText: "Q",
    questionOrigin: "predetermined",
    evaluationMode: "model_answer_based",
    overallScore: 0,
    modelAnswerComparison: {
      requiredPointCoverage: [
        { pointId: "p1", description: "core", weight: 100, status: "full", score: 100, evidence: [], comment: "" },
      ],
      secondaryPointCoverage: [],
      excellentSignalCoverage: [],
      incorrectClaims: [],
      coverageScore: 100,
    },
    dimensionScores: {
      model_answer_alignment: { score: 100, reason: "", evidence: [], missing: [] },
    },
    strengths: [],
    weaknesses: [],
    redFlags: [],
    followUpRecommendations: [],
    evaluationConfidence: "high",
    summary: "",
    transcriptOnly: true,
  };

  const fiveCriticals = {
    ...baseEvaluation,
    redFlags: Array.from({ length: 5 }, (_, index) => ({
      label: `flag ${index}`,
      severity: "critical",
      reason: "r",
    })),
  };

  const params = (evaluation) => ({
    evaluation,
    context: { interviewType: "technical" },
    question: { questionId: "q1", questionText: "Q", questionType: "technical_theory", questionOrigin: "predetermined", modelAnswer: "" },
  });

  const cleanScore = calculateFinalAnswerScore(params(baseEvaluation));
  const flaggedScore = calculateFinalAnswerScore(params(fiveCriticals));

  // 5 criticals would be -175 uncapped; the cap keeps the drop at MAX_RED_FLAG_PENALTY.
  assert.equal(cleanScore - flaggedScore, MAX_RED_FLAG_PENALTY);
});
