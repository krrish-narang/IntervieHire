const assert = require("node:assert/strict");
const test = require("node:test");

const { buildCandidateFacingReport, isBorderlineRecommendation } = require("../dist/evaluation/scoring.js");

const fullReport = {
  interviewId: "i1",
  candidateId: "c1",
  roleTitle: "Senior Backend Engineer",
  interviewType: "technical",
  overallScore: 81,
  recommendation: "proceed",
  recommendationConfidence: "high",
  candidateConfidence: { score: 77, level: "high", reliability: "high", summary: "..." },
  summary: "Candidate scored 81/100 with a proceed recommendation.",
  strengths: ["Strength in technical accuracy: 88/100 across 3 responses"],
  weaknesses: ["Development opportunity in edge-case awareness: 41/100 across 2 responses"],
  redFlags: [{ label: "Possible incorrect claim", severity: "high", reason: "Said HTTP is stateful" }],
  skillScores: [
    { skill: "factual_correctness", score: 88, evidenceAnswerIds: ["a1"] },
    { skill: "reasoning_quality", score: 84, evidenceAnswerIds: ["a1"] },
    { skill: "edge_cases", score: 41, evidenceAnswerIds: ["a2"] },
    { skill: "clarity_structure", score: 52, evidenceAnswerIds: ["a2"] },
  ],
  questionBreakdown: [
    { questionText: "Explain TCP vs UDP", overallScore: 80, summary: "covered most points" },
  ],
  suggestedNextSteps: ["Probe further: idempotency"],
  transcriptOnly: true,
  futureSignalPlaceholders: { audioAnalysisEnabled: false, videoAnalysisEnabled: false },
};

test("candidate report exposes only role title, strengths, growth areas, and a summary", () => {
  const candidate = buildCandidateFacingReport(fullReport);
  assert.deepEqual(
    Object.keys(candidate).sort(),
    ["encouragementSummary", "growthAreas", "roleTitle", "strengths"],
  );
  assert.ok(candidate.strengths.length > 0);
  assert.ok(candidate.growthAreas.length > 0);
});

test("candidate report leaks no numbers, scores, recommendation, or question text", () => {
  const candidate = buildCandidateFacingReport(fullReport);
  const blob = JSON.stringify({
    strengths: candidate.strengths,
    growthAreas: candidate.growthAreas,
    encouragementSummary: candidate.encouragementSummary,
  });

  // No digits at all (scores, "x/100", counts).
  assert.equal(/\d/.test(blob), false, `candidate text contained a digit: ${blob}`);
  // No leaked recommendation verbs or internal mechanics.
  for (const banned of ["proceed", "reject", "/100", "rubric", "dimension", "TCP", "UDP"]) {
    assert.equal(blob.toLowerCase().includes(banned.toLowerCase()), false, `leaked: ${banned}`);
  }
});

test("borderline detection flags scores within 3 of a cutoff", () => {
  assert.equal(isBorderlineRecommendation(88), true);
  assert.equal(isBorderlineRecommendation(90), true);
  assert.equal(isBorderlineRecommendation(85), true);
  assert.equal(isBorderlineRecommendation(72), true);
  assert.equal(isBorderlineRecommendation(80), false);
  assert.equal(isBorderlineRecommendation(95), false);
});
