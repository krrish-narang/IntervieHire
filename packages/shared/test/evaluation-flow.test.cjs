const assert = require("node:assert/strict");
const test = require("node:test");

const { pairAnsweredEvalQuestions } = require("../dist/evaluation/flow.js");

const questions = [
  { id: "q1", text: "First question" },
  { id: "q2", text: "Second question" },
  { id: "q3", text: "Third question" },
  { id: "q4", text: "Fourth question" },
];

test("pairs only answered questions from the transcript", () => {
  const pairs = pairAnsweredEvalQuestions([
    { speaker: "ai", text: "First question", questionIndex: 0 },
    { speaker: "candidate", text: "Answer one", questionIndex: 0 },
    { speaker: "ai", text: "Second question", questionIndex: 1 },
    { speaker: "candidate", text: "Answer two", questionIndex: 1 },
    { speaker: "ai", text: "Third question", questionIndex: 2 },
  ], questions);

  assert.equal(pairs.length, 2);
  assert.deepEqual(pairs.map((pair) => pair.question.id), ["q1", "q2"]);
  assert.deepEqual(pairs.map((pair) => pair.answer), ["Answer one", "Answer two"]);
});

test("uses explicit candidate questionIndex over active question context", () => {
  const pairs = pairAnsweredEvalQuestions([
    { speaker: "ai", text: "First question", questionIndex: 0 },
    { speaker: "candidate", text: "Answer for third", questionIndex: 2 },
  ], questions);

  assert.equal(pairs.length, 1);
  assert.equal(pairs[0].question.id, "q3");
  assert.equal(pairs[0].questionIndex, 2);
});

test("falls back in question order when no valid question index exists", () => {
  const pairs = pairAnsweredEvalQuestions([
    { speaker: "candidate", text: "Fallback answer one" },
    { speaker: "candidate", text: "Fallback answer two" },
  ], questions);

  assert.equal(pairs.length, 2);
  assert.deepEqual(pairs.map((pair) => pair.question.id), ["q1", "q2"]);
});

test("merges consecutive candidate chunks for the same question", () => {
  const pairs = pairAnsweredEvalQuestions([
    { speaker: "ai", text: "First question", questionIndex: 0 },
    { speaker: "candidate", text: "First chunk", questionIndex: 0 },
    { speaker: "candidate", text: "Second chunk", questionIndex: 0 },
    { speaker: "ai", text: "Second question", questionIndex: 1 },
    { speaker: "candidate", text: "Next answer", questionIndex: 1 },
  ], questions);

  assert.equal(pairs.length, 2);
  assert.equal(pairs[0].answer, "First chunk\nSecond chunk");
  assert.equal(pairs[1].answer, "Next answer");
});

test("does not merge candidate answers across an ai entry with invalid question index", () => {
  const pairs = pairAnsweredEvalQuestions([
    { speaker: "ai", text: "First question", questionIndex: 0 },
    { speaker: "candidate", text: "Answer one", questionIndex: 0 },
    { speaker: "ai", text: "Closing prompt", questionIndex: null },
    { speaker: "candidate", text: "Answer two" },
  ], questions);

  assert.equal(pairs.length, 2);
  assert.deepEqual(pairs.map((pair) => pair.answer), ["Answer one", "Answer two"]);
});

test("ignores empty answers and invalid question indexes", () => {
  const pairs = pairAnsweredEvalQuestions([
    { speaker: "candidate", text: "   ", questionIndex: 0 },
    { speaker: "candidate", text: "Out of range", questionIndex: 99 },
    { speaker: "ai", text: "Third question", questionIndex: 2 },
    { speaker: "candidate", text: "Answer three" },
  ], questions);

  assert.equal(pairs.length, 2);
  assert.deepEqual(pairs.map((pair) => pair.question.id), ["q1", "q3"]);
  assert.deepEqual(pairs.map((pair) => pair.answer), ["Out of range", "Answer three"]);
});
