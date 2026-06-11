# Evaluation Module

This module defines the shared transcript-only evaluation contract. It does not
implement backend routes, frontend views, question generation, or model calls.

## Current Assumption

Every evaluated answer has a model answer. The model answer is converted into a
rubric and used as the reference for evaluation. The evaluator should compare
concepts, not exact wording.

## Evaluation Modes

- `model_answer_based`: a normal answer compared to its model answer.
- `followup_model_answer_contextual`: a generated follow-up answer compared to
  its own model answer while also considering the original question and previous
  answer.

## Evaluation Output

Each answer evaluation must include:

- Overall score
- Model-answer comparison
- Required/secondary/excellent point coverage
- Incorrect claims
- Dimension scores with evidence
- Strengths
- Weaknesses
- Red flags
- Suggested follow-up probes
- Evaluation confidence

## Score Philosophy

The evaluator should:

- Credit equivalent correct ideas.
- Credit valid alternative approaches.
- Penalize unsupported buzzwords.
- Penalize contradictions and factual errors.
- Lower confidence when the transcript is short, ambiguous, or internally
  inconsistent.

## Future Signals

Audio and video fields exist only as placeholders. Transcript remains the source
of truth for scoring.
