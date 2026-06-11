import { callDeepSeekJson } from './deepseek.service.js';
import type { EvalModelAnswerRubric, EvalQuestionConfig } from '@interviehire/shared';

export async function generateModelAnswerRubricWithDeepSeek(params: {
  questionText: string;
  questionType?: EvalQuestionConfig['questionType'];
  difficulty?: EvalQuestionConfig['difficulty'];
  modelAnswer: string;
}): Promise<EvalModelAnswerRubric> {
  return callDeepSeekJson<EvalModelAnswerRubric>({
    systemInstruction: [
      'You convert model answers into compact structured grading rubrics.',
      'Extract concepts, not exact wording.',
      'The evaluator will use this rubric to credit semantically equivalent candidate answers.',
      'Return strict JSON only.',
    ].join(' '),
    prompt: [
      `Question: ${params.questionText}`,
      `Question type: ${params.questionType ?? 'custom'}`,
      `Difficulty: ${params.difficulty ?? 'custom'}`,
      `Model answer: ${params.modelAnswer}`,
      '',
      'Return JSON with this shape:',
      JSON.stringify({
        requiredPoints: [
          {
            id: 'short_snake_case_id',
            description: 'Core concept required for a good answer.',
            keywords: ['small list of concept anchors and synonyms, not exact-answer-only terms'],
            weight: 30,
          },
        ],
        secondaryPoints: [
          {
            id: 'short_snake_case_id',
            description: 'Useful supporting concept.',
            keywords: ['supporting concept anchors'],
            weight: 10,
          },
        ],
        excellentAnswerSignals: [
          {
            id: 'short_snake_case_id',
            description: 'Signal that the answer exceeds expectations.',
            keywords: ['strong-answer indicators'],
            weight: 10,
          },
        ],
        redFlags: [
          {
            id: 'short_snake_case_id',
            description: 'Clearly incorrect or risky claim to watch for.',
            severity: 'low | medium | high | critical',
          },
        ],
        notes: 'Short grading guidance.',
      }),
    ].join('\n'),
    maxOutputTokens: Number(process.env.DEEPSEEK_RUBRIC_MAX_TOKENS || 3000),
    temperature: 0.1,
  });
}
