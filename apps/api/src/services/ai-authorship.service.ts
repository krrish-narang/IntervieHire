import { callDeepSeekJson } from './deepseek.service.js';
import { callGeminiJson } from './gemini.service.js';
import type {
  EvalAiAuthorshipAssessment,
  EvalConfidence,
  EvalQuestionConfig,
} from '@interviehire/shared';

type AuthorshipInput = {
  answerId: string;
  question: EvalQuestionConfig;
  transcript: string;
};

type RawAuthorshipAssessment = {
  answerId?: string;
  probability?: number;
  confidence?: EvalConfidence;
  reasons?: string[];
};

type RawAuthorshipResponse = {
  assessments?: RawAuthorshipAssessment[];
};

const DISCLAIMER = 'Text-only AI-authorship detection is probabilistic and cannot prove plagiarism.';

export async function analyzeAiAuthorship(
  answers: AuthorshipInput[],
): Promise<Map<string, EvalAiAuthorshipAssessment>> {
  if (!answers.length) {
    return new Map();
  }

  const prompt = buildAiAuthorshipPrompt(answers);
  const systemInstruction = [
    'You assess whether interview-answer text may have been generated or heavily rewritten by an AI system.',
    'This is an uncertain text-only inference, not proof of plagiarism or misconduct.',
    'Use the full 0-100 range conservatively and do not equate correctness, fluency, or technical vocabulary alone with AI authorship.',
    'Consider unusually polished templating, generic completeness, repetitive transition patterns, model-answer-like phrasing, lack of answer-specific reasoning, and natural spoken disfluencies.',
    'Short answers provide weak evidence and should usually have low assessment confidence.',
    'Evaluate each answer independently and return strict JSON only.',
  ].join(' ');

  const hasDeepSeek = hasConfiguredKey(process.env.DEEPSEEK_API_KEY);
  const hasGemini = hasConfiguredKey(process.env.GEMINI_API_KEY);
  let response: RawAuthorshipResponse;
  let provider: EvalAiAuthorshipAssessment['provider'];

  if (hasDeepSeek) {
    try {
      response = await callDeepSeekJson<RawAuthorshipResponse>({
        systemInstruction,
        prompt,
        maxOutputTokens: Number(process.env.DEEPSEEK_AUTHORSHIP_MAX_TOKENS || 4000),
        temperature: 0.1,
      });
      provider = 'deepseek';
      return normalizeAssessments(response, answers, provider);
    } catch (error) {
      console.error('DeepSeek AI-authorship analysis failed.', error);
    }
  }

  if (hasGemini) {
    try {
      response = await callGeminiJson<RawAuthorshipResponse>({
        systemInstruction,
        prompt,
        model: process.env.GEMINI_AUTHORSHIP_MODEL || process.env.GEMINI_EVALUATION_MODEL,
        maxOutputTokens: Number(process.env.GEMINI_AUTHORSHIP_MAX_TOKENS || 4000),
        temperature: 0.1,
      });
      provider = 'gemini';
      return normalizeAssessments(response, answers, provider);
    } catch (error) {
      console.error('Gemini AI-authorship analysis failed.', error);
    }
  }

  return new Map();
}

function buildAiAuthorshipPrompt(answers: AuthorshipInput[]): string {
  return [
    'Estimate AI-authorship likelihood for each candidate answer.',
    '',
    'Calibration:',
    '- 0-20: little textual evidence of AI assistance.',
    '- 21-49: weak or ambiguous indicators.',
    '- 50-74: several notable indicators, still uncertain.',
    '- 75-100: strong and repeated indicators; never describe this as proof.',
    '- Natural spoken phrasing, corrections, fragments, and answer-specific examples lower likelihood.',
    '- Generic polished structure, uniform sentence rhythm, canned transitions, exhaustive checklist style, and close model-answer phrasing may raise likelihood.',
    '- Do not raise likelihood merely because an answer is correct, concise, formal, or uses expected terminology.',
    '- Return one assessment for every answerId.',
    '- Give 1-3 reasons, each no longer than 12 words.',
    '',
    `Answers: ${JSON.stringify(answers.map((answer) => ({
      answerId: answer.answerId,
      question: answer.question.questionText,
      modelAnswer: answer.question.modelAnswer,
      candidateTranscript: answer.transcript,
    })))}`,
    '',
    'Return JSON only:',
    JSON.stringify({
      assessments: [
        {
          answerId: 'same answerId from input',
          probability: 0,
          confidence: 'high | medium | low',
          reasons: [
            'Short reason based only on textual indicators.',
          ],
        },
      ],
    }),
  ].join('\n');
}

function normalizeAssessments(
  response: RawAuthorshipResponse,
  answers: AuthorshipInput[],
  provider: EvalAiAuthorshipAssessment['provider'],
): Map<string, EvalAiAuthorshipAssessment> {
  const validAnswerIds = new Set(answers.map((answer) => answer.answerId));
  const assessments = new Map<string, EvalAiAuthorshipAssessment>();

  for (const item of response.assessments ?? []) {
    if (!item.answerId || !validAnswerIds.has(item.answerId)) {
      continue;
    }

    assessments.set(item.answerId, {
      probability: clampPercentage(item.probability),
      confidence: normalizeConfidence(item.confidence),
      reasons: normalizeReasons(item.reasons),
      provider,
      disclaimer: DISCLAIMER,
    });
  }

  return assessments;
}

function hasConfiguredKey(value: string | undefined): boolean {
  return Boolean(value && value !== 'replace-me');
}

function clampPercentage(value: number | undefined): number {
  return Math.max(0, Math.min(100, Math.round(Number.isFinite(value) ? Number(value) : 0)));
}

function normalizeConfidence(value: EvalConfidence | undefined): EvalConfidence {
  return value === 'high' || value === 'medium' ? value : 'low';
}

function normalizeReasons(reasons: string[] | undefined): string[] {
  const normalized = (reasons ?? [])
    .map((reason) => String(reason).trim())
    .filter(Boolean)
    .slice(0, 3);

  return normalized.length ? normalized : ['Insufficient distinctive text for a reliable assessment.'];
}
