import { callDeepSeekJson } from './deepseek.service.js';

type GeneratedQuestion = {
  text: string;
  difficulty: 'EASY' | 'MEDIUM' | 'HARD';
  topicCategories: string[];
  aiEvaluationGuidance: string;
};

type DeepSeekQuestionResponse = {
  questions: Array<{
    text: string;
    difficulty?: 'EASY' | 'MEDIUM' | 'HARD';
    topicCategories?: string[];
    questionType?: string;
    modelAnswer: string;
    rubric?: {
      requiredPoints?: Array<{ id?: string; description: string; keywords?: string[]; weight?: number }>;
      secondaryPoints?: Array<{ id?: string; description: string; keywords?: string[]; weight?: number }>;
      excellentAnswerSignals?: Array<{ id?: string; description: string; keywords?: string[]; weight?: number }>;
      redFlags?: Array<{ id?: string; description: string; severity?: 'low' | 'medium' | 'high' | 'critical' }>;
    };
  }>;
};

export async function generateQuestions(input: {roleType: string; jobDescription: string; companyName: string; jobTitle?: string}) {
  const competencyMap: Record<string,string> = {
    PRODUCT_MANAGEMENT: 'user empathy, prioritization, product lifecycle, metrics, cross-functional collaboration',
    BUSINESS_ANALYST: 'analytical rigor, requirements gathering, stakeholder management, documentation, insight generation',
    FOUNDERS_OFFICE: 'entrepreneurial mindset, ownership, ambiguity handling, leadership, strategic thinking',
    CONSULTING: 'problem decomposition, client skills, structured thinking, executive communication, industry awareness'
  };
  const prompt = [
    `Generate 8 interview questions for ${input.companyName}.`,
    `Role: ${input.jobTitle || input.roleType}.`,
    `Focus areas: ${competencyMap[input.roleType] || 'role competencies'}.`,
    `Job description: ${input.jobDescription}`,
    '',
    'Each question must include a modelAnswer that can be used as the reference answer during evaluation.',
    'Each rubric must be compact, concept based, and suitable for semantic grading.',
    'Return JSON only in this shape:',
    JSON.stringify({
      questions: [
        {
          text: 'question text',
          difficulty: 'EASY | MEDIUM | HARD',
          topicCategories: ['skill or topic'],
          questionType: 'technical_theory | system_design | behavioral | product_sense | case_study',
          modelAnswer: 'reference answer with the important concepts and tradeoffs',
          rubric: {
            requiredPoints: [
              { id: 'stable_snake_case_id', description: 'required concept', keywords: ['keyword'], weight: 30 },
            ],
            secondaryPoints: [],
            excellentAnswerSignals: [],
            redFlags: [
              { id: 'stable_snake_case_id', description: 'incorrect or risky claim', severity: 'medium' },
            ],
          },
        },
      ],
    }),
  ].join('\n');

  try {
    const response = await callDeepSeekJson<DeepSeekQuestionResponse>({
      systemInstruction: 'You are an expert interview designer. Return strict JSON and include model answers for every question.',
      prompt,
      maxOutputTokens: Number(process.env.DEEPSEEK_QUESTION_MAX_TOKENS || 8000),
      temperature: 0.25,
    });

    return normalizeGeneratedQuestions(response.questions);
  } catch {
    return [{
      text: 'Walk me through a recent project where you created measurable impact.',
      difficulty: 'MEDIUM',
      topicCategories: ['impact'],
      aiEvaluationGuidance: JSON.stringify({
        questionType: 'behavioral',
        modelAnswer: 'A strong answer explains the situation, the candidate ownership, the specific actions taken, measurable impact, tradeoffs, collaboration, and what they learned or would improve.',
        rubric: {
          requiredPoints: [
            { id: 'context_and_goal', description: 'Explains the project context and goal.', keywords: ['context', 'goal', 'problem'], weight: 25 },
            { id: 'ownership_actions', description: 'Clearly describes personal ownership and actions.', keywords: ['ownership', 'action', 'decision'], weight: 35 },
            { id: 'measurable_impact', description: 'Gives measurable impact or outcome.', keywords: ['metric', 'impact', 'outcome'], weight: 30 },
          ],
          secondaryPoints: [
            { id: 'reflection', description: 'Reflects on tradeoffs or lessons learned.', keywords: ['tradeoff', 'learned', 'improve'], weight: 10 },
          ],
          excellentAnswerSignals: [],
          redFlags: [],
        },
      }),
    }] satisfies GeneratedQuestion[];
  }
}

function normalizeGeneratedQuestions(questions: DeepSeekQuestionResponse['questions'] = []): GeneratedQuestion[] {
  return questions
    .filter((question) => question.text?.trim() && question.modelAnswer?.trim())
    .map((question) => {
      const rubric = question.rubric ?? {};

      return {
        text: question.text.trim(),
        difficulty: question.difficulty || 'MEDIUM',
        topicCategories: question.topicCategories?.length ? question.topicCategories : ['role competencies'],
        aiEvaluationGuidance: JSON.stringify({
          questionType: question.questionType || 'technical_theory',
          modelAnswer: question.modelAnswer.trim(),
          rubric: {
            requiredPoints: normalizeRubricPoints(rubric.requiredPoints, question.modelAnswer),
            secondaryPoints: normalizeRubricPoints(rubric.secondaryPoints),
            excellentAnswerSignals: normalizeRubricPoints(rubric.excellentAnswerSignals),
            redFlags: (rubric.redFlags ?? []).map((flag, index) => ({
              id: flag.id || `red_flag_${index + 1}`,
              description: flag.description,
              severity: flag.severity || 'medium',
            })).filter((flag) => flag.description?.trim()),
          },
        }),
      };
    });
}

function normalizeRubricPoints(
  points: Array<{ id?: string; description: string; keywords?: string[]; weight?: number }> | undefined,
  fallbackDescription?: string,
) {
  const source = points?.length ? points : fallbackDescription ? [{ description: fallbackDescription, weight: 100 }] : [];

  return source
    .map((point, index) => ({
      id: point.id || `point_${index + 1}`,
      description: point.description,
      keywords: point.keywords ?? [],
      weight: point.weight ?? 25,
    }))
    .filter((point) => point.description?.trim());
}
