import PDFDocument from 'pdfkit';
import { prisma } from '../lib/prisma.js';
import fs from 'node:fs';
import path from 'node:path';
import { callDeepSeekJson } from './deepseek.service.js';
import { analyzeAiAuthorship } from './ai-authorship.service.js';
import {
  aggregateEvalCandidateReport,
  analyzeTranscriptConfidence,
  calculateFinalAnswerScore,
  normalizeEvalResponseEvaluation,
  pairAnsweredEvalQuestions,
  type EvalCandidateReport,
  type EvalDimensionScore,
  type EvalExpectedRedFlag,
  type EvalInterviewContext,
  type EvalModelAnswerComparison,
  type EvalModelAnswerRubric,
  type EvalPointCoverage,
  type EvalQuestionConfig,
  type EvalRedFlagSeverity,
  type EvalResponseEvaluation,
} from '@interviehire/shared';

type TranscriptEntry = {
  speaker?: string;
  text?: string;
  timestamp?: string;
  metrics?: Record<string, unknown>;
  type?: string;
  questionIndex?: number | null;
};

type QuestionWithGuidance = {
  id: string;
  text: string;
  difficulty: 'EASY' | 'MEDIUM' | 'HARD';
  topicCategories: string[];
  aiEvaluationGuidance: string;
};

type PreparedAnswer = {
  answerId: string;
  question: QuestionWithGuidance;
  questionConfig: EvalQuestionConfig;
  transcript: string;
  transcriptConfidence: ReturnType<typeof analyzeTranscriptConfidence>;
  localFallbackEvaluation: EvalResponseEvaluation;
};

type LlmEvaluationResponse = {
  responseEvaluations: EvalResponseEvaluation[];
};

export async function evaluateInterview(sessionId: string): Promise<EvalCandidateReport> {
  const session = await prisma.interviewSession.findUnique({
    where: { id: sessionId },
    include: {
      company: true,
      candidate: true,
      jobRole: { include: { questions: { where: { isActive: true }, orderBy: { createdAt: 'asc' } } } },
      proctoringLogs: true,
    },
  });

  if (!session) throw new Error('Session not found');

  const transcript = normalizeTranscript(session.transcript);
  const questions = session.jobRole.questions as QuestionWithGuidance[];
  const context: EvalInterviewContext = {
    interviewId: session.id,
    candidateId: session.candidateId,
    companyId: session.companyId,
    roleTitle: session.jobRole.title,
    roleLevel: 'junior',
    interviewType: 'technical',
    mustHaveSkills: session.jobRole.primaryCriteria,
    niceToHaveSkills: session.jobRole.secondaryCriteria,
    companyEvaluationNotes: 'Transcript-only evaluation. Proctoring events are shown separately and do not change technical scoring.',
  };

  const askedQuestionIndexes = new Set(
    transcript
      .filter((entry) => entry?.speaker === 'ai' && isValidQuestionIndex(entry.questionIndex, questions.length))
      .map((entry) => Number(entry.questionIndex)),
  );
  const answeredQuestions = pairAnsweredEvalQuestions(transcript, questions)
    .filter(({ questionIndex }) => askedQuestionIndexes.has(questionIndex));
  const preparedAnswers = answeredQuestions.map(({ question, answer, questionIndex, answerTurn }) => {
    return prepareAnswerEvaluation({
      answerId: `${session.id}-question-${questionIndex + 1}-answer-${answerTurn}`,
      question,
      transcript: answer,
      context,
    });
  });
  const [evaluations, authorshipAssessments] = await Promise.all([
    evaluatePreparedAnswers(context, preparedAnswers),
    analyzeAiAuthorship(preparedAnswers.map((answer) => ({
      answerId: answer.answerId,
      question: answer.questionConfig,
      transcript: answer.transcript,
    }))),
  ]);
  const evaluationsWithAuthorship = evaluations.map((evaluation) => ({
    ...evaluation,
    aiAuthorshipAssessment: authorshipAssessments.get(evaluation.answerId),
  }));
  const report = aggregateEvalCandidateReport(context, evaluationsWithAuthorship);
  const proctoringSummary = {
    eventCount: session.proctoringLogs.length,
    criticalOrHighCount: session.proctoringLogs.filter((log) => ['CRITICAL', 'HIGH'].includes(log.severity)).length,
  };
  const finalReport = {
    ...report,
    proctoringSummary,
  } as EvalCandidateReport & { proctoringSummary: typeof proctoringSummary };

  await prisma.interviewSession.update({
    where: { id: sessionId },
    data: { evaluation: finalReport as any, status: 'EVALUATED', completedAt: new Date() },
  });

  return finalReport;
}

export async function generatePdfReport(sessionId: string) {
  const session = await prisma.interviewSession.findUnique({
    where: { id: sessionId },
    include: { company: true, candidate: true, jobRole: true, proctoringLogs: true },
  });

  if (!session?.evaluation) throw new Error('Run evaluation first');

  const evaluation = session.evaluation as any;
  const outDir = path.resolve('reports');
  fs.mkdirSync(outDir, { recursive: true });
  const filePath = path.join(outDir, `${sessionId}.pdf`);
  const doc = new PDFDocument({ margin: 48 });
  doc.pipe(fs.createWriteStream(filePath));
  doc.fontSize(22).text('IntervieHire Candidate Report');
  doc.moveDown(0.5).fontSize(11).fillColor('#444').text(`${session.candidate.fullName} - ${session.jobRole.title} - ${session.company.name}`);
  doc.moveDown().fillColor('#111').fontSize(16).text(`Overall Score: ${evaluation.overallScore ?? '-'} / 100`);
  doc.fontSize(13).text(`Recommendation: ${(evaluation.recommendation ?? '-').replace?.(/_/g, ' ') ?? evaluation.recommendation}`);
  doc.fontSize(11).text(`Recommendation confidence: ${evaluation.recommendationConfidence ?? '-'}`);
  if (evaluation.candidateConfidence) {
    doc.fontSize(11).text(
      `Expressed confidence: ${evaluation.candidateConfidence.score}/100 (${evaluation.candidateConfidence.level}, ${evaluation.candidateConfidence.reliability} reliability)`,
    );
    doc.fontSize(9).fillColor('#555').text(evaluation.candidateConfidence.summary || '');
    doc.fillColor('#111');
  }
  doc.moveDown();
  doc.fontSize(14).text('Demonstrated Strengths');
  (evaluation.strengths || []).forEach((item: string) => doc.fontSize(10).text(`- ${item}`));
  doc.moveDown().fontSize(14).text('Development Areas');
  (evaluation.weaknesses || []).forEach((item: string) => doc.fontSize(10).text(`- ${item}`));
  doc.moveDown().fontSize(14).text('Question Breakdown');
  (evaluation.questionBreakdown || []).forEach((item: any, index: number) => {
    doc.fontSize(11).fillColor('#111').text(`Q${index + 1}: ${item.questionText || 'Asked question'} (${item.overallScore}/100)`);
    doc.fontSize(9).fillColor('#555').text(item.summary || '').moveDown(0.4);
    if (item.aiAuthorshipAssessment) {
      doc.fontSize(9).fillColor('#111').text(
        `AI-authorship likelihood: ${item.aiAuthorshipAssessment.probability}% (${item.aiAuthorshipAssessment.confidence} confidence)`,
      );
      (item.aiAuthorshipAssessment.reasons || []).forEach((reason: string) => {
        doc.fontSize(8).fillColor('#555').text(`- ${reason}`);
      });
      doc.fontSize(7).fillColor('#777').text(item.aiAuthorshipAssessment.disclaimer || '');
      doc.moveDown(0.4);
    }
  });
  doc.moveDown().fillColor('#111').fontSize(14).text('Proctoring Summary');
  doc.fontSize(10).text(session.proctoringLogs.length ? `${session.proctoringLogs.length} events flagged.` : 'No flagged events.');
  doc.moveDown().fontSize(14).text('Summary');
  doc.fontSize(10).text(evaluation.summary || '');
  doc.end();
  await new Promise(resolve => doc.on('end', resolve));
  await prisma.interviewSession.update({ where: { id: sessionId }, data: { reportUrl: filePath } });
  return filePath;
}

function prepareAnswerEvaluation(params: {
  answerId: string;
  question: QuestionWithGuidance;
  transcript: string;
  context: EvalInterviewContext;
}): PreparedAnswer {
  const guidance = parseEvaluationGuidance(params.question.aiEvaluationGuidance);
  const questionConfig: EvalQuestionConfig = {
    questionId: params.question.id,
    questionText: params.question.text,
    questionType: guidance.questionType ?? 'technical_theory',
    questionOrigin: 'predetermined',
    modelAnswer: guidance.modelAnswer,
    difficulty: params.question.difficulty.toLowerCase() as EvalQuestionConfig['difficulty'],
    skillTags: params.question.topicCategories,
    modelAnswerRubric: guidance.rubric,
  };
  const comparison = compareWithModelAnswer(params.transcript, guidance.rubric);
  const confidence = analyzeTranscriptConfidence(params.transcript);
  const dimensionScores = buildDimensionScores({
    transcript: params.transcript,
    modelAnswer: guidance.modelAnswer,
    comparison,
    confidencePenalty: confidence.confidencePenalty,
  });
  const baseEvaluation: EvalResponseEvaluation = {
    answerId: params.answerId,
    questionId: params.question.id,
    questionText: params.question.text,
    questionOrigin: 'predetermined',
    evaluationMode: 'model_answer_based',
    overallScore: 0,
    modelAnswerComparison: comparison,
    dimensionScores,
    transcriptConfidence: confidence,
    strengths: buildStrengths(comparison),
    weaknesses: buildWeaknesses(comparison),
    redFlags: buildRedFlags(comparison, params.transcript),
    followUpRecommendations: buildFollowUpRecommendations(comparison, params.question.text),
    evaluationConfidence: getEvaluationConfidence(params.transcript, comparison),
    summary: buildAnswerSummary(comparison),
    transcriptOnly: true,
  };
  const calculatedScore = calculateFinalAnswerScore({
    evaluation: baseEvaluation,
    context: params.context,
    question: questionConfig,
  });
  const score = applyAnswerScoreGuardrails(
    calculatedScore,
    baseEvaluation,
    params.transcript,
  );

  const localFallbackEvaluation = {
    ...baseEvaluation,
    overallScore: score,
  };

  return {
    answerId: params.answerId,
    question: params.question,
    questionConfig,
    transcript: params.transcript,
    transcriptConfidence: confidence,
    localFallbackEvaluation,
  };
}

async function evaluatePreparedAnswers(
  context: EvalInterviewContext,
  preparedAnswers: PreparedAnswer[],
): Promise<EvalResponseEvaluation[]> {
  if (preparedAnswers.length === 0) {
    return [];
  }

  const hasDeepSeekKey = Boolean(process.env.DEEPSEEK_API_KEY && process.env.DEEPSEEK_API_KEY !== 'replace-me');

  if (!hasDeepSeekKey) {
    return preparedAnswers.map((answer) => answer.localFallbackEvaluation);
  }

  try {
    const llmEvaluations = await evaluatePreparedAnswersWithDeepSeek(context, preparedAnswers);
    const byAnswerId = new Map(llmEvaluations.map((evaluation) => [evaluation.answerId, evaluation]));

    return preparedAnswers.map((prepared) => {
      const llmEvaluation = byAnswerId.get(prepared.answerId);

      if (!llmEvaluation) {
        return prepared.localFallbackEvaluation;
      }

      return finalizeLlmEvaluation({
        context,
        prepared,
        llmEvaluation,
      });
    });
  } catch (error) {
    console.error('DeepSeek semantic evaluation failed. Falling back to local evaluator.', error);
    return preparedAnswers.map((answer) => answer.localFallbackEvaluation);
  }
}

async function evaluatePreparedAnswersWithDeepSeek(
  context: EvalInterviewContext,
  preparedAnswers: PreparedAnswer[],
): Promise<EvalResponseEvaluation[]> {
  const response = await callDeepSeekJson<LlmEvaluationResponse>({
    systemInstruction: [
      'You are a rigorous but fair hiring evaluator.',
      'Evaluate candidate answers semantically using the role, question, model answer, and rubric.',
      'Do not require exact wording. Credit equivalent correct explanations, synonyms, examples, and valid alternative approaches.',
      'The model answer defines expected substance, but it is not a script and is not necessarily the only valid solution.',
      'Penalize missing concepts, factual errors, contradictions, vague buzzwords, and non-answers.',
      'Judge each answer independently. Never transfer credit or evidence between questions.',
      'Use only transcript text as evidence. Do not infer tone, accent, body language, or audio/video confidence.',
      'Return strict JSON matching the requested schema.',
    ].join(' '),
    prompt: buildBatchEvaluationPrompt(context, preparedAnswers),
    maxOutputTokens: Number(process.env.DEEPSEEK_EVALUATION_MAX_TOKENS || 12000),
    temperature: 0.1,
  });

  return Array.isArray(response.responseEvaluations) ? response.responseEvaluations : [];
}

function finalizeLlmEvaluation(params: {
  context: EvalInterviewContext;
  prepared: PreparedAnswer;
  llmEvaluation: EvalResponseEvaluation;
}): EvalResponseEvaluation {
  const reconciledComparison = reconcileModelAnswerComparison(
    params.llmEvaluation.modelAnswerComparison,
    params.prepared.questionConfig.modelAnswerRubric,
  );
  const normalized = normalizeEvalResponseEvaluation({
    ...params.llmEvaluation,
    modelAnswerComparison: reconciledComparison,
    answerId: params.prepared.answerId,
    questionId: params.prepared.question.id,
    questionText: params.prepared.question.text,
    questionOrigin: 'predetermined',
    evaluationMode: 'model_answer_based',
    transcriptConfidence: params.prepared.transcriptConfidence,
    transcriptOnly: true,
  });
  const scoreBeforeConfidencePenalty = calculateFinalAnswerScore({
    evaluation: normalized,
    context: params.context,
    question: params.prepared.questionConfig,
  });
  const finalScore = applyAnswerScoreGuardrails(
    scoreBeforeConfidencePenalty,
    normalized,
    params.prepared.transcript,
  );

  return {
    ...normalized,
    overallScore: finalScore,
    strengths: normalized.strengths.length ? normalized.strengths : params.prepared.localFallbackEvaluation.strengths,
    weaknesses: normalized.weaknesses,
    redFlags: normalized.redFlags,
    followUpRecommendations: normalized.followUpRecommendations.length
      ? normalized.followUpRecommendations
      : params.prepared.localFallbackEvaluation.followUpRecommendations,
    summary: normalized.summary || params.prepared.localFallbackEvaluation.summary,
  };
}

function buildBatchEvaluationPrompt(
  context: EvalInterviewContext,
  preparedAnswers: PreparedAnswer[],
): string {
  return [
    'Evaluate all interview answers in one batch.',
    '',
    'Important scoring rules:',
    '- Compare candidate answers to the model answer by meaning, not exact words.',
    '- The model answer is a reference answer, not the only acceptable phrasing.',
    '- Award full credit when the candidate explains the same concept with different wording.',
    '- Award credit for valid alternative implementations or examples when they satisfy the question.',
    '- Evaluate every rubric point exactly once. Preserve its point id, description, and weight; do not invent or omit points.',
    '- Point status scale: full = complete and correct understanding (100); partial = correct but incomplete understanding (50); missing = not demonstrated (0); contradicted = an incompatible claim was made (0).',
    '- A keyword mention without a correct explanation is not full coverage.',
    '- A concise answer can score highly when it fully answers the question; length alone is not quality.',
    '- A polished or verbose answer must not score highly when core concepts are absent or wrong.',
    '- Record each factual contradiction in incorrectClaims with the candidate claim and its correction.',
    '- Evidence must be short snippets or concise paraphrases from the candidate transcript.',
    '- Do not score filler words, pauses, or vocal confidence. Those are handled separately by local code.',
    '- If the transcript is too short or empty, mark confidence low and score accordingly.',
    '',
    `Interview context: ${JSON.stringify(context)}`,
    '',
    `Answers to evaluate: ${JSON.stringify(preparedAnswers.map((answer) => ({
      answerId: answer.answerId,
      questionId: answer.question.id,
      questionText: answer.question.text,
      questionType: answer.questionConfig.questionType,
      difficulty: answer.questionConfig.difficulty,
      skillTags: answer.questionConfig.skillTags,
      modelAnswer: answer.questionConfig.modelAnswer,
      rubric: answer.questionConfig.modelAnswerRubric,
      candidateTranscript: answer.transcript,
      localTranscriptConfidence: answer.transcriptConfidence,
    })))}`,
    '',
    'Return JSON only in this shape:',
    JSON.stringify({
      responseEvaluations: [
        {
          answerId: 'same answerId from input',
          questionId: 'same questionId from input',
          questionText: 'same questionText from input',
          questionOrigin: 'predetermined',
          evaluationMode: 'model_answer_based',
          overallScore: 0,
          modelAnswerComparison: {
            requiredPointCoverage: [
              {
                pointId: 'rubric point id',
                description: 'rubric point description',
                weight: 30,
                status: 'full | partial | missing | contradicted',
                score: '100 for full, 50 for partial, 0 for missing or contradicted',
                evidence: ['short transcript evidence'],
                comment: 'why this status was assigned',
              },
            ],
            secondaryPointCoverage: [],
            excellentSignalCoverage: [],
            incorrectClaims: [
              {
                claim: 'incorrect candidate claim',
                severity: 'low | medium | high | critical',
                correction: 'correct version',
              },
            ],
            coverageScore: 0,
          },
          dimensionScores: {
            model_answer_alignment: {
              score: 0,
              reason: 'semantic alignment with the model answer and rubric',
              evidence: ['short transcript evidence'],
              missing: ['missing concept'],
            },
            factual_correctness: {
              score: 0,
              reason: 'accuracy assessment',
              evidence: ['short transcript evidence'],
              missing: ['incorrect or unsupported idea'],
            },
            concept_coverage: {
              score: 0,
              reason: 'coverage of required and secondary concepts',
              evidence: ['short transcript evidence'],
              missing: ['missing concept'],
            },
            reasoning_quality: {
              score: 0,
              reason: 'quality of explanation, tradeoffs, examples, and justification',
              evidence: ['short transcript evidence'],
              missing: ['weak reasoning element'],
            },
            technical_specificity: {
              score: 0,
              reason: 'precision of technical detail',
              evidence: ['short transcript evidence'],
              missing: ['missing technical detail'],
            },
            clarity_structure: {
              score: 0,
              reason: 'clarity and organization of transcript answer',
              evidence: ['short transcript evidence'],
              missing: ['clarity issue'],
            },
            communication_quality: {
              score: 0,
              reason: 'written transcript communication only, excluding filler words',
              evidence: ['short transcript evidence'],
              missing: ['communication issue'],
            },
          },
          strengths: ['specific strength'],
          weaknesses: ['specific weakness'],
          redFlags: [
            {
              label: 'red flag label',
              severity: 'low | medium | high | critical',
              reason: 'why this matters',
            },
          ],
          followUpRecommendations: ['suggested probe'],
          evaluationConfidence: 'high | medium | low',
          summary: 'short answer-level summary',
          transcriptOnly: true,
        },
      ],
    }),
  ].join('\n');
}

function mergeUniqueStrings(values: string[]): string[] {
  const seen = new Set<string>();
  const merged: string[] = [];

  for (const value of values) {
    const normalized = value.trim();

    if (!normalized || seen.has(normalized.toLowerCase())) {
      continue;
    }

    seen.add(normalized.toLowerCase());
    merged.push(normalized);
  }

  return merged;
}

function parseEvaluationGuidance(raw: string): {
  modelAnswer: string;
  rubric: EvalModelAnswerRubric;
  questionType?: EvalQuestionConfig['questionType'];
} {
  const rawText = typeof raw === 'string' ? raw.trim() : '';

  try {
    const parsed = JSON.parse(raw);
    const modelAnswer = typeof parsed?.modelAnswer === 'string' && parsed.modelAnswer.trim()
      ? parsed.modelAnswer.trim()
      : rawText;

    return {
      modelAnswer,
      rubric: normalizeStoredRubric(parsed?.rubric, modelAnswer),
      questionType: normalizeQuestionType(parsed?.questionType),
    };
  } catch {
    const modelAnswer = rawText || 'No model answer guidance was provided for this question.';

    return {
      modelAnswer,
      questionType: 'technical_theory',
      rubric: buildRubricFromText(modelAnswer),
    };
  }
}

function normalizeStoredRubric(
  rawRubric: unknown,
  modelAnswer: string,
): EvalModelAnswerRubric {
  if (!rawRubric || typeof rawRubric !== 'object') {
    return buildRubricFromText(modelAnswer);
  }

  const rubric = rawRubric as Partial<EvalModelAnswerRubric>;
  const normalized: EvalModelAnswerRubric = {
    requiredPoints: normalizeRubricPoints(rubric.requiredPoints),
    secondaryPoints: normalizeRubricPoints(rubric.secondaryPoints),
    excellentAnswerSignals: normalizeRubricPoints(rubric.excellentAnswerSignals),
    redFlags: normalizeRubricRedFlags(rubric.redFlags),
    notes: typeof rubric.notes === 'string' ? rubric.notes.trim() : undefined,
  };

  return normalized.requiredPoints.length ? normalized : buildRubricFromText(modelAnswer);
}

function normalizeRubricPoints(
  points: EvalModelAnswerRubric['requiredPoints'] | undefined,
): EvalModelAnswerRubric['requiredPoints'] {
  return (Array.isArray(points) ? points : [])
    .map((point, index) => ({
      id: typeof point?.id === 'string' && point.id.trim()
        ? point.id.trim()
        : `rubric_point_${index + 1}`,
      description: typeof point?.description === 'string' ? point.description.trim() : '',
      keywords: Array.isArray(point?.keywords)
        ? point.keywords.map((keyword) => String(keyword).trim()).filter(Boolean).slice(0, 12)
        : undefined,
      weight: Number.isFinite(point?.weight) && Number(point?.weight) > 0
        ? Number(point.weight)
        : 1,
    }))
    .filter((point) => point.description);
}

function normalizeRubricRedFlags(
  redFlags: EvalModelAnswerRubric['redFlags'] | undefined,
): EvalModelAnswerRubric['redFlags'] {
  return (Array.isArray(redFlags) ? redFlags : [])
    .map((flag, index) => ({
      id: typeof flag?.id === 'string' && flag.id.trim()
        ? flag.id.trim()
        : `red_flag_${index + 1}`,
      description: typeof flag?.description === 'string' ? flag.description.trim() : '',
      severity: normalizeRedFlagSeverity(flag?.severity),
    }))
    .filter((flag) => flag.description);
}

function normalizeQuestionType(
  questionType: unknown,
): EvalQuestionConfig['questionType'] | undefined {
  const validQuestionTypes = new Set<EvalQuestionConfig['questionType']>([
    'technical_theory',
    'coding',
    'system_design',
    'behavioral',
    'case_study',
    'sales_roleplay',
    'hr_screening',
    'general',
    'followup',
    'custom',
  ]);

  return validQuestionTypes.has(questionType as EvalQuestionConfig['questionType'])
    ? questionType as EvalQuestionConfig['questionType']
    : undefined;
}

function normalizeRedFlagSeverity(severity: unknown): EvalRedFlagSeverity {
  return severity === 'low' || severity === 'medium' || severity === 'high' || severity === 'critical'
    ? severity
    : 'medium';
}

function buildRubricFromText(modelAnswer: string): EvalModelAnswerRubric {
  const keywords = extractKeywords(modelAnswer).slice(0, 8);

  return {
    requiredPoints: [
      {
        id: 'model_answer_core',
        description: modelAnswer,
        keywords,
        weight: 100,
      },
    ],
    secondaryPoints: [],
    excellentAnswerSignals: [],
    redFlags: [],
  };
}

function compareWithModelAnswer(
  transcript: string,
  rubric: EvalModelAnswerRubric,
): EvalModelAnswerComparison {
  const requiredPointCoverage = rubric.requiredPoints.map((point) => scorePointCoverage(transcript, point));
  const secondaryPointCoverage = rubric.secondaryPoints.map((point) => scorePointCoverage(transcript, point));
  const excellentSignalCoverage = rubric.excellentAnswerSignals.map((point) => scorePointCoverage(transcript, point));
  const incorrectClaims = detectIncorrectClaims(transcript, rubric.redFlags);
  const coverageScore = weightedCoverage([
    ...requiredPointCoverage.map((point) => ({ score: point.score, weight: point.weight ?? 3 })),
    ...secondaryPointCoverage.map((point) => ({ score: point.score, weight: point.weight ?? 1.5 })),
    ...excellentSignalCoverage.map((point) => ({ score: point.score, weight: point.weight ?? 1 })),
  ]);

  return {
    requiredPointCoverage,
    secondaryPointCoverage,
    excellentSignalCoverage,
    incorrectClaims,
    coverageScore,
  };
}

function scorePointCoverage(transcript: string, point: EvalModelAnswerRubric['requiredPoints'][number]): EvalPointCoverage {
  const keywords = point.keywords?.length ? point.keywords : extractKeywords(point.description);
  const matches = keywords.map((keyword) => getConceptMatch(transcript, keyword));
  const matched = matches.filter((match) => match.matched).map((match) => match.evidence);
  const weightedMatches = matches.reduce((sum, match) => sum + match.score, 0);
  const ratio = keywords.length === 0 ? 0 : weightedMatches / keywords.length;
  const descriptionMatch = getDescriptionConceptMatch(transcript, point.description);
  const finalRatio = Math.max(ratio, descriptionMatch);
  const status = finalRatio >= 0.55 ? 'full' : finalRatio >= 0.2 ? 'partial' : 'missing';
  const score = status === 'full' ? Math.round(82 + Math.min(18, finalRatio * 18)) : status === 'partial' ? Math.round(35 + finalRatio * 70) : 0;

  return {
    pointId: point.id,
    description: point.description,
    weight: point.weight,
    status,
    score,
    evidence: Array.from(new Set(matched)).slice(0, 6),
    comment: matched.length
      ? `Matched concept evidence: ${Array.from(new Set(matched)).slice(0, 6).join(', ')}`
      : 'Expected concept was not clearly found in the transcript.',
  };
}

function reconcileModelAnswerComparison(
  llmComparison: EvalModelAnswerComparison | undefined,
  rubric: EvalModelAnswerRubric | undefined,
): EvalModelAnswerComparison {
  if (!rubric) {
    return llmComparison ?? {
      requiredPointCoverage: [],
      secondaryPointCoverage: [],
      excellentSignalCoverage: [],
      incorrectClaims: [],
      coverageScore: 0,
    };
  }

  const reconcilePoints = (
    expectedPoints: EvalModelAnswerRubric['requiredPoints'],
    returnedPoints: EvalPointCoverage[] | undefined,
  ): EvalPointCoverage[] => {
    const returnedById = new Map((returnedPoints ?? []).map((point) => [point.pointId, point]));

    return expectedPoints.map((expected) => {
      const returned = returnedById.get(expected.id);
      const status = normalizeCoverageStatus(returned?.status);

      return {
        pointId: expected.id,
        description: expected.description,
        weight: expected.weight,
        status,
        score: coverageStatusScore(status),
        evidence: (returned?.evidence ?? []).map((item) => item.trim()).filter(Boolean).slice(0, 4),
        comment: returned?.comment?.trim()
          || 'The evaluator did not provide a specific justification for this rubric point.',
      };
    });
  };

  const requiredPointCoverage = reconcilePoints(
    rubric.requiredPoints,
    llmComparison?.requiredPointCoverage,
  );
  const secondaryPointCoverage = reconcilePoints(
    rubric.secondaryPoints,
    llmComparison?.secondaryPointCoverage,
  );
  const excellentSignalCoverage = reconcilePoints(
    rubric.excellentAnswerSignals,
    llmComparison?.excellentSignalCoverage,
  );

  return {
    requiredPointCoverage,
    secondaryPointCoverage,
    excellentSignalCoverage,
    incorrectClaims: llmComparison?.incorrectClaims ?? [],
    coverageScore: weightedCoverage([
      ...requiredPointCoverage.map((point) => ({ score: point.score, weight: point.weight ?? 3 })),
      ...secondaryPointCoverage.map((point) => ({ score: point.score, weight: point.weight ?? 1.5 })),
      ...excellentSignalCoverage.map((point) => ({ score: point.score, weight: point.weight ?? 1 })),
    ]),
  };
}

function normalizeCoverageStatus(status: EvalPointCoverage['status'] | undefined): EvalPointCoverage['status'] {
  return status === 'full' || status === 'partial' || status === 'contradicted' ? status : 'missing';
}

function coverageStatusScore(status: EvalPointCoverage['status']): number {
  return status === 'full' ? 100 : status === 'partial' ? 50 : 0;
}

function applyAnswerScoreGuardrails(
  score: number,
  evaluation: EvalResponseEvaluation,
  transcript: string,
): number {
  const required = evaluation.modelAnswerComparison.requiredPointCoverage;

  if (countWords(transcript) < 5) {
    return Math.min(score, 20);
  }

  if (!required.length) {
    return score;
  }

  const contradictedCount = required.filter((point) => point.status === 'contradicted').length;
  const missingCount = required.filter((point) => point.status === 'missing').length;
  const fullCount = required.filter((point) => point.status === 'full').length;

  if (contradictedCount > 0) {
    return Math.min(score, 49);
  }

  if (missingCount / required.length >= 0.5) {
    return Math.min(score, 59);
  }

  if (fullCount === 0) {
    return Math.min(score, 69);
  }

  return score;
}

function detectIncorrectClaims(
  transcript: string,
  redFlags: EvalExpectedRedFlag[],
): EvalModelAnswerComparison['incorrectClaims'] {
  return redFlags
    .filter((flag) => extractKeywords(flag.description).some((keyword) => includesConcept(transcript, keyword)))
    .map((flag) => ({
      claim: flag.description,
      severity: flag.severity,
      correction: 'Review the model answer and probe the candidate for clarification.',
    }));
}

function buildDimensionScores(params: {
  transcript: string;
  modelAnswer: string;
  comparison: EvalModelAnswerComparison;
  confidencePenalty: number;
}): Record<string, EvalDimensionScore> {
  const wordCount = countWords(params.transcript);
  const reasoningMarkers = countMatches(params.transcript, ['because', 'therefore', 'tradeoff', 'for example', 'edge case', 'complexity', 'depends']);
  const modelKeywordCoverage = keywordCoverage(params.transcript, params.modelAnswer);
  const completenessScore = params.comparison.coverageScore;
  const clarityScore = Math.max(35, Math.min(100, 70 + Math.min(20, reasoningMarkers * 4) - (wordCount < 20 ? 20 : 0)));
  const communicationScore = Math.max(20, 90 - params.confidencePenalty * 3 - (wordCount < 15 ? 15 : 0));

  return {
    model_answer_alignment: {
      score: params.comparison.coverageScore,
      reason: 'Measures how well the response covered the model answer concepts.',
      evidence: collectCoverageEvidence(params.comparison),
      missing: collectMissingPoints(params.comparison),
    },
    factual_correctness: {
      score: Math.max(0, params.comparison.coverageScore - severityPenalty(params.comparison)),
      reason: 'Penalizes detected contradictions or incorrect claims against the expected answer.',
      evidence: collectCoverageEvidence(params.comparison),
      missing: params.comparison.incorrectClaims.map((claim) => claim.claim),
    },
    concept_coverage: {
      score: completenessScore,
      reason: 'Measures required, secondary, and excellent-answer concept coverage.',
      evidence: collectCoverageEvidence(params.comparison),
      missing: collectMissingPoints(params.comparison),
    },
    reasoning_quality: {
      score: Math.max(25, Math.min(100, 45 + reasoningMarkers * 10 + (wordCount > 45 ? 15 : 0))),
      reason: 'Rewards explanation, examples, tradeoffs, and explicit reasoning.',
      evidence: [],
      missing: reasoningMarkers < 2 ? ['Add more reasoning, examples, or tradeoff discussion.'] : [],
    },
    technical_specificity: {
      score: Math.max(20, Math.round(modelKeywordCoverage * 100)),
      reason: 'Measures use of concrete technical terms from the expected answer.',
      evidence: [],
      missing: modelKeywordCoverage < 0.4 ? ['Use more precise technical details from the expected answer.'] : [],
    },
    clarity_structure: {
      score: clarityScore,
      reason: 'Rewards a clear, structured answer with sufficient substance.',
      evidence: [],
      missing: wordCount < 20 ? ['Answer was very short.'] : [],
    },
    communication_quality: {
      score: communicationScore,
      reason: 'Assesses written communication clarity with a small, conservative adjustment for reliable transcript hesitation markers.',
      evidence: [],
      missing: params.confidencePenalty ? ['Transcript hesitation markers reduced this score slightly.'] : [],
    },
  };
}

function buildStrengths(comparison: EvalModelAnswerComparison): string[] {
  const strengths = comparison.requiredPointCoverage
    .filter((point) => point.status === 'full')
    .slice(0, 3)
    .map((point) => `Covered expected concept: ${point.description}`);

  return strengths.length ? strengths : ['Some relevant concepts were attempted, but coverage was limited.'];
}

function buildWeaknesses(
  comparison: EvalModelAnswerComparison,
): string[] {
  const weaknesses = collectMissingPoints(comparison).slice(0, 4);

  return weaknesses.length ? weaknesses : ['No major weakness detected for this answer.'];
}

function buildRedFlags(
  comparison: EvalModelAnswerComparison,
  transcript: string,
): EvalResponseEvaluation['redFlags'] {
  const flags = comparison.incorrectClaims.map((claim) => ({
    label: 'Possible incorrect claim',
    severity: claim.severity,
    reason: claim.claim,
  }));

  if (countWords(transcript) < 5) {
    flags.push({
      label: 'Very short answer',
      severity: 'medium' as EvalRedFlagSeverity,
      reason: 'The answer was too short to evaluate with high confidence.',
    });
  }

  return flags;
}

function buildFollowUpRecommendations(
  comparison: EvalModelAnswerComparison,
  questionText: string,
): string[] {
  const missing = collectMissingPoints(comparison).slice(0, 2);

  if (!missing.length) {
    return [`Ask the candidate to apply this answer to a concrete example related to: ${questionText}`];
  }

  return missing.map((item) => `Probe further: ${item}`);
}

function getEvaluationConfidence(
  transcript: string,
  comparison: EvalModelAnswerComparison,
): EvalResponseEvaluation['evaluationConfidence'] {
  if (countWords(transcript) < 10) return 'low';
  if (comparison.requiredPointCoverage.every((point) => point.status === 'missing')) return 'low';
  if (comparison.requiredPointCoverage.some((point) => point.status === 'missing')) return 'medium';
  return 'high';
}

function buildAnswerSummary(comparison: EvalModelAnswerComparison): string {
  const covered = comparison.requiredPointCoverage.filter((point) => point.status === 'full').length;
  const total = comparison.requiredPointCoverage.length;
  return `Covered ${covered}/${total} required model-answer concepts. Expressed confidence is reported separately from answer correctness.`;
}

function normalizeTranscript(raw: unknown): TranscriptEntry[] {
  if (Array.isArray(raw)) return raw as TranscriptEntry[];
  if (typeof raw === 'string') {
    try {
      const parsed = JSON.parse(raw);
      return Array.isArray(parsed) ? parsed : [];
    } catch {
      return [];
    }
  }
  return [];
}

function isValidQuestionIndex(value: unknown, questionCount: number): value is number {
  return Number.isInteger(value) && Number(value) >= 0 && Number(value) < questionCount;
}

function collectCoverageEvidence(comparison: EvalModelAnswerComparison): string[] {
  return [
    ...comparison.requiredPointCoverage,
    ...comparison.secondaryPointCoverage,
    ...comparison.excellentSignalCoverage,
  ].flatMap((point) => point.evidence).slice(0, 8);
}

function collectMissingPoints(comparison: EvalModelAnswerComparison): string[] {
  return [
    ...comparison.requiredPointCoverage,
    ...comparison.secondaryPointCoverage,
  ]
    .filter((point) => point.status === 'missing' || point.status === 'partial')
    .map((point) => point.description);
}

function severityPenalty(comparison: EvalModelAnswerComparison): number {
  return comparison.incorrectClaims.reduce((penalty, claim) => {
    if (claim.severity === 'critical') return penalty + 40;
    if (claim.severity === 'high') return penalty + 25;
    if (claim.severity === 'medium') return penalty + 12;
    return penalty + 4;
  }, 0);
}

function weightedCoverage(values: Array<{ score: number; weight: number }>): number {
  const totalWeight = values.reduce((sum, item) => sum + item.weight, 0);
  if (!totalWeight) return 0;
  return Math.round(values.reduce((sum, item) => sum + item.score * item.weight, 0) / totalWeight);
}

function keywordCoverage(transcript: string, modelAnswer: string): number {
  const keywords = extractKeywords(modelAnswer).slice(0, 12);
  if (!keywords.length) return 0;
  return keywords.reduce((sum, keyword) => sum + getConceptMatch(transcript, keyword).score, 0) / keywords.length;
}

function extractKeywords(text: string): string[] {
  const stop = new Set(['the', 'and', 'for', 'that', 'with', 'from', 'this', 'into', 'are', 'you', 'your', 'can', 'has', 'have', 'should', 'would', 'when', 'then', 'than', 'they', 'them', 'their', 'because', 'about']);
  const words = text.toLowerCase().match(/[a-z][a-z0-9+.#-]*/g) ?? [];
  return Array.from(new Set(words.filter((word) => word.length > 2 && !stop.has(word))));
}

function includesConcept(transcript: string, keyword: string): boolean {
  return getConceptMatch(transcript, keyword).matched;
}

function getConceptMatch(transcript: string, keyword: string): { matched: boolean; score: number; evidence: string } {
  const normalizedTranscript = normalizeForConceptMatch(transcript);
  const normalizedKeyword = normalizeForConceptMatch(keyword);
  const variants = getConceptVariants(normalizedKeyword);

  for (const variant of variants) {
    if (normalizedTranscript.includes(variant)) {
      return { matched: true, score: 1, evidence: keyword };
    }
  }

  const transcriptTokens = tokenizeForConceptMatch(normalizedTranscript);
  const keywordTokens = tokenizeForConceptMatch(normalizedKeyword);

  if (!keywordTokens.length) {
    return { matched: false, score: 0, evidence: keyword };
  }

  const tokenMatches = keywordTokens.filter((token) => {
    const variantsForToken = getConceptVariants(token).flatMap(tokenizeForConceptMatch);
    const acceptableTokens = new Set([token, ...variantsForToken].map(stemToken));
    return transcriptTokens.some((candidate) => acceptableTokens.has(stemToken(candidate)));
  }).length;
  const tokenRatio = tokenMatches / keywordTokens.length;

  if (tokenRatio >= 0.75) {
    return { matched: true, score: 0.85, evidence: keyword };
  }

  if (tokenRatio >= 0.5) {
    return { matched: true, score: 0.6, evidence: keyword };
  }

  return { matched: false, score: tokenRatio * 0.5, evidence: keyword };
}

function getDescriptionConceptMatch(transcript: string, description: string): number {
  const descriptionKeywords = extractKeywords(description).slice(0, 10);

  if (!descriptionKeywords.length) {
    return 0;
  }

  return descriptionKeywords.reduce((sum, keyword) => sum + getConceptMatch(transcript, keyword).score, 0) / descriptionKeywords.length;
}

function normalizeForConceptMatch(value: string): string {
  return value
    .toLowerCase()
    .replace(/o\s*\(\s*1\s*\)/g, 'constant time')
    .replace(/o\s*\(\s*n\s*\)/g, 'linear time')
    .replace(/o\s*\(\s*log\s*n\s*\)/g, 'logarithmic time')
    .replace(/o\s*\(\s*n\s*\^\s*2\s*\)|o\s*\(\s*n\s*squared\s*\)/g, 'quadratic time')
    .replace(/[^a-z0-9+#.\s-]/g, ' ')
    .replace(/\s+/g, ' ')
    .trim();
}

function tokenizeForConceptMatch(value: string): string[] {
  return value.match(/[a-z0-9+#.]+/g) ?? [];
}

function stemToken(token: string): string {
  if (token.length <= 4) return token;
  return token
    .replace(/(ing|edly|edly|ed|ly)$/g, '')
    .replace(/(tion|ions)$/g, 't')
    .replace(/(ies)$/g, 'y')
    .replace(/(s)$/g, '');
}

function getConceptVariants(keyword: string): string[] {
  const normalized = normalizeForConceptMatch(keyword);
  const variants = new Set<string>([normalized]);

  for (const group of CONCEPT_SYNONYM_GROUPS) {
    if (group.some((item) => normalized.includes(item))) {
      group.forEach((item) => variants.add(item));
    }
  }

  return Array.from(variants);
}

function countWords(text: string): number {
  return text.trim().split(/\s+/).filter(Boolean).length;
}

function countMatches(text: string, needles: string[]): number {
  const normalized = text.toLowerCase();
  return needles.filter((needle) => normalized.includes(needle)).length;
}

const CONCEPT_SYNONYM_GROUPS = [
  ['contiguous', 'continuous', 'adjacent', 'next to each other', 'single block', 'memory block'],
  ['index', 'indexed', 'position', 'direct access', 'random access', 'access by position'],
  ['constant time', 'o(1)', 'fixed time', 'same time'],
  ['linear time', 'o(n)', 'one by one', 'scan', 'traverse', 'walk through'],
  ['logarithmic time', 'o(log n)', 'binary search', 'halve', 'divide'],
  ['quadratic time', 'o(n^2)', 'n squared', 'nested loop'],
  ['node', 'element object', 'record'],
  ['pointer', 'reference', 'link', 'next address', 'next node'],
  ['insert', 'insertion', 'add', 'append'],
  ['delete', 'deletion', 'remove'],
  ['lookup', 'access', 'retrieve', 'find', 'read'],
  ['memory', 'space', 'storage', 'overhead', 'extra space'],
  ['cache', 'cache locality', 'locality', 'cpu cache'],
  ['input', 'input size', 'data size', 'number of items', 'amount of data'],
  ['grow', 'growth', 'increase', 'scale'],
  ['runtime', 'running time', 'execution time', 'time taken', 'speed'],
  ['big o', 'asymptotic', 'order of growth', 'complexity class'],
  ['efficient', 'efficiency', 'performant', 'faster', 'optimized'],
  ['reproduce', 'replicate', 'try the same request', 'simulate'],
  ['logs', 'logging', 'log file', 'console output'],
  ['stack trace', 'traceback', 'error trace', 'exception trace'],
  ['request', 'payload', 'body', 'params', 'headers'],
  ['database', 'db', 'query', 'sql'],
  ['dependency', 'downstream service', 'external service', 'third party'],
  ['isolate', 'narrow down', 'pinpoint', 'find the failing layer'],
  ['debugger', 'breakpoint', 'step through'],
  ['test', 'unit test', 'integration test', 'verify'],
  ['api', 'endpoint', 'route', 'handler'],
  ['redirect', 'forward', '302', '301'],
  ['store', 'save', 'persist', 'write'],
  ['mapping', 'lookup table', 'key value', 'dictionary'],
  ['unique', 'non duplicate', 'distinct'],
  ['collision', 'duplicate', 'conflict'],
  ['hash', 'hashing', 'random id', 'short code', 'base62'],
  ['analytics', 'metrics', 'tracking', 'click count'],
  ['expiry', 'expiration', 'ttl', 'time to live'],
  ['validation', 'validate', 'sanitize', 'check input'],
  ['abuse', 'spam', 'malicious', 'rate limit', 'throttle'],
];
