import type {
  EvalFollowupContext,
  EvalQuestionConfig,
  EvalResponseInput,
  EvalTranscriptInput,
  EvalTurnLink,
} from "./types";

export interface EvalTranscriptEntryForPairing {
  speaker?: string;
  text?: string | null;
  questionIndex?: number | null;
}

export interface PairedEvalAnswer<TQuestion> {
  question: TQuestion;
  answer: string;
  questionIndex: number;
  answerTurn: number;
}

export function createEvalResponseInput(params: {
  answerId: string;
  question: EvalQuestionConfig;
  transcript: string;
  language?: string;
  followupContext?: EvalFollowupContext;
}): EvalResponseInput {
  return {
    answerId: params.answerId,
    question: params.question,
    response: createEvalTranscriptInput(params.transcript, params.language),
    followupContext: params.followupContext,
  };
}

export function createEvalTurnLink(params: EvalTurnLink): EvalTurnLink {
  return {
    originalQuestionId: params.originalQuestionId,
    originalAnswerId: params.originalAnswerId,
    followupQuestionId: params.followupQuestionId,
    followupAnswerId: params.followupAnswerId,
  };
}

export function pairAnsweredEvalQuestions<TQuestion>(
  transcript: EvalTranscriptEntryForPairing[],
  questions: TQuestion[],
): Array<PairedEvalAnswer<TQuestion>> {
  const answeredQuestions: Array<PairedEvalAnswer<TQuestion>> = [];
  let activeQuestionIndex: number | null = null;
  let fallbackQuestionIndex = 0;
  let canMergeWithPreviousCandidate = false;

  for (const entry of transcript) {
    if (entry?.speaker === "ai") {
      activeQuestionIndex = isValidPairingQuestionIndex(entry.questionIndex, questions.length)
        ? entry.questionIndex
        : null;
      canMergeWithPreviousCandidate = false;
      continue;
    }

    const answer = entry?.speaker === "candidate" ? String(entry.text ?? "").trim() : "";

    if (!answer) {
      continue;
    }

    const explicitQuestionIndex = isValidPairingQuestionIndex(entry.questionIndex, questions.length)
      ? entry.questionIndex
      : null;
    const questionIndex = explicitQuestionIndex ?? activeQuestionIndex ?? fallbackQuestionIndex;
    const question = questions[questionIndex];

    if (!question) {
      continue;
    }

    const previousAnswer = answeredQuestions[answeredQuestions.length - 1];

    if (
      previousAnswer
      && previousAnswer.questionIndex === questionIndex
      && canMergeWithPreviousCandidate
    ) {
      previousAnswer.answer = mergeAnswerChunks(previousAnswer.answer, answer);
    } else {
      answeredQuestions.push({
        question,
        answer,
        questionIndex,
        answerTurn: answeredQuestions.length + 1,
      });
    }

    fallbackQuestionIndex = Math.max(fallbackQuestionIndex, questionIndex + 1);
    activeQuestionIndex = null;
    canMergeWithPreviousCandidate = true;
  }

  return answeredQuestions;
}

function createEvalTranscriptInput(transcript: string, language?: string): EvalTranscriptInput {
  return {
    source: "transcript",
    transcript,
    language,
  };
}

function isValidPairingQuestionIndex(value: unknown, questionCount: number): value is number {
  return Number.isInteger(value) && Number(value) >= 0 && Number(value) < questionCount;
}

function mergeAnswerChunks(previous: string, next: string): string {
  return [previous, next].filter(Boolean).join("\n").trim();
}
