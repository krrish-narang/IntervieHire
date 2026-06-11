import { prisma } from '../lib/prisma.js';

export async function handleCandidateTranscript(
  sessionId: string,
  text: string,
  metrics: Record<string, unknown> = {},
) {
  const session = await prisma.interviewSession.findUnique({
    where: { id: sessionId },
    include: {
      company: true,
      jobRole: { include: { questions: { where: { isActive: true }, orderBy: { createdAt: 'asc' } } } },
      candidate: true,
    },
  });

  if (!session) throw new Error('Interview session not found');

  const transcript = Array.isArray(session.transcript) ? session.transcript as any[] : [];
  const activeQuestionIndex = [...transcript]
    .reverse()
    .find((entry) => entry?.speaker === 'ai' && Number.isInteger(entry?.questionIndex))
    ?.questionIndex;
  const answeredCount = transcript.filter((entry) => entry?.speaker === 'candidate').length;
  const questionIndex = Number.isInteger(activeQuestionIndex)
    ? activeQuestionIndex
    : answeredCount;

  transcript.push({
    speaker: 'candidate',
    text,
    timestamp: new Date().toISOString(),
    metrics,
    questionIndex,
  });

  const nextQuestionIndex = questionIndex + 1;
  const nextQuestion = session.jobRole.questions[nextQuestionIndex]?.text;
  const aiText = nextQuestion
    ? nextQuestion
    : 'Thanks. That completes the structured interview. You can click Complete session when you are ready to see the report.';

  transcript.push({
    speaker: 'ai',
    text: aiText,
    timestamp: new Date().toISOString(),
    questionIndex: nextQuestion ? nextQuestionIndex : null,
  });

  await prisma.interviewSession.update({
    where: { id: sessionId },
    data: { transcript, status: 'IN_PROGRESS' },
  });

  return {
    text: aiText,
    interviewPhase: nextQuestion ? 'questioning' : 'closing',
    emotionState: nextQuestion ? 'curious' : 'encouraging',
  } as const;
}
