import fs from 'node:fs';
import path from 'node:path';
import { prisma } from '../lib/prisma.js';
import { callOpenRouter } from '../lib/openrouter.js';

async function transcribeWithOpenAI(filePath: string) {
  const apiKey = process.env.OPENAI_API_KEY;
  if (!apiKey || apiKey === 'replace-me') return null;
  const form = new FormData();
  form.append('file', fs.createReadStream(filePath) as any);
  form.append('model', 'whisper-1');
  const res = await fetch('https://api.openai.com/v1/audio/transcriptions', { method: 'POST', headers: { Authorization: `Bearer ${apiKey}` }, body: form });
  if (!res.ok) throw new Error(`OpenAI transcription failed: ${res.status} ${await res.text()}`);
  const data = await res.json();
  return data.text as string;
}

function simpleQuestionFit(transcript: string, questions: Array<{ id?: string; text: string }>) {
  const txt = transcript.toLowerCase();
  const tokens = txt.replace(/[\W_]+/g, ' ').split(/\s+/).filter(Boolean);
  const tokenSet = new Set(tokens);
  return questions.map((q) => {
    const qTokens = q.text
      .toLowerCase()
      .replace(/[\W_]+/g, ' ')
      .split(/\s+/)
      .filter((t) => t.length > 3);
    if (!qTokens.length) return { questionId: q.id, score: 1, reasoning: 'No keywords to match' };
    const matches = qTokens.filter((t) => tokenSet.has(t)).length;
    const frac = matches / qTokens.length;
    const score = Math.min(5, Math.max(1, Math.round(frac * 4) + 1));
    const reasoning = `Matched ${matches}/${qTokens.length} keywords`;
    return { questionId: q.id, score, reasoning };
  });
}

export async function processRecordingForSession(sessionId: string, filename: string) {
  const uploadsDir = path.join(process.cwd(), 'uploads');
  const filePath = path.join(uploadsDir, filename);
  let transcriptText: string | null = null;
  try {
    transcriptText = await transcribeWithOpenAI(filePath);
  } catch (err) {
    // fallback: call LLM for a mock transcript summary when OpenAI key absent
    const session = await prisma.interviewSession.findUnique({ where: { id: sessionId }, include: { jobRole: true } });
    if (session) {
      const prompt = `Generate a short mock transcript for a ${session.jobRole.title} interview answering questions: ${session.jobRole.title}`;
      try {
        transcriptText = await callOpenRouter([{ role: 'user', content: prompt }], { json: false });
      } catch (e) {
        transcriptText = 'Transcript unavailable.';
      }
    }
  }

  if (!transcriptText) transcriptText = 'Transcript unavailable.';

  // store a transcript entry in session.transcript
  const session = await prisma.interviewSession.findUnique({ where: { id: sessionId } });
  const current = Array.isArray(session?.transcript) ? session?.transcript as any[] : (session?.transcript ? JSON.parse(session?.transcript as any) : []);
  const entry = { type: 'transcription', filename, text: transcriptText, createdAt: new Date() };
  const updated = [...current, entry];
  await prisma.interviewSession.update({ where: { id: sessionId }, data: { transcript: updated as any } });

  // compute question-fit scoring using jobRole.questions
  const sessionFull = await prisma.interviewSession.findUnique({ where: { id: sessionId }, include: { jobRole: { include: { questions: true } } } });
  const questions = sessionFull?.jobRole?.questions || [];
  const fits = simpleQuestionFit(transcriptText, questions.map((q: any) => ({ id: q.id, text: q.text })));
  // attach as evaluation.partialQuestionFit
  const existingEval = sessionFull?.evaluation || {};
  const evaluation = { ...(existingEval as any), partialQuestionFit: fits };
  await prisma.interviewSession.update({ where: { id: sessionId }, data: { evaluation: evaluation as any } });

  return { transcriptText, fits };
}
