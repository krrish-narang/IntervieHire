import { prisma } from '../lib/prisma.js';
import { callOpenRouter } from '../lib/openrouter.js';
export async function handleCandidateTranscript(sessionId: string, text: string, metrics: Record<string, unknown> = {}) {
  const session = await prisma.interviewSession.findUnique({where:{id:sessionId}, include:{company:true, jobRole:true, candidate:true}});
  if (!session) throw new Error('Interview session not found');
  const transcript = Array.isArray(session.transcript) ? session.transcript as any[] : [];
  transcript.push({speaker:'candidate', text, timestamp: new Date().toISOString(), metrics});
  const system = `You are a warm but rigorous AI interviewer for ${session.company.name}. Role: ${session.jobRole.title}. Company context: ${session.company.description || 'N/A'}. Ask concise follow-up questions. Probe for examples, metrics, trade-offs, and reflection. Keep one question at a time.`;
  const history = transcript.slice(-8).map((t:any) => ({role: t.speaker === 'candidate' ? 'user' : 'assistant', content: t.text})) as any[];
  const aiText = await callOpenRouter([{role:'system', content:system}, ...history]);
  transcript.push({speaker:'ai', text: aiText, timestamp: new Date().toISOString()});
  await prisma.interviewSession.update({where:{id:sessionId}, data:{transcript, status:'IN_PROGRESS'}});
  return { text: aiText, interviewPhase: transcript.length < 4 ? 'greeting' : 'follow_up', emotionState: 'curious' } as const;
}
