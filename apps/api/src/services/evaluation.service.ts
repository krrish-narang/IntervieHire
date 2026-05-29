import PDFDocument from 'pdfkit';
import { prisma } from '../lib/prisma.js';
import { callOpenRouter } from '../lib/openrouter.js';
import type { EvaluationReport } from '@interviehire/shared';
import fs from 'node:fs';
import path from 'node:path';

export async function evaluateInterview(sessionId: string): Promise<EvaluationReport> {
  const session = await prisma.interviewSession.findUnique({where:{id:sessionId}, include:{company:true,candidate:true,jobRole:true,proctoringLogs:true}});
  if (!session) throw new Error('Session not found');
  const prompt = `Evaluate this interview transcript for a ${session.jobRole.title} position at ${session.company.name}. Return a strict JSON object with answerDepth, confidence, communication, domainKnowledge, problemSolving, overallScore, recommendation, strengths, risks, summary. Each metric has score 1-5 and reasoning. Transcript: ${JSON.stringify(session.transcript)}. Proctoring logs: ${JSON.stringify(session.proctoringLogs)}`;
  const raw = await callOpenRouter([{role:'system', content:'You are an objective hiring evaluator. Return valid JSON only.'},{role:'user', content:prompt}], {json:true});
  const evaluation = JSON.parse(raw) as EvaluationReport;
  await prisma.interviewSession.update({where:{id:sessionId}, data:{evaluation: evaluation as any, status:'EVALUATED', completedAt: new Date()}});
  return evaluation;
}

export async function generatePdfReport(sessionId: string) {
  const session = await prisma.interviewSession.findUnique({where:{id:sessionId}, include:{company:true,candidate:true,jobRole:true,proctoringLogs:true}});
  if (!session?.evaluation) throw new Error('Run evaluation first');
  const evaluation = session.evaluation as any;
  const outDir = path.resolve('reports'); fs.mkdirSync(outDir, {recursive:true});
  const filePath = path.join(outDir, `${sessionId}.pdf`);
  const doc = new PDFDocument({ margin: 48 });
  doc.pipe(fs.createWriteStream(filePath));
  doc.fontSize(22).text('IntervieHire Candidate Report', { continued: false });
  doc.moveDown(0.5).fontSize(11).fillColor('#444').text(`${session.candidate.fullName} • ${session.jobRole.title} • ${session.company.name}`);
  doc.moveDown().fillColor('#111').fontSize(16).text(`Overall Score: ${evaluation.overallScore}/5`);
  doc.fontSize(13).text(`Recommendation: ${evaluation.recommendation}`);
  doc.moveDown();
  for (const key of ['answerDepth','confidence','communication','domainKnowledge','problemSolving']) {
    const metric = evaluation[key] || {};
    doc.fontSize(13).fillColor('#111').text(`${key}: ${metric.score ?? '-'} / 5`);
    doc.fontSize(10).fillColor('#555').text(metric.reasoning ?? '-').moveDown(0.5);
  }
  doc.fillColor('#111').fontSize(14).text('Strengths');
  (evaluation.strengths || []).forEach((s: string) => doc.fontSize(10).text(`• ${s}`));
  doc.moveDown().fontSize(14).text('Risks / Follow-up Areas');
  (evaluation.risks || []).forEach((s: string) => doc.fontSize(10).text(`• ${s}`));
  doc.moveDown().fontSize(14).text('Proctoring Summary');
  doc.fontSize(10).text(session.proctoringLogs.length ? `${session.proctoringLogs.length} events flagged.` : 'No flagged events.');
  doc.moveDown().fontSize(14).text('Summary');
  doc.fontSize(10).text(evaluation.summary || '');
  doc.end();
  await new Promise(resolve => doc.on('end', resolve));
  await prisma.interviewSession.update({where:{id:sessionId}, data:{reportUrl:filePath}});
  return filePath;
}
