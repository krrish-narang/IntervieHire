import { FastifyInstance } from 'fastify';
import { prisma } from '../lib/prisma.js';
import { evaluateInterview, generatePdfReport } from '../services/evaluation.service.js';
import nodemailer from 'nodemailer';
import fs from 'node:fs';
import path from 'node:path';
import { buildVapiAssistantConfig } from '../services/vapi-config.service.js';
import { processRecordingForSession } from '../services/transcription.service.js';

export async function interviewRoutes(app: FastifyInstance) {
  app.get('/demo-session', async () => {
    const company = await prisma.company.upsert({
      where: { slug: 'demo-consulting' },
      update: {},
      create: {
        name: 'Demo Consulting Group',
        slug: 'demo-consulting',
        description: 'A strategy and operations firm hiring for client-facing problem solvers.',
        reportEmail: 'hr@example.com',
        primaryColor: '#0e7490',
      },
    });

    let role = await prisma.jobRole.findFirst({ where: { companyId: company.id, title: 'Associate Consultant' } });
    if (!role) {
      role = await prisma.jobRole.create({
        data: {
          companyId: company.id,
          title: 'Associate Consultant',
          roleType: 'CONSULTING',
          description: 'Entry-level consulting role across strategy and operations.',
          requirements: 'Structured thinking, client communication, analytics, business judgment.',
          primaryCriteria: ['problem-solving', 'client skills', 'structured thinking'],
          secondaryCriteria: ['industry knowledge', 'analytics', 'presentation'],
          atsScoringWeights: { primary: 0.4, secondary: 0.3, education: 0.1, experience: 0.1, communication: 0.1 },
          evaluationCriteria: { answerDepth: 1, confidence: 1, communication: 1, domainKnowledge: 1, problemSolving: 1 },
        },
      });
    }

    let candidate = await prisma.candidate.findFirst({ where: { companyId: company.id, email: 'aarav@example.com' } });
    if (!candidate) {
      candidate = await prisma.candidate.create({
        data: {
          companyId: company.id,
          fullName: 'Aarav Sharma',
          email: 'aarav@example.com',
          parsedResume: { yearsOfExperience: 2, skills: ['analytics', 'presentation', 'client communication', 'problem-solving'] },
          atsScore: 82,
          atsBreakdown: { demo: true },
        },
      });
    }

    let session = await prisma.interviewSession.findFirst({
      where: { companyId: company.id, candidateId: candidate.id, jobRoleId: role.id, status: 'SCHEDULED' },
      orderBy: { createdAt: 'desc' },
    });
    if (!session) {
      session = await prisma.interviewSession.create({
        data: {
          companyId: company.id,
          candidateId: candidate.id,
          jobRoleId: role.id,
          status: 'SCHEDULED',
          scheduledAt: new Date(),
        },
      });
    }

    return { sessionId: session.id, companyId: company.id, roleId: role.id, candidateId: candidate.id };
  });

  app.get('/sessions/:id', async (req:any) => prisma.interviewSession.findUnique({where:{id:req.params.id}, include:{company:true,candidate:true,jobRole:{include:{questions:true}},proctoringLogs:true}}));
  app.post('/sessions/:id/start', async (req:any) => prisma.interviewSession.update({where:{id:req.params.id}, data:{status:'IN_PROGRESS', startedAt:new Date()}}));
  app.get('/sessions/:id/vapi-config', async (req:any) => {
    const session = await prisma.interviewSession.findUniqueOrThrow({where:{id:req.params.id}, include:{company:true,jobRole:{include:{questions:true}}}});
    return buildVapiAssistantConfig({companyName:session.company.name, companyDescription:session.company.description || undefined, jobRole:session.jobRole.title, roleRequirements:session.jobRole.requirements, questions:session.jobRole.questions.map(q=>q.text), evaluationCriteria:session.jobRole.evaluationCriteria as any});
  });
  app.post('/sessions/:id/complete', async (req:any) => prisma.interviewSession.update({where:{id:req.params.id}, data:{status:'COMPLETED', completedAt:new Date()}}));
  app.post('/sessions/:id/evaluate', async (req:any) => ({evaluation: await evaluateInterview(req.params.id)}));
  app.post('/sessions/:id/report', async (req:any) => ({filePath: await generatePdfReport(req.params.id)}));
  app.post('/sessions/:id/email-report', async (req:any) => {
    const session = await prisma.interviewSession.findUnique({where:{id:req.params.id}, include:{company:true,candidate:true}});
    if (!session) throw new Error('Session not found');
    const filePath = session.reportUrl || await generatePdfReport(req.params.id);
    const transporter = nodemailer.createTransport({host:process.env.SMTP_HOST, port:Number(process.env.SMTP_PORT || 587), secure:false, auth:{user:process.env.SMTP_USER, pass:process.env.SMTP_PASS}});
    await transporter.sendMail({from:process.env.REPORT_FROM, to: session.company.reportEmail || req.body?.to, subject:`Interview report: ${session.candidate.fullName}`, text:'Attached is the IntervieHire evaluation report.', attachments:[{filename:`${session.candidate.fullName}-report.pdf`, content:fs.createReadStream(filePath)}]});
    return {sent:true};
  });

  // Accept a recording upload (audio/video blob) and attach metadata to the session transcript JSON
  app.post('/sessions/:id/recording', async (req:any, reply) => {
    // requires @fastify/multipart registered
    const part = await req.file();
    if (!part) return reply.code(400).send({ error: 'No file uploaded' });
    const uploadsDir = path.resolve(process.cwd(), 'uploads');
    if (!fs.existsSync(uploadsDir)) fs.mkdirSync(uploadsDir, { recursive: true });
    const filename = `${Date.now()}-${part.filename || 'recording.webm'}`;
    const dest = path.join(uploadsDir, filename);
    const buffer = await part.toBuffer();
    fs.writeFileSync(dest, buffer);

    // Attach a recording entry to the session transcript JSON
    const session = await prisma.interviewSession.findUnique({ where: { id: req.params.id } });
    const entry = { type: 'recording', filename, url: `/uploads/${filename}`, createdAt: new Date() };
    if (session) {
      const current = Array.isArray(session?.transcript) ? session?.transcript as any[] : (session?.transcript ? JSON.parse(session.transcript as any) : []);
      const updated = [...current, entry];
      await prisma.interviewSession.update({ where: { id: req.params.id }, data: { transcript: updated as any } });
      // kick off transcription and question-fit processing (async)
      processRecordingForSession(req.params.id, filename).catch((err) => app.log.error('Transcription error', err));
      return { url: `/uploads/${filename}`, entry };
    }

    // If session not found, return upload info — file saved but not attached to a session
    return { url: `/uploads/${filename}`, entry, note: 'session not found; recording stored but not linked' };
  });

  // Serve uploaded files
  app.get('/uploads/:file', async (req:any, reply) => {
    const p = path.join(process.cwd(), 'uploads', req.params.file);
    if (!fs.existsSync(p)) return reply.code(404).send({ error: 'Not found' });
    return reply.send(fs.createReadStream(p));
  });
}
