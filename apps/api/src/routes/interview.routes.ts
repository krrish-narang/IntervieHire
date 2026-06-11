import { FastifyInstance } from 'fastify';
import { prisma } from '../lib/prisma.js';
import { evaluateInterview, generatePdfReport } from '../services/evaluation.service.js';
import nodemailer from 'nodemailer';
import fs from 'node:fs';
import path from 'node:path';
import { buildVapiAssistantConfig } from '../services/vapi-config.service.js';
import { processRecordingForSession, transcribeUploadedFile } from '../services/transcription.service.js';
import { handleCandidateTranscript } from '../services/interview-conversation.service.js';

const juniorSdeQuestions = [
  {
    text: 'Explain the difference between an array and a linked list. When would you choose one over the other?',
    difficulty: 'EASY' as const,
    topicCategories: ['data structures', 'fundamentals'],
    aiEvaluationGuidance: JSON.stringify({
      questionType: 'technical_theory',
      modelAnswer: 'Arrays store elements contiguously and allow O(1) index access, but insertion or deletion in the middle can be O(n). Linked lists store nodes with references, so insertion or deletion can be O(1) when the node is known, but random access is O(n) and there is extra pointer memory overhead. Choose arrays for fast indexing and cache locality; choose linked lists when frequent insertions or deletions are needed and traversal is acceptable.',
      rubric: {
        requiredPoints: [
          { id: 'array_contiguous_indexing', description: 'Arrays use contiguous storage and support fast index-based access.', keywords: ['contiguous', 'index', 'o(1)', 'random access'], weight: 30 },
          { id: 'linked_list_nodes', description: 'Linked lists use nodes and references/pointers rather than contiguous storage.', keywords: ['node', 'pointer', 'reference', 'linked'], weight: 25 },
          { id: 'operation_tradeoffs', description: 'Explains insertion/deletion and access-time tradeoffs.', keywords: ['insert', 'delete', 'o(n)', 'o(1)', 'access'], weight: 30 },
        ],
        secondaryPoints: [
          { id: 'memory_cache_tradeoff', description: 'Mentions memory overhead or cache locality.', keywords: ['memory', 'cache', 'overhead', 'locality'], weight: 10 },
        ],
        excellentAnswerSignals: [
          { id: 'use_case_choice', description: 'Gives a clear rule for choosing one structure over the other.', keywords: ['choose', 'when', 'frequent', 'indexing'], weight: 10 },
        ],
        redFlags: [
          { id: 'claims_same_structure', description: 'Claims arrays and linked lists are essentially the same structure.', severity: 'high' },
        ],
      },
    }),
  },
  {
    text: 'What is time complexity, and why does it matter when writing code?',
    difficulty: 'EASY' as const,
    topicCategories: ['algorithms', 'complexity'],
    aiEvaluationGuidance: JSON.stringify({
      questionType: 'technical_theory',
      modelAnswer: 'Time complexity describes how an algorithm’s running time grows as input size grows, usually using Big O notation such as O(1), O(log n), O(n), or O(n squared). It matters because code that works on small inputs may become too slow on large inputs. Engineers use it to compare approaches, choose efficient algorithms, and understand scalability.',
      rubric: {
        requiredPoints: [
          { id: 'growth_with_input', description: 'Defines complexity as runtime growth relative to input size.', keywords: ['input', 'size', 'grow', 'runtime', 'running time'], weight: 35 },
          { id: 'big_o', description: 'Mentions Big O or common complexity classes.', keywords: ['big o', 'o(n)', 'o(1)', 'o(log', 'o(n^2)'], weight: 25 },
          { id: 'scalability_reason', description: 'Explains why complexity matters for larger inputs and scalability.', keywords: ['large', 'slow', 'scale', 'scalability', 'efficient'], weight: 30 },
        ],
        secondaryPoints: [
          { id: 'compare_approaches', description: 'Uses complexity to compare possible implementations.', keywords: ['compare', 'approach', 'algorithm', 'choose'], weight: 10 },
        ],
        excellentAnswerSignals: [
          { id: 'concrete_example', description: 'Gives a concrete example such as nested loops or binary search.', keywords: ['example', 'nested', 'binary search', 'loop'], weight: 10 },
        ],
        redFlags: [
          { id: 'only_actual_seconds', description: 'Defines complexity only as exact seconds on one machine.', severity: 'medium' },
        ],
      },
    }),
  },
  {
    text: 'How would you debug an API endpoint that is returning a 500 error?',
    difficulty: 'MEDIUM' as const,
    topicCategories: ['debugging', 'backend'],
    aiEvaluationGuidance: JSON.stringify({
      questionType: 'technical_theory',
      modelAnswer: 'Start by reproducing the request and checking logs, stack traces, and recent changes. Verify inputs, request body, authentication, database calls, environment variables, and downstream services. Add targeted logging or use a debugger, isolate the failing layer, write or update a test once the issue is understood, and return a safe error response without leaking internal details.',
      rubric: {
        requiredPoints: [
          { id: 'reproduce_and_logs', description: 'Reproduces the issue and checks logs or stack traces.', keywords: ['reproduce', 'logs', 'stack trace', 'trace'], weight: 30 },
          { id: 'check_inputs_dependencies', description: 'Checks request inputs and dependencies such as database or downstream services.', keywords: ['input', 'request', 'database', 'dependency', 'service'], weight: 30 },
          { id: 'isolate_fix_verify', description: 'Isolates the failing layer, fixes it, and verifies with testing.', keywords: ['isolate', 'debugger', 'test', 'verify', 'fix'], weight: 25 },
        ],
        secondaryPoints: [
          { id: 'safe_error_handling', description: 'Mentions safe errors and avoiding leaked internals.', keywords: ['safe', 'error', 'leak', 'internal'], weight: 10 },
        ],
        excellentAnswerSignals: [
          { id: 'recent_changes_observability', description: 'Mentions recent deploys, metrics, or observability.', keywords: ['recent', 'deploy', 'metrics', 'observability'], weight: 10 },
        ],
        redFlags: [
          { id: 'guess_without_logs', description: 'Suggests changing random code without checking logs or reproducing.', severity: 'medium' },
        ],
      },
    }),
  },
  {
    text: 'Describe how you would design a simple URL shortener.',
    difficulty: 'MEDIUM' as const,
    topicCategories: ['system design', 'backend'],
    aiEvaluationGuidance: JSON.stringify({
      questionType: 'system_design',
      modelAnswer: 'A simple URL shortener needs an API to create a short code for a long URL and another API to redirect from the code to the long URL. Store mappings in a database with fields like code, long URL, created time, and optional expiry. Generate unique codes using hashing, random IDs, or an incrementing ID encoded in base62, and handle collisions. Discuss redirects, validation, analytics, caching for popular links, and basic abuse prevention.',
      rubric: {
        requiredPoints: [
          { id: 'core_apis', description: 'Defines create-short-link and redirect APIs.', keywords: ['api', 'create', 'redirect', 'short', 'long url'], weight: 25 },
          { id: 'storage_mapping', description: 'Stores short code to long URL mapping in a database.', keywords: ['database', 'store', 'mapping', 'code', 'url'], weight: 25 },
          { id: 'unique_code_generation', description: 'Explains unique code generation and collision handling.', keywords: ['unique', 'code', 'hash', 'base62', 'collision'], weight: 25 },
        ],
        secondaryPoints: [
          { id: 'cache_analytics_expiry', description: 'Mentions caching, analytics, expiry, or validation.', keywords: ['cache', 'analytics', 'expiry', 'validation'], weight: 15 },
        ],
        excellentAnswerSignals: [
          { id: 'abuse_and_scale', description: 'Mentions abuse prevention or scaling popular redirects.', keywords: ['abuse', 'rate limit', 'scale', 'popular'], weight: 10 },
        ],
        redFlags: [
          { id: 'no_persistence', description: 'Design has no persistence for URL mappings.', severity: 'high' },
        ],
      },
    }),
  },
];

export async function interviewRoutes(app: FastifyInstance) {
  app.get('/demo-session', async () => {
    const company = await prisma.company.upsert({
      where: { slug: 'demo-junior-sde' },
      update: {
        name: 'IntervieHire Demo Engineering',
        description: 'A demo engineering team hiring junior software development engineers.',
        primaryColor: '#0e7490',
      },
      create: {
        name: 'IntervieHire Demo Engineering',
        slug: 'demo-junior-sde',
        description: 'A demo engineering team hiring junior software development engineers.',
        reportEmail: 'hr@example.com',
        primaryColor: '#0e7490',
      },
    });

    let role = await prisma.jobRole.findFirst({ where: { companyId: company.id, title: 'Junior Software Development Engineer' } });
    if (!role) {
      role = await prisma.jobRole.create({
        data: {
          companyId: company.id,
          title: 'Junior Software Development Engineer',
          roleType: 'GENERAL',
          description: 'Junior engineering role focused on fundamentals, debugging, backend basics, and clear technical communication.',
          requirements: 'Data structures, algorithms, debugging, APIs, databases, and basic system design.',
          primaryCriteria: ['data structures', 'algorithms', 'debugging', 'backend fundamentals'],
          secondaryCriteria: ['communication', 'system design basics', 'testing'],
          atsScoringWeights: { primary: 0.4, secondary: 0.3, education: 0.1, experience: 0.1, communication: 0.1 },
          evaluationCriteria: { modelAnswerAlignment: 1, correctness: 1, reasoning: 1, communication: 1, confidence: 1 },
        },
      });
    }

    for (const question of juniorSdeQuestions) {
      const existing = await prisma.question.findFirst({
        where: { companyId: company.id, jobRoleId: role.id, text: question.text },
      });
      if (!existing) {
        await prisma.question.create({
          data: {
            companyId: company.id,
            jobRoleId: role.id,
            text: question.text,
            roleApplicability: ['GENERAL'],
            difficulty: question.difficulty,
            topicCategories: question.topicCategories,
            aiEvaluationGuidance: question.aiEvaluationGuidance,
          },
        });
      } else {
        await prisma.question.update({
          where: { id: existing.id },
          data: {
            difficulty: question.difficulty,
            topicCategories: question.topicCategories,
            aiEvaluationGuidance: question.aiEvaluationGuidance,
            isActive: true,
          },
        });
      }
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
  app.post('/sessions/:id/start', async (req:any) => {
    const session = await prisma.interviewSession.findUniqueOrThrow({
      where: { id: req.params.id },
      include: { jobRole: { include: { questions: { where: { isActive: true }, orderBy: { createdAt: 'asc' } } } } },
    });
    const firstQuestion = session.jobRole.questions[0]?.text ?? 'Tell me about your software engineering background.';
    const transcript = Array.isArray(session.transcript) ? session.transcript as any[] : [];
    const hasAiQuestion = transcript.some((entry) => entry?.speaker === 'ai');
    const updatedTranscript = hasAiQuestion
      ? transcript
      : [...transcript, { speaker: 'ai', text: firstQuestion, timestamp: new Date().toISOString(), questionIndex: 0 }];
    const updated = await prisma.interviewSession.update({
      where: { id: req.params.id },
      data: { status: 'IN_PROGRESS', startedAt: session.startedAt ?? new Date(), transcript: updatedTranscript as any },
    });
    return { session: updated, initialQuestion: firstQuestion };
  });
  app.get('/sessions/:id/vapi-config', async (req:any) => {
    const session = await prisma.interviewSession.findUniqueOrThrow({where:{id:req.params.id}, include:{company:true,jobRole:{include:{questions:true}}}});
    return buildVapiAssistantConfig({companyName:session.company.name, companyDescription:session.company.description || undefined, jobRole:session.jobRole.title, roleRequirements:session.jobRole.requirements, questions:session.jobRole.questions.map((q: { text: string })=>q.text), evaluationCriteria:session.jobRole.evaluationCriteria as any});
  });
  app.post('/sessions/:id/complete', async (req:any) => prisma.interviewSession.update({where:{id:req.params.id}, data:{status:'COMPLETED', completedAt:new Date()}}));
  app.post('/sessions/:id/answers', async (req:any, reply) => {
    const text = String(req.body?.text ?? '').trim();
    if (!text) return reply.code(400).send({ error: 'Answer text is required' });

    const ai = await handleCandidateTranscript(req.params.id, text, req.body?.metrics ?? {});
    return { answer: { text }, ai };
  });
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

  app.post('/sessions/:id/transcript', async (req:any, reply) => {
    const session = await prisma.interviewSession.findUnique({ where: { id: req.params.id } });
    if (!session) return reply.code(404).send({ error: 'Session not found' });

    const body = req.body ?? {};
    const transcriptId = typeof body.transcriptId === 'string' && body.transcriptId.trim()
      ? body.transcriptId.trim()
      : 'browser-speech-recognition';
    const segments = Array.isArray(body.transcript) ? body.transcript : [];
    const transcript: SpeechTranscriptSegment[] = segments
      .map((segment: any) => ({
        speaker: 'candidate' as const,
        text: typeof segment?.text === 'string' ? segment.text.trim() : '',
        timestamp: typeof segment?.timestamp === 'string' ? segment.timestamp : new Date().toISOString(),
        source: 'speech_to_text' as const,
      }))
      .filter((segment: SpeechTranscriptSegment) => segment.text.length > 0);

    if (!transcript.length) {
      return reply.code(400).send({ error: 'Transcript must contain at least one text segment' });
    }

    const fullText = typeof body.fullText === 'string' && body.fullText.trim()
      ? body.fullText.trim()
      : transcript.map((segment) => segment.text).join('\n');

    const entry = {
      type: 'speech_to_text_transcript',
      source: 'speech_to_text',
      text: fullText,
      segments: transcript,
      sessionId: session.id,
      candidateId: session.candidateId,
      transcriptId,
      finalized: body.finalized === true,
      createdAt: typeof body.createdAt === 'string' ? body.createdAt : new Date().toISOString(),
      updatedAt: new Date().toISOString(),
    };

    const current = readTranscript(session.transcript);
    const existingIndex = current.findIndex(
      (item) => item?.type === 'speech_to_text_transcript' && item?.transcriptId === transcriptId,
    );
    const updated = existingIndex >= 0
      ? current.map((item, index) => index === existingIndex ? { ...item, ...entry, createdAt: item.createdAt ?? entry.createdAt } : item)
      : [...current, entry];
    await prisma.interviewSession.update({ where: { id: req.params.id }, data: { transcript: updated as any } });

    return { stored: true, entry };
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
      const current = readTranscript(session.transcript);
      const updated = [...current, entry];
      await prisma.interviewSession.update({ where: { id: req.params.id }, data: { transcript: updated as any } });
      // kick off transcription and question-fit processing (async)
      processRecordingForSession(req.params.id, filename).catch((err) => app.log.error('Transcription error', err));
      return { url: `/uploads/${filename}`, entry };
    }

    // If session not found, return upload info — file saved but not attached to a session
    return { url: `/uploads/${filename}`, entry, note: 'session not found; recording stored but not linked' };
  });

  app.post('/sessions/:id/answer-transcription', async (req:any, reply) => {
    const part = await req.file();
    if (!part) return reply.code(400).send({ error: 'No file uploaded' });

    const uploadsDir = path.resolve(process.cwd(), 'uploads');
    if (!fs.existsSync(uploadsDir)) fs.mkdirSync(uploadsDir, { recursive: true });

    const filename = `${Date.now()}-${part.filename || 'answer.webm'}`;
    const dest = path.join(uploadsDir, filename);
    const buffer = await part.toBuffer();
    fs.writeFileSync(dest, buffer);

    try {
      const text = await transcribeUploadedFile(dest);
      return { text, filename };
    } catch (error:any) {
      app.log.error('Answer transcription error', error);
      return reply.code(502).send({ error: error?.message || 'Answer transcription failed' });
    }
  });

  // Serve uploaded files
  app.get('/uploads/:file', async (req:any, reply) => {
    const p = path.join(process.cwd(), 'uploads', req.params.file);
    if (!fs.existsSync(p)) return reply.code(404).send({ error: 'Not found' });
    return reply.send(fs.createReadStream(p));
  });
}
