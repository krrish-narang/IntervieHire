import { FastifyInstance } from 'fastify';
import { z } from 'zod';
import { prisma } from '../lib/prisma.js';
import { scoreCandidate } from '../services/ats-screening.service.js';
import { generateQuestions } from '../services/question-generation.service.js';

export async function companyRoutes(app: FastifyInstance) {
  app.get('/dashboard/:companyId', async (req:any) => {
    const companyId = req.params.companyId;
    const [company, candidates, roles, sessions] = await Promise.all([
      prisma.company.findUnique({where:{id:companyId}}),
      prisma.candidate.findMany({where:{companyId}, orderBy:{createdAt:'desc'}}),
      prisma.jobRole.findMany({where:{companyId}, include:{questions:true}}),
      prisma.interviewSession.findMany({where:{companyId}, include:{candidate:true,jobRole:true,proctoringLogs:true}, orderBy:{createdAt:'desc'}})
    ]);
    return {company, candidates, roles, sessions};
  });

  app.post('/candidates', async (req:any) => {
    const body = z.object({companyId:z.string(), fullName:z.string(), email:z.string().email(), phone:z.string().optional(), parsedResume:z.record(z.any()).default({}), resumeText:z.string().optional(), jobRoleId:z.string()}).parse(req.body);
    const role = await prisma.jobRole.findUniqueOrThrow({where:{id:body.jobRoleId}});
    const ats = scoreCandidate({...body.parsedResume, resumeText: body.resumeText}, role);
    const candidate = await prisma.candidate.upsert({
      where:{companyId_email:{companyId:body.companyId,email:body.email}},
      create:{companyId:body.companyId, fullName:body.fullName,email:body.email,phone:body.phone,parsedResume:body.parsedResume,resumeText:body.resumeText,atsScore:ats.score,atsBreakdown:ats.breakdown as any},
      update:{fullName:body.fullName,phone:body.phone,parsedResume:body.parsedResume,resumeText:body.resumeText,atsScore:ats.score,atsBreakdown:ats.breakdown as any}
    });
    const session = await prisma.interviewSession.create({data:{companyId:body.companyId,candidateId:candidate.id,jobRoleId:body.jobRoleId,status:'SCHEDULED'}});
    return {candidate, ats, session};
  });

  app.post('/questions/generate', async (req:any) => {
    const body = z.object({companyId:z.string(), jobRoleId:z.string(), jobDescription:z.string(), roleType:z.string(), companyName:z.string(), jobTitle:z.string().optional()}).parse(req.body);
    const questions = await generateQuestions(body);
    const created = await prisma.$transaction(questions.slice(0,10).map((q:any) => prisma.question.create({data:{companyId:body.companyId,jobRoleId:body.jobRoleId,text:q.text,difficulty:q.difficulty || 'MEDIUM',topicCategories:q.topicCategories || [],roleApplicability:[body.roleType as any],aiEvaluationGuidance:q.aiEvaluationGuidance || 'Assess specificity, structure, and role relevance.'}})));
    return {questions: created};
  });

  app.put('/questions/:id', async (req:any) => prisma.question.update({where:{id:req.params.id}, data:req.body}));
}
