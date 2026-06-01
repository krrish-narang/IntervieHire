import { prisma } from '../src/lib/prisma';

async function main(){
  // create company
  const company = await prisma.company.create({ data: { name: 'Test Company', slug: `test-company-${Date.now()}`, description: 'Temporary test company' } });
  const role = await prisma.jobRole.create({ data: { companyId: company.id, title: 'Test Role', description: 'Test', requirements: 'Test', primaryCriteria: [], secondaryCriteria: [], atsScoringWeights: {} as any } });
  const candidate = await prisma.candidate.create({ data: { companyId: company.id, fullName: 'Test Candidate', email: `test+${Date.now()}@example.com` } });
  const session = await prisma.interviewSession.create({ data: { companyId: company.id, candidateId: candidate.id, jobRoleId: role.id, status: 'SCHEDULED' } });
  console.log('Created test session:', session.id);
  await prisma.$disconnect();
}

main().catch(err=>{ console.error(err); process.exitCode=1; });
