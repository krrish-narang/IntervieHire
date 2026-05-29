import { prisma } from '../src/lib/prisma';

(async function main() {
  try {
    const rows = await prisma.proctoringLog.findMany({ take: 20, orderBy: { occurredAt: 'desc' } });
    console.log(JSON.stringify(rows, null, 2));
  } catch (err) {
    console.error('Error querying ProctoringLog:', err);
    process.exitCode = 1;
  } finally {
    await prisma.$disconnect();
  }
})();
