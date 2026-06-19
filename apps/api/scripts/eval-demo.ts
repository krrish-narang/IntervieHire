/**
 * Standalone demo of the evaluation pipeline. Runs the REAL per-answer evaluation + aggregation on
 * in-memory sample data — no database and no API keys required (it uses the local evaluator unless
 * DEEPSEEK_API_KEY is set, in which case you'll see real LLM output).
 *
 * Run from the repo root:  npx tsx apps/api/scripts/eval-demo.ts
 */
import dotenv from 'dotenv';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { evaluateInterviewData } from '../src/services/evaluation.service.js';
import { buildCandidateFacingReport } from '@interviehire/shared';

// Load the repo-root .env so the script sees DEEPSEEK_API_KEY etc. (the server does this too).
const scriptDir = path.dirname(fileURLToPath(import.meta.url));
dotenv.config({ path: path.resolve(scriptDir, '../../../.env') });

const questions = [
  {
    id: 'q1',
    text: 'Explain the difference between an array and a linked list. When would you choose one over the other?',
    difficulty: 'EASY' as const,
    topicCategories: ['data structures'],
    aiEvaluationGuidance: JSON.stringify({
      questionType: 'technical_theory',
      modelAnswer:
        'Arrays store elements contiguously and allow O(1) index access, but inserting or deleting in the middle is O(n). Linked lists use nodes connected by pointers, so insertion or deletion at a known node is O(1), but random access is O(n) and there is pointer memory overhead. Choose arrays for fast indexing and cache locality; choose linked lists for frequent insertions and deletions.',
      rubric: {
        requiredPoints: [
          { id: 'contiguous_vs_nodes', description: 'Arrays are contiguous in memory; linked lists use nodes connected by pointers.', keywords: ['contiguous', 'pointer', 'node'], weight: 25 },
          { id: 'access_complexity', description: 'Arrays give O(1) index access; linked lists are O(n) to access an element.', keywords: ['o(1)', 'index', 'o(n)', 'access'], weight: 25 },
          { id: 'insert_delete', description: 'Linked lists are O(1) to insert/delete at a known node; arrays are O(n) in the middle.', keywords: ['insert', 'delete'], weight: 25 },
          { id: 'tradeoff_choice', description: 'When to choose each (indexing/cache locality vs frequent insert/delete).', keywords: ['cache', 'locality', 'frequent'], weight: 25 },
        ],
        secondaryPoints: [],
        excellentAnswerSignals: [],
        redFlags: [
          { id: 'wrong_access', description: 'Claims a linked list supports O(1) random access by index.', severity: 'high' },
        ],
      },
    }),
  },
  {
    id: 'q2',
    text: 'What is Big O notation and why does it matter?',
    difficulty: 'MEDIUM' as const,
    topicCategories: ['complexity'],
    aiEvaluationGuidance: JSON.stringify({
      questionType: 'technical_theory',
      modelAnswer:
        'Big O describes how an algorithm\'s runtime or space grows as the input size n grows, in the worst case. Common classes are O(1) constant, O(log n) logarithmic, O(n) linear, and O(n^2) quadratic. It lets us compare algorithms independent of hardware. Space complexity separately measures extra memory used.',
      rubric: {
        requiredPoints: [
          { id: 'growth_with_input', description: 'Big O describes how runtime/space grows as input size n grows.', keywords: ['input', 'grow', 'runtime'], weight: 30 },
          { id: 'common_classes', description: 'Knows common classes O(1), O(log n), O(n), O(n^2).', keywords: ['o(1)', 'o(n)', 'o(log n)', 'o(n^2)'], weight: 30 },
          { id: 'worst_case', description: 'Mentions worst-case / asymptotic analysis.', keywords: ['worst case', 'asymptotic'], weight: 20 },
          { id: 'space_complexity', description: 'Distinguishes time complexity from space complexity.', keywords: ['space', 'memory'], weight: 20 },
        ],
        secondaryPoints: [],
        excellentAnswerSignals: [],
        redFlags: [
          { id: 'always_faster', description: 'Claims a lower Big O is always faster for all inputs.', severity: 'medium' },
        ],
      },
    }),
  },
];

// A transcript: the AI asks each question (by index), the candidate answers. One strong answer, one weak.
const transcript = [
  { speaker: 'ai', text: questions[0].text, questionIndex: 0, timestamp: '2026-06-19T10:00:00.000Z' },
  {
    speaker: 'candidate',
    questionIndex: 0,
    timestamp: '2026-06-19T10:00:30.000Z',
    text:
      'An array stores its elements in one contiguous block of memory, so you can access any index in constant O(1) time, but inserting or deleting in the middle is O(n) because you have to shift elements. A linked list uses nodes connected by pointers, so inserting or deleting at a known node is O(1), but you lose random access, so finding an element is O(n). I would use an array when I need fast indexing and good cache locality, and a linked list when there are frequent insertions and deletions.',
  },
  { speaker: 'ai', text: questions[1].text, questionIndex: 1, timestamp: '2026-06-19T10:02:00.000Z' },
  {
    speaker: 'candidate',
    questionIndex: 1,
    timestamp: '2026-06-19T10:02:20.000Z',
    text: 'Big O is basically how fast your code runs. A lower Big O is always faster. I think O(n) is good and O(n squared) is bad.',
  },
];

function divider(title: string) {
  console.log('\n' + '='.repeat(72) + '\n' + title + '\n' + '='.repeat(72));
}

async function main() {
  const usingLlm = Boolean(process.env.DEEPSEEK_API_KEY && process.env.DEEPSEEK_API_KEY !== 'replace-me');
  console.log(`Evaluator: ${usingLlm ? 'DeepSeek LLM' : 'local fallback (set DEEPSEEK_API_KEY to use the real LLM)'}`);

  const report = await evaluateInterviewData({
    interviewId: 'demo-interview',
    candidateId: 'demo-candidate',
    jobRole: {
      title: 'Junior Software Engineer',
      roleType: 'GENERAL',
      evaluationCriteria: {},
      primaryCriteria: ['data structures', 'algorithms'],
      secondaryCriteria: ['communication'],
    },
    questions,
    transcript,
  });

  divider('COMPANY REPORT — headline');
  console.log(`Role:                     ${report.roleTitle}`);
  console.log(`Interview type (derived): ${report.interviewType}`);
  console.log(`Overall score:            ${report.overallScore}/100`);
  console.log(`Recommendation:           ${report.recommendation}  (confidence: ${report.recommendationConfidence})`);
  console.log(`Expressed confidence:     ${report.candidateConfidence.score}/100 (${report.candidateConfidence.level})`);
  console.log(`Summary:                  ${report.summary}`);

  divider('COMPANY REPORT — per-question breakdown');
  for (const answer of report.questionBreakdown) {
    console.log(`\nQ: ${answer.questionText}`);
    console.log(`   score: ${answer.overallScore}/100   confidence: ${answer.evaluationConfidence}`);
    console.log('   rubric coverage:');
    for (const point of answer.modelAnswerComparison.requiredPointCoverage) {
      const evidence = point.evidence.length ? `  ← "${point.evidence[0]}"` : '';
      console.log(`     [${point.status.padEnd(11)}] ${point.description}${evidence}`);
    }
    if (answer.redFlags.length) {
      console.log('   red flags:');
      for (const flag of answer.redFlags) {
        console.log(`     (${flag.severity}) ${flag.label}: ${flag.reason}`);
      }
    }
  }

  divider('CANDIDATE-FACING REPORT (score-free, what the interviewee may see)');
  const candidate = buildCandidateFacingReport(report);
  console.log(JSON.stringify(candidate, null, 2));

  divider('FULL COMPANY REPORT (raw JSON — the complete AI output)');
  console.log(JSON.stringify(report, null, 2));
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
