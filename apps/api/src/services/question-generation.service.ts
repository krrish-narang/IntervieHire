import { callOpenRouter } from '../lib/openrouter.js';
export async function generateQuestions(input: {roleType: string; jobDescription: string; companyName: string; jobTitle?: string}) {
  const competencyMap: Record<string,string> = {
    PRODUCT_MANAGEMENT: 'user empathy, prioritization, product lifecycle, metrics, cross-functional collaboration',
    BUSINESS_ANALYST: 'analytical rigor, requirements gathering, stakeholder management, documentation, insight generation',
    FOUNDERS_OFFICE: 'entrepreneurial mindset, ownership, ambiguity handling, leadership, strategic thinking',
    CONSULTING: 'problem decomposition, client skills, structured thinking, executive communication, industry awareness'
  };
  const prompt = `Generate 8 interview questions as JSON array for ${input.jobTitle || input.roleType}. Focus on ${competencyMap[input.roleType] || 'role competencies'}. Each item must have text, difficulty, topicCategories, aiEvaluationGuidance. Job description: ${input.jobDescription}`;
  const raw = await callOpenRouter([{role:'system', content:'You are an expert interview designer. Return only valid JSON.'},{role:'user', content: prompt}], {json:true});
  try { return JSON.parse(raw); } catch { return [{text:'Walk me through a recent project where you created measurable impact.', difficulty:'MEDIUM', topicCategories:['impact'], aiEvaluationGuidance:'Look for clarity, ownership, metrics, and reflection.'}]; }
}
