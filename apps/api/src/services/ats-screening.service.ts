import type { JobRole } from '@prisma/client';

type CandidateProfile = Record<string, any>;
const defaultWeights = {
  CONSULTING: { primary: 0.4, secondary: 0.3, education: 0.1, experience: 0.1, communication: 0.1 },
  PRODUCT_MANAGEMENT: { primary: 0.35, secondary: 0.35, education: 0.1, experience: 0.1, communication: 0.1 },
  BUSINESS_ANALYST: { primary: 0.45, secondary: 0.25, education: 0.1, experience: 0.1, communication: 0.1 },
  FOUNDERS_OFFICE: { primary: 0.4, secondary: 0.3, education: 0.1, experience: 0.1, communication: 0.1 },
  GENERAL: { primary: 0.35, secondary: 0.25, education: 0.15, experience: 0.15, communication: 0.1 }
};
function normalizedMatch(text: string, terms: string[]) {
  const lower = text.toLowerCase();
  if (!terms.length) return 0.5;
  const hits = terms.filter(t => lower.includes(t.toLowerCase())).length;
  return Math.min(1, hits / Math.max(1, terms.length));
}
function numericScore(value: any, max = 10) { return Math.max(0, Math.min(1, Number(value || 0) / max)); }
export function scoreCandidate(candidate: CandidateProfile, role: JobRole) {
  const resumeText = JSON.stringify(candidate).toLowerCase();
  const weights = Object.assign({}, defaultWeights[role.roleType], role.atsScoringWeights as object);
  const primary = normalizedMatch(resumeText, role.primaryCriteria);
  const secondary = normalizedMatch(resumeText, role.secondaryCriteria);
  const education = normalizedMatch(resumeText, ['mba','b.tech','bachelor','master','economics','engineering','business']);
  const experience = numericScore(candidate.yearsOfExperience ?? candidate.experienceYears, 8);
  const communication = normalizedMatch(resumeText, ['presentation','stakeholder','client','communication','written','verbal']);
  const breakdown = { primary, secondary, education, experience, communication, weights };
  const score = Object.entries({primary, secondary, education, experience, communication}).reduce((sum, [k, v]) => sum + Number(v) * Number((weights as any)[k] ?? 0), 0) * 100;
  return { score: Math.round(score * 10) / 10, breakdown };
}
