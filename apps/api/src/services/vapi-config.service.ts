export function buildVapiAssistantConfig(input: {companyName:string; companyDescription?:string; jobRole:string; roleRequirements:string; questions:string[]; evaluationCriteria?:Record<string, unknown>; voiceId?:string}) {
  const systemPrompt = `You are an experienced professional interviewer for ${input.companyName}, conducting interviews for the ${input.jobRole} position. Assess domain knowledge, communication skills, problem-solving ability, and cultural fit. Ask the provided questions naturally and ask relevant follow-ups that probe examples, trade-offs, metrics, and lessons learned. Company Description: ${input.companyDescription || 'N/A'} Role Requirements: ${input.roleRequirements} Questions to Cover: ${input.questions.join(' | ')} Evaluation Focus Areas: ${JSON.stringify(input.evaluationCriteria || {})}`;
  return {
    name: `Interview Assistant - ${input.companyName}`,
    model: { provider: 'openai', model: 'gpt-4o-mini', messages: [{ role: 'system', content: systemPrompt }] },
    voice: { provider: 'elevenlabs', voiceId: input.voiceId || 'default' },
    transcriber: { provider: 'deepgram', model: 'nova-2' }
  };
}
