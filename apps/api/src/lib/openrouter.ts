export async function callOpenRouter(messages: Array<{role: 'system' | 'user' | 'assistant'; content: string}>, options?: {json?: boolean}) {
  const apiKey = process.env.OPENROUTER_API_KEY;
  if (!apiKey || apiKey === 'replace-me') {
    return options?.json
      ? JSON.stringify({
          answerDepth:{score:4,reasoning:'Mock evaluation: strong examples and layered answers.'},
          confidence:{score:4,reasoning:'Mock evaluation: steady and concise communication.'},
          communication:{score:4,reasoning:'Mock evaluation: clear structure and suitable pacing.'},
          domainKnowledge:{score:3,reasoning:'Mock evaluation: good fundamentals with some gaps.'},
          problemSolving:{score:4,reasoning:'Mock evaluation: structured approach with trade-offs.'},
          overallScore:3.8,recommendation:'HIRE',strengths:['Structured answers','Good communication'],risks:['Validate domain depth in final round'],summary:'Mock report generated because OPENROUTER_API_KEY is not configured.'
        })
      : 'Thanks. I would like to go one layer deeper. Can you share a concrete example, the trade-offs you considered, and the outcome you achieved?';
  }
  const res = await fetch('https://openrouter.ai/api/v1/chat/completions', {
    method: 'POST',
    headers: {'Authorization': `Bearer ${apiKey}`, 'Content-Type':'application/json', 'HTTP-Referer':'https://interviehire.local', 'X-Title':'IntervieHire'},
    body: JSON.stringify({model: process.env.OPENROUTER_MODEL || 'openai/gpt-4o-mini', messages, response_format: options?.json ? {type:'json_object'} : undefined})
  });
  if (!res.ok) throw new Error(`OpenRouter failed: ${res.status} ${await res.text()}`);
  const data = await res.json() as any;
  return data.choices?.[0]?.message?.content ?? '';
}
