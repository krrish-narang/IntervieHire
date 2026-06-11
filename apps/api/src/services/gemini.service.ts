export async function callGemini(messages: Array<{ role: 'user' | 'assistant'; content: string }>, options?: { systemInstruction?: string }) {
  const apiKey = process.env.GEMINI_API_KEY;
  if (!apiKey || apiKey === 'replace-me') {
    return 'Gemini is not configured yet. Set GEMINI_API_KEY in your .env file to enable the assistant.';
  }

  const model = process.env.GEMINI_MODEL || 'gemini-1.5-flash';
  const res = await fetch(`https://generativelanguage.googleapis.com/v1beta/models/${model}:generateContent?key=${apiKey}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      systemInstruction: options?.systemInstruction ? { parts: [{ text: options.systemInstruction }] } : undefined,
      contents: messages.map((message) => ({
        role: message.role === 'assistant' ? 'model' : 'user',
        parts: [{ text: message.content }],
      })),
      generationConfig: {
        temperature: 0.4,
        topP: 0.95,
        maxOutputTokens: 512,
      },
    }),
  });

  if (!res.ok) {
    throw new Error(`Gemini failed: ${res.status} ${await res.text()}`);
  }

  const data = await res.json() as any;
  return data.candidates?.[0]?.content?.parts?.map((part: { text?: string }) => part.text || '').join('').trim() || 'I could not generate a response.';
}

export async function callGeminiJson<T>(params: {
  prompt: string;
  systemInstruction: string;
  model?: string;
  maxOutputTokens?: number;
  temperature?: number;
}): Promise<T> {
  const apiKey = process.env.GEMINI_API_KEY;

  if (!apiKey || apiKey === 'replace-me') {
    throw new Error('Gemini is not configured. Set GEMINI_API_KEY to enable semantic evaluation.');
  }

  const model = params.model || process.env.GEMINI_EVALUATION_MODEL || 'gemini-2.5-flash-lite';
  const res = await fetch(`https://generativelanguage.googleapis.com/v1beta/models/${model}:generateContent?key=${apiKey}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      systemInstruction: { parts: [{ text: params.systemInstruction }] },
      contents: [
        {
          role: 'user',
          parts: [{ text: params.prompt }],
        },
      ],
      generationConfig: {
        temperature: params.temperature ?? 0.15,
        topP: 0.9,
        maxOutputTokens: params.maxOutputTokens ?? 8192,
        responseMimeType: 'application/json',
      },
    }),
  });

  if (!res.ok) {
    throw new Error(`Gemini failed: ${res.status} ${await res.text()}`);
  }

  const data = await res.json() as any;
  const text = data.candidates?.[0]?.content?.parts?.map((part: { text?: string }) => part.text || '').join('').trim();

  if (!text) {
    throw new Error('Gemini returned an empty response.');
  }

  try {
    return JSON.parse(text) as T;
  } catch {
    const jsonMatch = text.match(/\{[\s\S]*\}/);

    if (!jsonMatch) {
      throw new Error(`Gemini returned non-JSON output: ${text.slice(0, 300)}`);
    }

    return JSON.parse(jsonMatch[0]) as T;
  }
}
