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