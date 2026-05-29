import { FastifyInstance } from 'fastify';
import { z } from 'zod';

import { callGemini } from '../services/gemini.service.js';

export async function assistantRoutes(app: FastifyInstance) {
  app.post('/chat', async (req: any) => {
    const body = z.object({
      messages: z.array(z.object({ role: z.enum(['user', 'assistant']), content: z.string().min(1) })).min(1),
      page: z.string().optional(),
      topic: z.string().optional(),
    }).parse(req.body);

    const systemInstruction = [
      'You are the IntervieHire assistant.',
      'Help users understand and use the app, answer hiring and interview workflow questions, and explain dashboard or candidate actions clearly.',
      'Keep answers concise, practical, and friendly.',
      body.page ? `Current page: ${body.page}.` : '',
      body.topic ? `User topic: ${body.topic}.` : '',
    ].filter(Boolean).join(' ');

    const answer = await callGemini(body.messages, { systemInstruction });
    return { answer };
  });
}