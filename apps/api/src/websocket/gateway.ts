import type { FastifyInstance } from 'fastify';
import { prisma } from '../lib/prisma.js';
import { handleCandidateTranscript } from '../services/interview-conversation.service.js';
import type { ClientToServerTranscript, ProctoringPayload, ServerToUESpeak } from '@interviehire/shared';

type Socket = any;
const candidates = new Map<string, Socket>();
const ueClients = new Map<string, Socket>();

function send(socket: Socket | undefined, payload: unknown) {
  if (socket && socket.readyState === 1) socket.send(JSON.stringify(payload));
}

export async function registerWebsocket(app: FastifyInstance) {
  app.get('/ws', { websocket: true }, (connection) => {
    const socket = connection;
    socket.on('message', async (raw: Buffer) => {
      try {
        const msg = JSON.parse(raw.toString());
        if (msg.type === 'register') {
          (msg.role === 'ue5' ? ueClients : candidates).set(msg.sessionId, socket);
          send(socket, {type:'registered', role: msg.role, sessionId: msg.sessionId});
          return;
        }
        if (msg.type === 'candidate_transcript') {
          const payload = msg as ClientToServerTranscript;
          const ai = await handleCandidateTranscript(payload.sessionId, payload.text, {latencyMs: payload.latencyMs, wpm: payload.wpm});
          const speak: ServerToUESpeak = {type:'avatar_speak', sessionId: payload.sessionId, ...ai};
          send(ueClients.get(payload.sessionId), speak);
          const { type: _t, ...speakNoType } = speak as any;
          send(candidates.get(payload.sessionId), {type:'ai_response', ...speakNoType});
          return;
        }
        if (msg.type === 'avatar_status') {
          send(candidates.get(msg.sessionId), msg);
          return;
        }
        if (msg.type === 'proctoring_event') {
          const event = msg as ProctoringPayload;
          await prisma.proctoringLog.create({data:{sessionId:event.sessionId,eventType:event.eventType,severity:event.severity as any,metadata:event.metadata as any,occurredAt:new Date(event.timestamp)}});
          send(candidates.get(event.sessionId), {type:'proctoring_ack', eventType:event.eventType});
        }
      } catch (error: any) {
        send(socket, {type:'error', message:error.message || 'WebSocket error'});
      }
    });
    socket.on('close', () => {
      for (const [k,v] of candidates.entries()) if (v === socket) candidates.delete(k);
      for (const [k,v] of ueClients.entries()) if (v === socket) ueClients.delete(k);
    });
  });
}
