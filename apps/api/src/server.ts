import 'dotenv/config';
import Fastify from 'fastify';
import cors from '@fastify/cors';
import websocket from '@fastify/websocket';
import rateLimit from '@fastify/rate-limit';
import multipart from '@fastify/multipart';
import { assistantRoutes } from './routes/assistant.routes.js';
import { companyRoutes } from './routes/company.routes.js';
import { interviewRoutes } from './routes/interview.routes.js';
import { registerWebsocket } from './websocket/gateway.js';

const app = Fastify({ logger: true });
await app.register(cors, { origin: true, credentials: true });
await app.register(rateLimit, { max: 200, timeWindow: '1 minute' });
await app.register(multipart);
await app.register(websocket);
app.get('/health', async () => ({ ok: true, service: 'interviehire-api' }));
await app.register(companyRoutes, { prefix: '/api/company' });
await app.register(interviewRoutes, { prefix: '/api/interview' });
await app.register(assistantRoutes, { prefix: '/api/assistant' });
await registerWebsocket(app);

const port = Number(process.env.PORT || 4000);
app.listen({ port, host: '0.0.0.0' }).catch(err => { app.log.error(err); process.exit(1); });
