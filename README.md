# IntervieHire — AI-Powered Interview Platform

A deployable full-stack MVP for AI-assisted hiring: ATS screening, structured AI interviews, avatar-ready WebSocket bridge, proctoring event pipeline, LLM evaluation, PDF reports, and a professional recruiter/candidate UI.

## What is included

- **Frontend:** Next.js, React, TypeScript, Tailwind, Framer Motion-ready UI, Recharts dashboard.
- **Backend:** Fastify, Prisma, PostgreSQL, WebSockets, OpenRouter integration, PDFKit reports, Nodemailer email delivery.
- **AI interview loop:** Candidate transcript → backend context builder → OpenRouter → frontend + UE5 avatar payload.
- **Avatar bridge:** WebSocket payloads for UE5/MetaHuman/Convai style lip-sync: `avatar_speak` and `avatar_status`.
- **ATS engine:** Weighted role scoring for Consulting, PM, Business Analyst, Founders' Office, and General roles.
- **Question builder:** LLM-generated questions with role-specific competencies and editable metadata.
- **Proctoring:** Webcam/mic permission flow, event pipeline, severity logs, backend persistence. MediaPipe is included as a dependency and the hook is structured for detector model activation.
- **Reports:** Post-interview evaluation JSON and professional PDF generation; optional email delivery.

## Local setup

Prerequisite: start Docker Desktop first and wait until the Linux engine is running. On Windows, `docker compose` needs access to the `dockerDesktopLinuxEngine` pipe before it can pull `postgres:16-alpine` and `redis:7-alpine`.

```bash
cp .env.example .env
cd infra && docker compose up -d postgres redis
cd ..
npm install
npm run db:generate
npm run db:migrate
npm run seed
npm run dev
```

If you see `failed to connect to the docker API at npipe:////./pipe/dockerDesktopLinuxEngine`, Docker Desktop is not running yet or WSL2/Linux containers are not ready. Start Docker Desktop, wait for it to finish initializing, then rerun the compose command.

Open:

- Web app: http://localhost:3000
- API health: http://localhost:4000/health
- Candidate room: http://localhost:3000/interview
- Company dashboard: http://localhost:3000/dashboard

After seeding, copy the printed `companyId` into browser localStorage:

```js
localStorage.setItem('companyId', 'PASTE_SEEDED_COMPANY_ID')
```

## Docker deployment

```bash
cp .env.example .env
cd infra
docker compose up --build
```

## Important environment variables

- `DATABASE_URL` — PostgreSQL connection string.
- `OPENROUTER_API_KEY` — enables live LLM question generation, interview follow-ups, and evaluation.
- `OPENROUTER_MODEL` — defaults to `openai/gpt-4o-mini`.
- `GEMINI_API_KEY` — enables the floating AI assistant in the app shell.
- `GEMINI_MODEL` — defaults to `gemini-1.5-flash`.
- `SMTP_*` and `REPORT_FROM` — enables email delivery of PDF reports.
- `NEXT_PUBLIC_API_URL`, `NEXT_PUBLIC_WS_URL` — frontend API and WebSocket endpoints.

## Core API endpoints

- `GET /health`
- `GET /api/company/dashboard/:companyId`
- `POST /api/company/candidates`
- `POST /api/company/questions/generate`
- `PUT /api/company/questions/:id`
- `GET /api/interview/sessions/:id`
- `POST /api/interview/sessions/:id/start`
- `GET /api/interview/sessions/:id/vapi-config`
- `POST /api/interview/sessions/:id/complete`
- `POST /api/interview/sessions/:id/evaluate`
- `POST /api/interview/sessions/:id/report`
- `POST /api/interview/sessions/:id/email-report`
- `WS /ws`

## WebSocket message examples

Candidate registration:

```json
{"type":"register","role":"candidate","sessionId":"SESSION_ID"}
```

UE5 registration:

```json
{"type":"register","role":"ue5","sessionId":"SESSION_ID"}
```

Candidate transcript:

```json
{"type":"candidate_transcript","sessionId":"SESSION_ID","text":"I led a pricing project...","timestamp":1710000000,"wpm":132,"latencyMs":2200}
```

Server to UE5 avatar:

```json
{"type":"avatar_speak","sessionId":"SESSION_ID","text":"Can you walk me through the trade-offs?","interviewPhase":"follow_up","emotionState":"curious"}
```

Proctoring event:

```json
{"type":"proctoring_event","sessionId":"SESSION_ID","eventType":"FACE_NOT_DETECTED","severity":"HIGH","metadata":{"durationMs":12000},"timestamp":1710000000}
```

## Production hardening checklist

- Add authentication and RBAC before external pilots.
- Replace demo localStorage company selection with authenticated tenancy.
- Add signed upload storage for resumes and interview recordings.
- Enable MediaPipe model assets for face count, gaze estimation, and object detection.
- Add audit logs for every candidate-data access.
- Configure data-retention and deletion workflows for GDPR/DPDP compliance.
- Add managed Postgres, Redis, object storage, CI/CD, Sentry, and structured logs.

