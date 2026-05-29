'use client';

import { useEffect, useMemo, useRef, useState } from 'react';
import { Bar, BarChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import { FileText, Plus, ShieldAlert, Sparkles, Upload, Users } from 'lucide-react';
import { api } from '@/lib/api';

type Dashboard = {
  company: any;
  candidates: any[];
  roles: any[];
  sessions: any[];
};

type CandidateForm = {
  fullName: string;
  email: string;
  phone: string;
  resumeText: string;
};

const fallback: Dashboard = {
  company: { id: 'demo', name: 'Demo Consulting Group' },
  candidates: [{ id: 'demo-cand-1', fullName: 'Aarav Sharma', email: 'aarav@example.com', atsScore: 82 }],
  roles: [
    {
      id: 'demo-role-1',
      title: 'Associate Consultant',
      questions: [{ id: 'demo-q-1', text: 'Tell me about an ambiguous problem.', difficulty: 'MEDIUM' }],
    },
  ],
  sessions: [],
};

const defaultCandidateForm: CandidateForm = {
  fullName: '',
  email: '',
  phone: '',
  resumeText: '',
};

export default function Dashboard() {
  const [data, setData] = useState<Dashboard>(fallback);
  const [jd, setJd] = useState(
    'Associate Consultant role requiring structured problem solving, client communication, analytics and executive presence.',
  );
  const [jobTitle, setJobTitle] = useState('Associate Consultant');
  const [jdFileName, setJdFileName] = useState('');
  const [questions, setQuestions] = useState<any[]>([]);
  const [showCandidateForm, setShowCandidateForm] = useState(false);
  const [candidateForm, setCandidateForm] = useState<CandidateForm>(defaultCandidateForm);
  const [savingCandidate, setSavingCandidate] = useState(false);
  const overviewRef = useRef<HTMLDivElement | null>(null);
  const candidatesRef = useRef<HTMLDivElement | null>(null);
  const questionsRef = useRef<HTMLDivElement | null>(null);
  const reportsRef = useRef<HTMLDivElement | null>(null);
  const activeRole = data.roles?.[0];
  // Consider a workspace "ready" when we have a company and an active role.
  // Allow the demo workspace to be interactive in-dev so the UI never gets stuck on "Workspace loading...".
  const workspaceReady = Boolean(data.company?.id && activeRole?.id);

  useEffect(() => {
    const id = localStorage.getItem('companyId');
    if (id) api<Dashboard>(`/api/company/dashboard/${id}`).then(setData).catch(() => {});
  }, []);

  useEffect(() => {
    if (activeRole?.title) {
      setJobTitle(activeRole.title);
    }
  }, [activeRole?.title]);

  function scrollToSection(section: 'overview' | 'candidates' | 'questions' | 'reports') {
    const target =
      section === 'overview'
        ? overviewRef.current
        : section === 'candidates'
          ? candidatesRef.current
          : section === 'questions'
            ? questionsRef.current
            : reportsRef.current;

    target?.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }

  async function handleJdUpload(file: File) {
    const text = await file.text();
    setJd(text);
    setJdFileName(file.name);
    if (!jobTitle.trim() && file.name) {
      setJobTitle(file.name.replace(/\.[^.]+$/, '').replace(/[-_]/g, ' '));
    }
  }

  async function gen() {
    if (!workspaceReady || !activeRole) return;
    const companyId = data.company.id;
    // If we're using the demo workspace, just surface the built-in role questions locally
    if (companyId === 'demo') {
      setQuestions(activeRole.questions || []);
      return;
    }
    const result = await api<any>('/api/company/questions/generate', {
      method: 'POST',
      body: JSON.stringify({
        companyId,
        jobRoleId: activeRole.id,
        jobTitle,
        jobDescription: jd,
        roleType: activeRole.roleType || 'CONSULTING',
        companyName: data.company.name,
      }),
    });
    setQuestions(result.questions);
  }

  async function addCandidate() {
    if (!workspaceReady || !activeRole) return;
    const companyId = data.company?.id || localStorage.getItem('companyId');
    if (!companyId) return;

    setSavingCandidate(true);
    try {
      // If demo workspace, simulate candidate creation locally to avoid API errors in dev
      if (companyId === 'demo') {
        const newCandidate = {
          id: `demo-cand-${Date.now()}`,
          fullName: candidateForm.fullName,
          email: candidateForm.email,
          phone: candidateForm.phone,
          atsScore: 0,
        } as any;
        const newSession = {
          id: `demo-sess-${Date.now()}`,
          candidate: newCandidate,
          jobRole: activeRole,
          status: 'SCHEDULED',
          proctoringLogs: [],
        } as any;
        setData((current) => ({
          ...current,
          candidates: [newCandidate, ...current.candidates.filter((candidate) => candidate.email !== newCandidate.email)],
          sessions: [newSession, ...current.sessions],
        }));
      } else {
        const result = await api<any>('/api/company/candidates', {
          method: 'POST',
          body: JSON.stringify({
            companyId,
            jobRoleId: activeRole.id,
            fullName: candidateForm.fullName,
            email: candidateForm.email,
            phone: candidateForm.phone || undefined,
            resumeText: candidateForm.resumeText || undefined,
            parsedResume: {
              fullName: candidateForm.fullName,
              email: candidateForm.email,
              phone: candidateForm.phone,
              resumeText: candidateForm.resumeText,
            },
          }),
        });

        setData((current) => ({
          ...current,
          candidates: [result.candidate, ...current.candidates.filter((candidate) => candidate.email !== result.candidate.email)],
          sessions: [result.session, ...current.sessions],
        }));
      }

      setCandidateForm(defaultCandidateForm);
      setShowCandidateForm(false);
      document.getElementById('candidates')?.scrollIntoView({ behavior: 'smooth', block: 'start' });
    } finally {
      setSavingCandidate(false);
    }
  }

  const chart = useMemo(
    () => (data.candidates || []).map((candidate) => ({ name: candidate.fullName?.split(' ')[0], score: candidate.atsScore || 0 })),
    [data.candidates],
  );

  const reportItems = data.sessions?.slice(0, 3) || [];

  return (
    <main className="min-h-screen bg-slate-50">
      <aside className="fixed inset-y-0 left-0 hidden w-72 border-r bg-ink p-6 text-white lg:block">
        <div className="text-2xl font-black">IntervieHire</div>
        <p className="mt-2 text-sm text-cyan-100">Enterprise hiring command center</p>
        <nav className="mt-10 space-y-2 text-sm">
          <a href="#overview" className="block rounded-2xl bg-white/10 px-4 py-3">
            Dashboard
          </a>
          <button type="button" onClick={() => scrollToSection('candidates')} className="block w-full rounded-2xl px-4 py-3 text-left text-cyan-100 transition hover:bg-white/5">
            Candidates
          </button>
          <button type="button" onClick={() => scrollToSection('questions')} className="block w-full rounded-2xl px-4 py-3 text-left text-cyan-100 transition hover:bg-white/5">
            Questions
          </button>
          <button type="button" onClick={() => scrollToSection('reports')} className="block w-full rounded-2xl px-4 py-3 text-left text-cyan-100 transition hover:bg-white/5">
            Reports
          </button>
        </nav>
      </aside>

      <section className="lg:pl-72">
        <header className="flex items-center justify-between border-b bg-white px-8 py-5">
          <div>
            <p className="text-sm text-slate-500">Company workspace</p>
            <h1 className="text-2xl font-black">{data.company?.name || 'Dashboard'}</h1>
          </div>
          <button
            type="button"
            onClick={() => {
              setShowCandidateForm((current) => !current);
              window.setTimeout(() => {
                document.getElementById('candidate-form')?.scrollIntoView({ behavior: 'smooth', block: 'start' });
              }, 0);
            }}
            className="rounded-full bg-ink px-5 py-3 text-sm font-semibold text-white"
          >
            <Plus className="mr-2 inline" size={16} />
            {showCandidateForm ? 'Close form' : 'Add candidate'}
          </button>
        </header>

        <div className="grid gap-6 p-8 xl:grid-cols-3">
          <div id="overview" ref={overviewRef} className="scroll-mt-8 grid gap-4 md:grid-cols-3 xl:col-span-3">
            {[
              [Users, 'Candidates', data.candidates?.length || 0],
              [FileText, 'Questions', data.roles?.[0]?.questions?.length || 0],
              [ShieldAlert, 'Flagged Events', data.sessions?.reduce((total, session) => total + (session.proctoringLogs?.length || 0), 0) || 0],
            ].map(([Icon, title, value]: any) => (
              <div key={title} className="rounded-3xl border bg-white p-6 shadow-sm">
                <Icon className="mb-4 text-brand" />
                <p className="text-sm text-slate-500">{title}</p>
                <b className="text-3xl">{value}</b>
              </div>
            ))}
          </div>

          <div id="candidates" ref={candidatesRef} className="scroll-mt-8 rounded-3xl border bg-white p-6 shadow-sm xl:col-span-2">
            <div className="mb-5 flex items-center justify-between">
              <h2 className="text-xl font-bold">Candidate pipeline</h2>
              <span className="text-sm text-slate-500">ATS-ranked</span>
            </div>

            <div className="overflow-hidden rounded-2xl border">
              <table className="w-full text-left text-sm">
                <thead className="bg-slate-50 text-slate-500">
                  <tr>
                    <th className="p-4">Candidate</th>
                    <th>Email</th>
                    <th>ATS</th>
                    <th>Status</th>
                  </tr>
                </thead>
                <tbody>
                  {data.candidates?.map((candidate) => (
                    <tr className="border-t" key={candidate.email}>
                      <td className="p-4 font-semibold">{candidate.fullName}</td>
                      <td>{candidate.email}</td>
                      <td>
                        <span className="rounded-full bg-emerald-50 px-3 py-1 font-semibold text-emerald-700">
                          {candidate.atsScore}
                        </span>
                      </td>
                      <td>Scheduled</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            <div className="mt-6 h-64">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={chart}>
                  <XAxis dataKey="name" />
                  <YAxis />
                  <Tooltip />
                  <Bar dataKey="score" radius={[8, 8, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>

            {showCandidateForm ? (
              <div id="candidate-form" className="mt-6 rounded-3xl border border-dashed border-slate-300 bg-slate-50 p-5">
                <div className="flex items-center justify-between gap-4">
                  <div>
                    <h3 className="text-lg font-bold">New candidate</h3>
                    <p className="text-sm text-slate-500">Create a candidate and schedule an interview against the first role in this workspace.</p>
                  </div>
                  <button
                    type="button"
                    onClick={() => setShowCandidateForm(false)}
                    className="rounded-full border px-4 py-2 text-sm font-semibold text-slate-600"
                  >
                    Dismiss
                  </button>
                </div>

                <div className="mt-4 grid gap-4 md:grid-cols-2">
                  <input
                    value={candidateForm.fullName}
                    onChange={(event) => setCandidateForm((current) => ({ ...current, fullName: event.target.value }))}
                    placeholder="Full name"
                    className="rounded-2xl border bg-white px-4 py-3 text-sm outline-none focus:ring-2 focus:ring-brand"
                  />
                  <input
                    value={candidateForm.email}
                    onChange={(event) => setCandidateForm((current) => ({ ...current, email: event.target.value }))}
                    placeholder="Email"
                    type="email"
                    className="rounded-2xl border bg-white px-4 py-3 text-sm outline-none focus:ring-2 focus:ring-brand"
                  />
                  <input
                    value={candidateForm.phone}
                    onChange={(event) => setCandidateForm((current) => ({ ...current, phone: event.target.value }))}
                    placeholder="Phone"
                    className="rounded-2xl border bg-white px-4 py-3 text-sm outline-none focus:ring-2 focus:ring-brand"
                  />
                  <div className="rounded-2xl border bg-white px-4 py-3 text-sm text-slate-500">
                    Role: {data.roles[0]?.title || 'No role available'}
                  </div>
                  <textarea
                    value={candidateForm.resumeText}
                    onChange={(event) => setCandidateForm((current) => ({ ...current, resumeText: event.target.value }))}
                    placeholder="Paste resume text or summary"
                    className="md:col-span-2 h-32 rounded-2xl border bg-white px-4 py-3 text-sm outline-none focus:ring-2 focus:ring-brand"
                  />
                </div>

                <button
                  type="button"
                  onClick={addCandidate}
                  disabled={savingCandidate || !workspaceReady || !candidateForm.fullName || !candidateForm.email}
                  className="mt-4 inline-flex items-center rounded-2xl bg-brand px-5 py-3 font-semibold text-white disabled:cursor-not-allowed disabled:opacity-60"
                >
                  <Plus className="mr-2 inline" size={16} />
                  {savingCandidate ? 'Saving...' : workspaceReady ? 'Save candidate' : 'Workspace loading...'}
                </button>
              </div>
            ) : null}
          </div>

          <div id="questions" ref={questionsRef} className="scroll-mt-8 rounded-3xl border bg-white p-6 shadow-sm">
            <h2 className="text-xl font-bold">AI question builder</h2>
            <p className="mt-2 text-sm leading-6 text-slate-600">
              Upload a JD or paste one below, then generate questions for the role title.
            </p>
            <div className="mt-4 space-y-3">
              <label className="block text-sm font-semibold text-slate-700">Job title</label>
              <input
                value={jobTitle}
                onChange={(event) => setJobTitle(event.target.value)}
                placeholder="Associate Consultant"
                className="w-full rounded-2xl border p-4 text-sm outline-none focus:ring-2 focus:ring-brand"
              />
              <label className="block text-sm font-semibold text-slate-700">Upload JD text file</label>
              <label className="inline-flex cursor-pointer items-center gap-2 rounded-2xl border border-dashed border-slate-300 bg-slate-50 px-4 py-3 text-sm font-semibold text-slate-600 hover:bg-slate-100">
                <Upload size={16} />
                {jdFileName || 'Upload JD (.txt, .md)'}
                <input
                  type="file"
                  accept=".txt,.md,.csv,.json"
                  className="hidden"
                  onChange={(event) => {
                    const file = event.target.files?.[0];
                    if (file) void handleJdUpload(file);
                  }}
                />
              </label>
            </div>
            <textarea
              value={jd}
              onChange={(event) => setJd(event.target.value)}
              className="mt-4 h-36 w-full rounded-2xl border p-4 text-sm outline-none focus:ring-2 focus:ring-brand"
              placeholder="Paste the job description here or upload a JD text file..."
            />
            <button
              type="button"
              onClick={gen}
              disabled={!workspaceReady}
              className="mt-4 w-full rounded-2xl bg-brand px-5 py-3 font-semibold text-white"
            >
              <Sparkles className="mr-2 inline" size={16} />
              {workspaceReady ? 'Suggest questions for title' : 'Workspace loading...'}
            </button>
            <p className="mt-3 text-xs text-slate-500">Suggestions are generated from the job title plus the uploaded JD text.</p>
            <div className="mt-5 space-y-3">
              {(questions.length ? questions : data.roles?.[0]?.questions || []).map((question: any, index: number) => (
                <div key={index} className="rounded-2xl bg-slate-50 p-4 text-sm">
                  <b>{question.difficulty}</b>
                  <p className="mt-1 text-slate-700">{question.text}</p>
                </div>
              ))}
            </div>
          </div>

          <div id="reports" ref={reportsRef} className="scroll-mt-8 rounded-3xl border bg-white p-6 shadow-sm xl:col-span-3">
            <div className="flex items-center justify-between gap-4">
              <div>
                <h2 className="text-xl font-bold">Reports and recent sessions</h2>
                <p className="mt-2 text-sm text-slate-600">Use this section for evaluation outputs, PDF reports, and recent interview activity.</p>
              </div>
              <span className="rounded-full bg-slate-100 px-4 py-2 text-sm font-semibold text-slate-600">
                {reportItems.length} recent session{reportItems.length === 1 ? '' : 's'}
              </span>
            </div>

            <div className="mt-5 grid gap-4 md:grid-cols-3">
              {reportItems.length ? (
                reportItems.map((session: any) => (
                  <div key={session.id} className="rounded-2xl border bg-slate-50 p-4 text-sm">
                    <p className="font-semibold">{session.candidate?.fullName || 'Candidate'}</p>
                    <p className="mt-1 text-slate-600">{session.jobRole?.title || 'Role'} · {session.status}</p>
                    <p className="mt-1 text-slate-500">Proctoring events: {session.proctoringLogs?.length || 0}</p>
                  </div>
                ))
              ) : (
                <div className="rounded-2xl border bg-slate-50 p-4 text-sm text-slate-600 md:col-span-3">
                  No report data yet. Complete an interview session to populate this area.
                </div>
              )}
            </div>
          </div>
        </div>
      </section>
    </main>
  );
}
