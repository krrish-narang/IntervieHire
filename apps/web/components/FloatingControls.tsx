'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { useMemo, useState } from 'react';
import { Home, Send, Sparkles, X } from 'lucide-react';

import { api } from '@/lib/api';

type AssistantMessage = {
  role: 'assistant' | 'user';
  content: string;
};

export function FloatingControls() {
  const pathname = usePathname();
  const [open, setOpen] = useState(false);
  const [topic, setTopic] = useState('');
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [messages, setMessages] = useState<AssistantMessage[]>([
    { role: 'assistant', content: 'Ask me about the product, setup, dashboard actions, or interview workflow.' },
  ]);

  const pageName = useMemo(() => {
    if (pathname === '/dashboard') return 'dashboard';
    if (pathname === '/interview') return 'candidate interview';
    return 'home';
  }, [pathname]);

  async function sendMessage() {
    const prompt = input.trim();
    if (!prompt || loading) return;

    const nextMessages: AssistantMessage[] = [...messages, { role: 'user', content: prompt }];
    setMessages(nextMessages);
    setInput('');
    setLoading(true);
    try {
      const result = await api<{ answer: string }>('/api/assistant/chat', {
        method: 'POST',
        body: JSON.stringify({ messages: nextMessages, page: pageName, topic: topic.trim() || undefined }),
      });
      setMessages((current) => [...current, { role: 'assistant', content: result.answer }]);
    } catch (error) {
      setMessages((current) => [
        ...current,
        { role: 'assistant', content: 'I could not reach the Gemini assistant right now. Check GEMINI_API_KEY and try again.' },
      ]);
    } finally {
      setLoading(false);
    }
  }

  return (
    <>
      <Link
        href="/"
        className="fixed bottom-4 right-4 z-50 inline-flex items-center gap-2 rounded-full border border-slate-200 bg-white px-4 py-2 text-sm font-semibold text-black shadow-lg shadow-slate-200/70 transition hover:-translate-y-0.5 hover:shadow-xl hover:shadow-slate-300/70"
        aria-label="Go to home page"
      >
        <Home size={16} />
        Home
      </Link>

      <button
        type="button"
        onClick={() => setOpen((current) => !current)}
        className="fixed bottom-4 left-4 z-50 inline-flex items-center gap-2 rounded-full bg-ink px-4 py-3 text-sm font-semibold text-white shadow-lg shadow-slate-300/70 transition hover:-translate-y-0.5 hover:shadow-xl"
        aria-label="Open AI assistant"
      >
        <Sparkles size={16} />
        AI Assistant
      </button>

      {open ? (
        <div className="fixed bottom-20 left-4 z-50 w-[min(92vw,22rem)] overflow-hidden rounded-[1.75rem] border border-slate-200 bg-white shadow-2xl shadow-slate-300/60">
          <div className="flex items-center justify-between border-b px-4 py-3">
            <div>
              <p className="text-xs font-semibold uppercase tracking-[0.2em] text-slate-400">Gemini assistant</p>
              <h2 className="text-sm font-bold text-ink">IntervieHire help desk</h2>
            </div>
            <button type="button" onClick={() => setOpen(false)} className="rounded-full p-2 text-slate-500 hover:bg-slate-100">
              <X size={16} />
            </button>
          </div>

          <div className="max-h-72 space-y-3 overflow-auto p-4 text-sm">
            {messages.map((message, index) => (
              <div key={`${message.role}-${index}`} className={`rounded-2xl px-3 py-2 leading-6 ${message.role === 'assistant' ? 'bg-slate-50 text-slate-700' : 'bg-cyan-50 text-slate-900'}`}>
                {message.content}
              </div>
            ))}
          </div>

          <div className="space-y-3 border-t p-4">
            <input
              value={topic}
              onChange={(event) => setTopic(event.target.value)}
              placeholder="Optional topic, like dashboard, interviews, or setup"
              className="w-full rounded-2xl border px-4 py-2.5 text-sm outline-none focus:ring-2 focus:ring-brand"
            />
            <div className="flex gap-2">
              <input
                value={input}
                onChange={(event) => setInput(event.target.value)}
                onKeyDown={(event) => event.key === 'Enter' && sendMessage()}
                placeholder="Ask a question..."
                className="min-w-0 flex-1 rounded-2xl border px-4 py-2.5 text-sm outline-none focus:ring-2 focus:ring-brand"
              />
              <button
                type="button"
                onClick={sendMessage}
                disabled={loading || !input.trim()}
                className="inline-flex items-center rounded-2xl bg-brand px-4 py-2.5 text-sm font-semibold text-white disabled:cursor-not-allowed disabled:opacity-60"
              >
                <Send size={16} />
              </button>
            </div>
          </div>
        </div>
      ) : null}
    </>
  );
}