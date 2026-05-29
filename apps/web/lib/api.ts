export const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:4000';
export const WS_URL = process.env.NEXT_PUBLIC_WS_URL || 'ws://localhost:4000/ws';
export async function api<T>(path:string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_URL}${path}`, { ...init, headers:{'Content-Type':'application/json', ...(init?.headers||{})}, cache:'no-store' });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}
