export const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:4000';
export const WS_URL = process.env.NEXT_PUBLIC_WS_URL || 'ws://localhost:4000/ws';

function isHtmlResponse(contentType: string, body: string) {
  return contentType.includes('text/html') || /^\s*(?:<!doctype html|<html)/i.test(body);
}

function responseErrorMessage(res: Response, body: string, data: unknown) {
  if (data && typeof data === 'object') {
    const error = data as { error?: unknown; message?: unknown };
    if (typeof error.message === 'string' && error.message.trim()) return error.message;
    if (typeof error.error === 'string' && error.error.trim()) return error.error;
  }

  if (isHtmlResponse(res.headers.get('content-type') || '', body)) {
    return `The API returned an HTML page instead of data (${res.status}). Check that the API is running at ${API_URL}.`;
  }

  const plainText = body.trim();
  return plainText && plainText.length <= 300
    ? plainText
    : `API request failed (${res.status} ${res.statusText || 'Unknown error'}).`;
}

export async function parseApiResponse<T>(res: Response): Promise<T> {
  const body = await res.text();
  const contentType = res.headers.get('content-type') || '';
  let data: unknown;

  if (body) {
    if (isHtmlResponse(contentType, body)) {
      throw new Error(responseErrorMessage(res, body, undefined));
    }

    try {
      data = JSON.parse(body);
    } catch {
      throw new Error(
        res.ok
          ? 'The API returned an invalid response. Please try again.'
          : responseErrorMessage(res, body, undefined),
      );
    }
  }

  if (!res.ok) {
    throw new Error(responseErrorMessage(res, body, data));
  }

  return data as T;
}

export async function api<T>(path:string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_URL}${path}`, { ...init, headers:{'Content-Type':'application/json', ...(init?.headers||{})}, cache:'no-store' });
  return parseApiResponse<T>(res);
}
