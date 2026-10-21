/** The one HTTP client for the whole app. */

const API_BASE = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

/** Milliseconds before a request is abandoned. A hung fetch never resolves. */
const TIMEOUT_MS = 20_000;

export class ApiError extends Error {
  constructor(
    readonly status: number,
    readonly url: string,
    message: string,
  ) {
    super(message);
    this.name = 'ApiError';
  }
}

function describe(status: number, body: string, url: string): string {
  const detail = extractDetail(body);
  if (status === 404) {
    return detail || `Not found: ${url}`;
  }
  return detail || `Request failed (${status})`;
}

/** FastAPI puts the useful message in `detail`; a validation error puts a list there. */
function extractDetail(body: string): string {
  if (!body) return '';
  try {
    const parsed = JSON.parse(body);
    if (typeof parsed?.detail === 'string') return parsed.detail;
    if (Array.isArray(parsed?.detail)) {
      return parsed.detail
        .map((d: { loc?: unknown[]; msg?: string }) =>
          `${(d.loc ?? []).slice(1).join('.')}: ${d.msg ?? ''}`.trim(),
        )
        .join('; ');
    }
  } catch {
    /* not JSON — fall through to the raw body */
  }
  return body.slice(0, 300);
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const url = `${API_BASE}${path}`;
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), TIMEOUT_MS);

  const headers = new Headers(init?.headers);
  if (init?.body && !headers.has('Content-Type')) {
    headers.set('Content-Type', 'application/json');
  }

  try {
    // **`no-store`, or polling silently serves stale bytes.** These responses
    // carry no `Cache-Control`, `ETag` or `Last-Modified`, and a response with
    // none of those is not uncacheable -- the browser falls back to HEURISTIC
    // freshness and may reuse it without revalidating. The symptom is a page
    // that polls on schedule, shows no error, and only updates when the user
    // hits reload: every request after the first was answered from disk.
    // Belt and braces with the `no-store` the API now sends, because this end
    // is the one that survives a proxy in between.
    const response = await fetch(url, {
      cache: 'no-store',
      ...init,
      headers,
      signal: controller.signal,
    });
    if (!response.ok) {
      throw new ApiError(response.status, url, describe(response.status, await response.text(), url));
    }
    if (response.status === 204) return undefined as T;
    return (await response.json()) as T;
  } catch (error) {
    if (error instanceof ApiError) throw error;
    if (error instanceof DOMException && error.name === 'AbortError') {
      throw new ApiError(0, url, `Timed out after ${TIMEOUT_MS / 1000}s`);
    }
    // A network failure is usually a stopped backend or a disallowed origin;
    // fetch cannot tell them apart.
    throw new ApiError(
      0,
      url,
      `Cannot reach the API at ${API_BASE}. Is the backend running, and is this origin in CORS_ALLOW_ORIGINS?`,
    );
  } finally {
    clearTimeout(timer);
  }
}

// A path to a router's root route must carry the trailing slash. Otherwise
// FastAPI answers with a 307 built from the proxied path, which loses nginx's
// `/api/` prefix and lands on the SPA's index.html.
export function get<T>(path: string): Promise<T> {
  return request<T>(path);
}

export { API_BASE };
