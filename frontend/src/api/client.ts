/**
 * One HTTP client for the whole app.
 *
 * Every api module used to declare its own `API_BASE` and its own
 * `fetchWithError`, five near-identical copies that had already drifted: three
 * threw `API error 404` with the body discarded, two included it. So a failing
 * request told you different things depending on which screen you were on.
 *
 * It also centralises the API token. `POST /research/launch` is authenticated and
 * fails closed, so a mutating call without the header gets a 401 that needs to
 * read as "set VITE_API_TOKEN", not as a generic failure.
 */

const API_BASE = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';
const API_TOKEN = import.meta.env.VITE_API_TOKEN || '';

/** Milliseconds before a request is abandoned. A hung fetch never resolves,
 *  which leaves a polling screen stuck on its last value with no indication. */
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

  /** True when the problem is a missing or wrong API token rather than the request. */
  get isAuthProblem(): boolean {
    return this.status === 401 || this.status === 503;
  }
}

function describe(status: number, body: string, url: string): string {
  const detail = extractDetail(body);
  if (status === 401) {
    return detail || 'Not authorised. Set VITE_API_TOKEN to the API_TOKEN the backend was started with.';
  }
  if (status === 503) {
    return detail || 'This action is disabled until API_TOKEN is set on the backend.';
  }
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
  if (API_TOKEN) headers.set('X-API-Token', API_TOKEN);

  try {
    const response = await fetch(url, { ...init, headers, signal: controller.signal });
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
    // A network-level failure is usually the backend being down or an origin the
    // API does not allow, and both look identical to fetch.
    throw new ApiError(
      0,
      url,
      `Cannot reach the API at ${API_BASE}. Is the backend running, and is this origin in CORS_ALLOW_ORIGINS?`,
    );
  } finally {
    clearTimeout(timer);
  }
}

export function get<T>(path: string): Promise<T> {
  return request<T>(path);
}

export function post<T>(path: string, body?: unknown): Promise<T> {
  return request<T>(path, {
    method: 'POST',
    body: body === undefined ? undefined : JSON.stringify(body),
  });
}

export function hasApiToken(): boolean {
  return Boolean(API_TOKEN);
}

export { API_BASE };
