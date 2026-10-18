// Examples users write themselves, beside the built-in catalogue (architecture stage 73).
//
// They live on the agent server under the author's account (/agent/examples), so they follow the
// author to another browser. Private until shared: the owner can make a link that other
// signed-in users open, and an admin can publish one to everyone. Running one is the same as
// running a built-in example: its files are fetched and uploaded into a NEW conversation through
// the ordinary upload route, so they land in the runner's own file store.
import { type AgentConfig, CREDENTIALS, absoluteUrl, authHeaders } from './agentClient';
import { authErrorFrom, isAuthStatus, withTokenRetry } from './auth';
import type { DemoCase, DemoStep } from './demos';

export type Visibility = 'private' | 'link' | 'published';

export interface UserExampleFile { name: string; bytes: number; url: string }

/** A user example has the built-in case's shape, so the panel shows and runs both the same way. */
export interface UserExample extends Omit<DemoCase, 'files'> {
  source: 'user';
  files: UserExampleFile[];
  mine: boolean;
  visibility: Visibility;
  published: boolean;
  /** Only on the owner's own examples. */
  shareToken?: string | null;
  /** Set when the owner's edit took a published example back to unpublished. */
  unpublishedByEdit?: string;
  copiedFrom?: string;
  createdAt?: string;
  updatedAt?: string;
}

export interface ExampleLimits { maxExamples: number; maxSteps: number; maxFiles: number; maxBytes: number }

export interface ExampleListing {
  mine: UserExample[];
  shared: UserExample[];
  published: UserExample[];
  canCreate: boolean;
  canPublish: boolean;
  limits: ExampleLimits;
}

export function isUserExample(c: DemoCase | UserExample): c is UserExample {
  return (c as UserExample).source === 'user';
}

// --- the editor's draft -------------------------------------------------------------------

export interface ExampleDraft {
  title: string;
  description: string;
  category: string;
  /** As typed: comma-separated. */
  tags: string;
  steps: DemoStep[];
  /** Names of the example's current files to keep (an edit). */
  keepFiles: string[];
}

export function emptyDraft(category: string): ExampleDraft {
  return { title: '', description: '', category, tags: '', steps: [{ label: '', prompt: '' }], keepFiles: [] };
}

export function draftFrom(ex: UserExample): ExampleDraft {
  return {
    title: ex.title, description: ex.blurb, category: ex.category, tags: ex.tags.join(', '),
    steps: ex.steps.map((s) => ({ ...s })), keepFiles: ex.files.map((f) => f.name),
  };
}

export function parseTags(raw: string): string[] {
  const out: string[] = [];
  for (const t of raw.split(',').map((s) => s.trim()).filter(Boolean)) if (!out.includes(t)) out.push(t);
  return out;
}

/** Move step `k` by `by` (−1 up, +1 down). Out of range leaves the list as it is. */
export function moveStep(steps: DemoStep[], k: number, by: number): DemoStep[] {
  const j = k + by;
  if (k < 0 || k >= steps.length || j < 0 || j >= steps.length) return steps;
  const next = steps.slice();
  [next[k], next[j]] = [next[j], next[k]];
  return next;
}

/** What is wrong with the draft before it is sent, or null. The server checks again and has the
 *  last word; this only saves a round trip for the obvious cases. Blank steps are dropped, not
 *  refused: "+ Add step" leaves one behind easily. */
export function draftProblem(d: ExampleDraft, newFiles: File[], limits?: ExampleLimits): string | null {
  if (!d.title.trim()) return 'Give the example a title.';
  if (!d.category.trim()) return 'Pick a category.';
  const steps = d.steps.filter((s) => s.prompt.trim() || s.label.trim());
  if (!steps.length) return 'Add at least one step with a question.';
  const k = steps.findIndex((s) => !s.prompt.trim());
  if (k >= 0) return `Step ${k + 1} needs a question.`;
  if (limits) {
    if (steps.length > limits.maxSteps) return `At most ${limits.maxSteps} steps.`;
    const names = new Set([...d.keepFiles, ...newFiles.map((f) => f.name)]);
    if (names.size > limits.maxFiles) return `At most ${limits.maxFiles} files.`;
    const bytes = newFiles.reduce((s, f) => s + f.size, 0);
    if (bytes > limits.maxBytes) return `The data may total at most ${Math.round(limits.maxBytes / 1048576)} MB.`;
  }
  return null;
}

/** The body the server takes: the definition as JSON in `example`, the new files under `files`. */
export function exampleForm(d: ExampleDraft, newFiles: File[], editing: boolean): FormData {
  const fd = new FormData();
  const body: Record<string, unknown> = {
    title: d.title.trim(), description: d.description.trim(), category: d.category.trim(),
    tags: parseTags(d.tags),
    steps: d.steps.filter((s) => s.prompt.trim() || s.label.trim())
      .map((s) => ({ label: s.label.trim(), prompt: s.prompt.trim() })),
  };
  if (editing) body.keepFiles = d.keepFiles;
  fd.append('example', JSON.stringify(body));
  newFiles.forEach((f) => fd.append('files', f, f.name));
  return fd;
}

// --- share links ----------------------------------------------------------------------------

export const SHARE_PARAM = 'example';

/** The page's own address with `?example=<token>`, so the link opens this UI on the example. */
export function shareLink(token: string, href: string = location.href): string {
  const u = new URL(href);
  u.search = '';
  u.hash = '';
  u.searchParams.set(SHARE_PARAM, token);
  return u.toString();
}

export function shareTokenFrom(href: string = location.href): string | null {
  try {
    const t = new URL(href).searchParams.get(SHARE_PARAM);
    return t && /^[A-Za-z0-9_-]{8,64}$/.test(t) ? t : null;
  } catch {
    return null;
  }
}

/** The address without the share token, so a reload does not open the link again. */
export function withoutShareToken(href: string = location.href): string {
  const u = new URL(href);
  u.searchParams.delete(SHARE_PARAM);
  return u.toString();
}

// --- the API --------------------------------------------------------------------------------

async function call(cfg: AgentConfig, path: string, init: RequestInit = {}, json = false): Promise<any> {
  return withTokenRetry(async () => {
    const r = await fetch(absoluteUrl(path, cfg), {
      ...init, credentials: CREDENTIALS, headers: authHeaders(cfg, json),
    });
    // A refusal with an example-specific reason is the server saying "nobody to own this with",
    // not an expired session; only the session ones go to the sign-in flow.
    if (isAuthStatus(r.status)) {
      const copy = r.clone();
      const body = await copy.json().catch(() => null);
      if (body?.reason !== 'examples_need_sign_in') throw await authErrorFrom(r);
    }
    const body = await r.json().catch(() => null);
    if (!r.ok) throw new Error(body?.error || `HTTP ${r.status}`);
    return body;
  });
}

const enc = encodeURIComponent;

/** Null when the server could not be asked, which must not read as "you have none". */
export async function listExamples(cfg: AgentConfig): Promise<ExampleListing | null> {
  try { return (await call(cfg, '/agent/examples')) as ExampleListing; } catch { return null; }
}

export function createExample(cfg: AgentConfig, d: ExampleDraft, files: File[]): Promise<UserExample> {
  return call(cfg, '/agent/examples', { method: 'POST', body: exampleForm(d, files, false) });
}

export function updateExample(cfg: AgentConfig, id: string, d: ExampleDraft, files: File[]): Promise<UserExample> {
  return call(cfg, `/agent/examples/${enc(id)}`, { method: 'PUT', body: exampleForm(d, files, true) });
}

export function deleteExample(cfg: AgentConfig, id: string): Promise<unknown> {
  return call(cfg, `/agent/examples/${enc(id)}`, { method: 'DELETE' });
}

export function shareExample(cfg: AgentConfig, id: string, rotate = false): Promise<{ shareToken: string; example: UserExample }> {
  return call(cfg, `/agent/examples/${enc(id)}/share`,
              { method: 'POST', body: JSON.stringify({ rotate }) }, true);
}

export function unshareExample(cfg: AgentConfig, id: string): Promise<{ example: UserExample }> {
  return call(cfg, `/agent/examples/${enc(id)}/share`, { method: 'DELETE' });
}

export function openSharedExample(cfg: AgentConfig, token: string): Promise<UserExample> {
  return call(cfg, '/agent/examples/open', { method: 'POST', body: JSON.stringify({ token }) }, true);
}

export function forgetSharedExample(cfg: AgentConfig, id: string): Promise<unknown> {
  return call(cfg, `/agent/examples/${enc(id)}/shared`, { method: 'DELETE' });
}

export function copyExample(cfg: AgentConfig, id: string): Promise<UserExample> {
  return call(cfg, `/agent/examples/${enc(id)}/copy`, { method: 'POST' });
}

export function setPublished(cfg: AgentConfig, id: string, on: boolean): Promise<UserExample> {
  return call(cfg, `/agent/examples/${enc(id)}/publish`, { method: on ? 'POST' : 'DELETE' });
}

/** The example's files as File objects, fetched from the agent with the page's credentials, so
 *  they go through the same upload path as a built-in example's or a hand-picked file. */
export async function loadUserExampleFiles(c: UserExample, cfg: AgentConfig): Promise<File[]> {
  return Promise.all(c.files.map(async (f) => {
    const r = await fetch(absoluteUrl(f.url, cfg), { headers: authHeaders(cfg, false), credentials: CREDENTIALS });
    if (!r.ok) throw new Error(`${f.name}: HTTP ${r.status}`);
    const blob = await r.blob();
    return new File([blob], f.name, { type: blob.type || 'application/octet-stream' });
  }));
}
