/** A runnable check of deleting a conversation from History. `npm run check:conversations`.
 *
 * Same reasoning as check:auth: there is no test runner here, and the failure is quiet. In token
 * mode the × removed only the browser's copy, by the wrong key (the row id is the server's
 * memoryId; IndexedDB is keyed by the client's session id), so it removed nothing. The server
 * kept the conversation and the next refresh listed it again (reported 2026-10-13, S9.13).
 */
import { deleteConversation, type AgentConfig } from './agentClient';
import { setRefreshUrl } from './auth';
import { localCopiesOf } from './sessionStore';

let bad = 0;
const eq = (label: string, got: unknown, want: unknown) => {
  const ok = JSON.stringify(got) === JSON.stringify(want);
  if (!ok) bad++;
  console.log(`${ok ? 'ok  ' : 'FAIL'} ${label}: got ${JSON.stringify(got)}`);
};

type Call = { url: string; method: string; credentials?: string; headers: Record<string, string> };
let calls: Call[] = [];
let answer: (url: string) => Promise<Response> = async () => reply(200, { deleted: true });
function reply(status: number, body: unknown): Response {
  return { ok: status >= 200 && status < 300, status, json: async () => body,
           clone() { return this; } } as unknown as Response;
}
(globalThis as unknown as { fetch: unknown }).fetch = async (url: string, init?: RequestInit) => {
  calls.push({ url: String(url), method: init?.method || 'GET', credentials: init?.credentials,
               headers: (init?.headers || {}) as Record<string, string> });
  return answer(String(url));
};

const cfg = { endpoint: 'https://agent.example/agent/chat/stream',
              uploadEndpoint: 'https://agent.example/agent/files/upload', apiKey: '' } as AgentConfig;
const MID = '0f8c2a51-7d3e-4b8e-9a51-2c1d0e6f4a7b';

// --- the request ----------------------------------------------------------------------
calls = [];
eq('a 200 is done', await deleteConversation(cfg, MID), { ok: true });
eq('it is one DELETE', calls.map((c) => c.method), ['DELETE']);
eq('  ...of that conversation', calls[0].url, `https://agent.example/agent/conversations/${MID}`);
eq('  ...carrying the sign-in cookie', calls[0].credentials, 'include');

calls = [];
await deleteConversation(cfg, 'a/b?c');
eq('the id is encoded, never a path', calls[0].url, 'https://agent.example/agent/conversations/a%2Fb%3Fc');

// --- outcomes ---------------------------------------------------------------------------
answer = async () => reply(404, { error: 'No conversation found for that id.' });
eq('a 404 is done: it is already gone', await deleteConversation(cfg, MID), { ok: true });

answer = async () => reply(500, { error: 'Internal server error: boom' });
const failed = await deleteConversation(cfg, MID);
eq('a 500 is a failure', failed.ok, false);
eq('  ...that says nothing was removed',
   !failed.ok && /HTTP 500/.test(failed.message) && /Nothing was removed/.test(failed.message), true);

answer = async () => reply(403, { error: 'Please sign in to use the agent.', reason: 'not_signed_in' });
const refused = await deleteConversation(cfg, MID);
eq('a 403 is a failure', refused.ok, false);
eq('  ...with the server\'s reason', !refused.ok && refused.message,
   'Please sign in to use the agent. Nothing was deleted.');

answer = async () => { throw new TypeError('Failed to fetch'); };
const offline = await deleteConversation(cfg, MID);
eq('unreachable is a failure, not a success', offline.ok, false);

// An expired token is refreshed once and the delete sent again, as every other call does.
setRefreshUrl('https://platform.example/refresh');
let first = true;
calls = [];
answer = async (url) => {
  if (url.endsWith('/refresh')) return reply(200, {});
  if (first) { first = false; return reply(401, { reason: 'token_expired' }); }
  return reply(200, { deleted: true });
};
eq('an expired token is refreshed and retried', await deleteConversation(cfg, MID), { ok: true });
eq('  ...DELETE, refresh, DELETE', calls.map((c) => c.method), ['DELETE', 'POST', 'DELETE']);
setRefreshUrl('');

// --- the local copies -------------------------------------------------------------------
const records = [
  { id: 'sess-a', memoryId: MID },          // the usual token-mode cache entry
  { id: 'sess-b', memoryId: 'another' },
  { id: MID, memoryId: null },              // keyed by the memoryId itself
  { id: 'sess-c' },                          // never reached the server
];
eq('copies are found by memoryId, not only by key', localCopiesOf(records, MID), ['sess-a', MID]);
eq('an empty id matches nothing', localCopiesOf(records, ''), []);

if (bad) { console.error(`${bad} check(s) failed`); process.exit(1); }
console.log('all conversation-delete checks passed');
