/** A runnable check that an upload names its conversation. `npm run check:upload`.
 *
 * Same reasoning as check:auth: there is no test runner here, and this fails in a way no
 * screenshot shows. An upload sent without its thread id still succeeds, still attaches and still
 * draws. The only symptom is on the server, which stores the file with no conversation, so the
 * agent's listing of this conversation's files never includes it. Outside token mode, any
 * conversation can then find it by name. Measured on 2026-10-02: every upload was stored that way.
 */
import { uploadFiles, type AgentConfig } from './agentClient';

let bad = 0;
const eq = (label: string, got: unknown, want: unknown) => {
  const ok = JSON.stringify(got) === JSON.stringify(want);
  if (!ok) bad++;
  console.log(`${ok ? 'ok  ' : 'FAIL'} ${label}: got ${JSON.stringify(got)}`);
};

/** The body of the last upload, as the browser would have sent it. */
let sent = new FormData();
(globalThis as unknown as { fetch: unknown }).fetch = async (_url: unknown, init?: RequestInit) => {
  sent = init?.body as FormData;
  return { ok: true, status: 200, json: async () => ({ files: [], count: 0 }) } as unknown as Response;
};

const cfg = { endpoint: 'https://agent.example/agent/chat/stream',
              uploadEndpoint: 'https://agent.example/agent/files/upload', apiKey: '' } as AgentConfig;
const file = new File(['payload'], 'probe.txt');

await uploadFiles([file], cfg, 'sess-a');
eq('the thread id travels with the upload', sent.getAll('thread_id'), ['sess-a']);
eq('  ...beside the file', (sent.get('files') as File | null)?.name, 'probe.txt');

// Absent and empty both send no field, and the server stores the file the way it always did.
// An empty one must not become a conversation called "" that every such upload would share.
await uploadFiles([file], cfg);
eq('no thread id sends no field', sent.has('thread_id'), false);
await uploadFiles([file], cfg, '');
eq('an empty thread id sends no field', sent.has('thread_id'), false);

console.log(bad ? `\n${bad} FAILED` : '\nall passed');
process.exit(bad ? 1 : 0);
