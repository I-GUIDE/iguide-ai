// Stage 93: the answer is shown while it is written. `answer_draft` events carry the text so far;
// the `answer` event that follows carries the checked answer, which is what the turn keeps.
// Run: npm run check:draft
import { streamChat, type AgentConfig } from './agentClient';

let bad = 0;
const eq = (name: string, got: unknown, want: unknown) => {
  const ok = JSON.stringify(got) === JSON.stringify(want);
  if (!ok) bad += 1;
  console.log(`${ok ? 'ok  ' : 'FAIL'} ${name}${ok ? '' : `\n     got  ${JSON.stringify(got)}\n     want ${JSON.stringify(want)}`}`);
};

const sse = (event: string, data: unknown) => `event: ${event}\ndata: ${JSON.stringify(data)}\n\n`;
const body = [
  sse('answer_draft', { text: 'The area' }),
  sse('answer_draft', { text: 'The area is 2,584.6 km²' }),
  sse('answer', { final_answer: 'The area is **2,584.6 km²**.\n\nSources: US Census TIGERweb.' }),
].join('');

(globalThis as unknown as { fetch: unknown }).fetch = async () =>
  new Response(body, { status: 200, headers: { 'content-type': 'text/event-stream' } });

const drafts: string[] = [];
const cfg = { endpoint: '/agent/chat/stream', uploadEndpoint: '', apiKey: '' } as AgentConfig;
const res = await streamChat('area of Champaign County', { threadId: 't' }, cfg, {
  onAnswerDraft: (t) => drafts.push(t),
});
eq('each draft reaches the page, in order', drafts, ['The area', 'The area is 2,584.6 km²']);
eq('the turn keeps the checked answer, not the last draft', res.answer,
   'The area is **2,584.6 km²**.\n\nSources: US Census TIGERweb.');

(globalThis as unknown as { fetch: unknown }).fetch = async () =>
  new Response(sse('answer_draft', { text: '' }) + sse('answer', { final_answer: 'ok' }), { status: 200 });
const empty: string[] = [];
await streamChat('q', { threadId: 't' }, cfg, { onAnswerDraft: (t) => empty.push(t) });
eq('an empty draft is not shown', empty, []);

console.log(bad ? `\n${bad} FAILED` : '\nall passed');
process.exit(bad ? 1 : 0);
