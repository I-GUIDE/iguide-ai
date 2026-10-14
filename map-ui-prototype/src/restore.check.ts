/** A runnable check that reopening a conversation re-renders its answers. `npm run check:restore`.
 *
 * The UI saved each answer's RENDERED html and reopened it as is, so a renderer fix never reached
 * an old chat. After S9.12 (deployed 2026-10-14) a new answer numbered its list 1 to 8, while a
 * conversation saved before it, reopened from History, still read "1." on every item.
 *
 * Imported as a namespace so the check runs, and fails, on a tree without rerenderMessages: there
 * restore is `setMessages(rec.messages)`, i.e. the identity below.
 */
import * as store from './sessionStore';
import type { ChatMessage } from './components/ChatPanel';
import { renderMarkdown } from './markdown';

const rerender: (m: ChatMessage[], r: (u: string) => string) => ChatMessage[] =
  (store as any).rerenderMessages ?? ((m: ChatMessage[]) => m);

let bad = 0;
const eq = (label: string, got: unknown, want: unknown) => {
  const ok = JSON.stringify(got) === JSON.stringify(want);
  if (!ok) bad++;
  console.log(`${ok ? 'ok  ' : 'FAIL'} ${label}: got ${JSON.stringify(got)}`);
};
const resolveUrl = (u: string) => `https://agent.example${u}`;
const count = (h: string | undefined, tag: string) => ((h || '').match(new RegExp(`<${tag}[ >]`, 'g')) || []).length;

// A loose list, as the prod answer on 2026-10-13 had it, and the html the pre-S9.12 renderer made
// of it: one <ol> per item, each numbered "1.".
const LOOSE = '1. a\n\n2. b\n\n3. c';
const OLD_HTML = '<ol><li>a</li></ol><ol><li>b</li></ol><ol><li>c</li></ol>';

const trace = [{ text: 'tool: x', kind: 'tool' }] as any;
const saved: ChatMessage[] = [
  { role: 'user', text: 'list three things' },
  // Saved before this change: the markdown survives only inside the terminal result.
  { role: 'agent', html: OLD_HTML, response: { answer: LOOSE }, trace, streaming: false },
  // Saved after it: the markdown is stored beside the html.
  { role: 'agent', html: OLD_HTML, markdown: LOOSE, streaming: false },
  // No markdown anywhere: the saved html is all there is.
  { role: 'agent', html: OLD_HTML, streaming: false },
  // An error turn: no html, the error as text. Its response is kept as it was.
  { role: 'agent', html: '', text: '⚠ boom', response: { answer: LOOSE }, streaming: false },
  // A turn saved mid-stream.
  { role: 'agent', streaming: true, trace: [], response: { answer: LOOSE } },
  // A link to a file the agent wrote resolves through the same resolver as a live answer.
  { role: 'agent', html: '<p>old</p>', markdown: '[map](/agent/files/f1/download)', streaming: false },
];
const out = rerender(saved, resolveUrl);

eq('a pre-change answer (markdown in response.answer) is one list again',
   count(out[1].html, 'ol'), 1);
eq('... with all three items', out[1].html, '<ol><li>a</li><li>b</li><li>c</li></ol>');
eq('... and keeps its other fields', [out[1].trace, out[1].response], [trace, { answer: LOOSE }]);
eq('an answer storing `markdown` is re-rendered from it', out[2].html, renderMarkdown(LOOSE, resolveUrl));
eq('an answer with no markdown keeps its saved html', out[3], saved[3]);
eq('an error turn is untouched', out[4], saved[4]);
eq('a streaming turn is untouched', out[5], saved[5]);
eq('the user message is untouched', out[0], saved[0]);
eq('links resolve through resolveUrl',
   out[6].html, '<p><a href="https://agent.example/agent/files/f1/download" target="_blank" rel="noopener noreferrer">map</a></p>');
eq('nothing is added or dropped', out.length, saved.length);

if (bad) { console.error(`\n${bad} check(s) failed`); process.exit(1); }
console.log('\nall restore checks passed');
