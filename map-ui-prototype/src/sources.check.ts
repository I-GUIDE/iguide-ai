/** A runnable check of where a source links to. `npm run check:sources`.
 *
 * Same reasoning as check:auth: there is no test runner here, and the failure is quiet. A source
 * the client cannot link is still listed, as plain text. On a prod turn on 2026-10-13 six of
 * eight sources were plain text: graph hits typed 'resource', and one element that keyword search
 * had returned typed 'publication' was listed from its untyped graph copy instead (Stage 57).
 */
import { evidenceDocs, sourceHref } from './answerFormat';

let bad = 0;
const eq = (label: string, got: unknown, want: unknown) => {
  const ok = JSON.stringify(got) === JSON.stringify(want);
  if (!ok) bad++;
  console.log(`${ok ? 'ok  ' : 'FAIL'} ${label}: got ${JSON.stringify(got)}`);
};

const ID = '8ff3dd3c-4660-493e-b94a-bf04d293cc85';
eq('a collection links to its platform page',
   sourceHref({ doc_id: ID, element_type: 'collection' }),
   `https://platform.i-guide.io/collections/${ID}`);
eq('documentation lives under /docs, not /documentations',
   sourceHref({ doc_id: 'getting-started', element_type: 'documentation' }),
   'https://platform.i-guide.io/docs/getting-started');
eq('a "resource" still gets no invented link',
   sourceHref({ doc_id: ID, element_type: 'resource' }), '');

const dup = '777a106d-0000-4000-8000-000000000000';
const docs = evidenceDocs({ agent_result: { orchestration_result: { evidence: [
  { doc_id: dup, element_type: 'resource', source: 'neo4j', title: 'Spatial and Temporal Variability' },
  { doc_id: dup, element_type: 'publication', source: 'keyword', title: 'Spatial and Temporal Variability' },
] } } });
eq('one entry per element', docs.length, 1);
eq('the linkable copy is the one listed', sourceHref(docs[0]),
   `https://platform.i-guide.io/publications/${dup}`);

if (bad) { console.error(`${bad} check(s) failed`); process.exit(1); }
console.log('all source-link checks passed');
