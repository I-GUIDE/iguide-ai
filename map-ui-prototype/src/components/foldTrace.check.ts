/** A runnable check for foldTrace. `npm run check:fold`.
 *
 * This prototype has no test runner, and adding one costs more than the function under test —
 * but the fold arithmetic is exactly the kind of thing that is wrong by one and looks right in
 * a screenshot, so it gets checked rather than eyeballed. esbuild transpiles it and node runs
 * it; nothing imports it, so it is tree-shaken out of the bundle.
 *
 * The fixture is a REAL turn, copied from the live trace on 2026-09-09 rather than invented,
 * because the thing being measured is how many rows a real turn renders. */
import { foldTrace, visibleSteps } from './ChatPanel';

// The turn measured live on 2026-09-09: "Show me the city boundary of Savoy, Illinois."
const savoy = [
  { text: 'Routing the request', kind: 'node' },
  { text: 'Routed to orchestrate', kind: 'node' },
  { text: 'Orchestrator agent started', kind: 'node' },
  { text: 'supervisor → analyze (decision)', kind: 'node' },
  { text: 'Running analysis workflow', kind: 'node' },
  { text: 'Asking gpt-5.6-luna', kind: 'llm' },
  { text: "admin_boundary({'area': 'Savoy'})", kind: 'tool' },
  { text: '1 feature · 0.38s', kind: 'result' },
  { text: 'map: shapes — Savoy village (1 feature)', kind: 'tool' },
  { text: "Savoy, Illinois' municipal boundary is now shown…" },  // llm_message, untagged
  { text: 'Analysis workflow complete', kind: 'node' },
  { text: 'supervisor → done (decision)', kind: 'node' },
  { text: 'Composing answer', kind: 'node' },
  { text: 'Supervisor graph completed', kind: 'node' },
];

const rows = foldTrace(savoy);
let bad = 0;
const eq = (label: string, got: unknown, want: unknown) => {
  const ok = JSON.stringify(got) === JSON.stringify(want);
  if (!ok) bad++;
  console.log(`${ok ? 'ok  ' : 'FAIL'} ${label}: got ${JSON.stringify(got)}`);
};

eq('rows rendered', rows.length, 7);
eq('opening run collapses to its first line', rows[0].line.text, 'Routing the request');
eq('  ...hiding four', rows[0].folded?.length, 4);
eq('closing run collapses to its first line', rows[6].line.text, 'Analysis workflow complete');
eq('  ...hiding three', rows[6].folded?.length, 3);
eq('the tool call survives', rows[2].line.text, "admin_boundary({'area': 'Savoy'})");
eq('so does its result', rows[3].line.text, '1 feature · 0.38s');
eq('nothing is lost', rows.reduce((n, r) => n + 1 + (r.folded?.length || 0), 0), savoy.length);

// A run of one must not grow a "+0" badge.
const single = foldTrace([{ text: 'a', kind: 'node' }, { text: 'b', kind: 'tool' }]);
eq('a lone node row is not a run', single[0].folded, undefined);

// The llm ladder still folds globally, not per run.
const ladder = foldTrace([
  { text: 'Asking m', kind: 'llm' }, { text: 't1', kind: 'tool' },
  { text: 'Asking m', kind: 'llm' }, { text: 't2', kind: 'tool' },
]);
eq('only the first model line survives', ladder.length, 3);

// An llm row between two node rows breaks the run — they are not the same phase.
const split = foldTrace([
  { text: 'n1', kind: 'node' }, { text: 'Asking m', kind: 'llm' }, { text: 'n2', kind: 'node' },
]);
eq('a run is consecutive only', split.length, 3);

// The tally must equal what is on screen. Runs render expanded, so that is every row the
// fold kept — not the collapsed row count, which called a fifteen-row transcript "7 steps".
eq('the tally counts every visible row', visibleSteps(savoy), 14);
eq('  ...every stored row, this turn asking the model once', visibleSteps(savoy), savoy.length);
eq('a second model line is excluded, being folded away', visibleSteps(
  [...savoy, { text: 'Asking gpt-5.6-luna', kind: 'llm' }]), 14);

console.log(bad ? `\n${bad} FAILED` : '\nall passed');
process.exit(bad ? 1 : 0);
