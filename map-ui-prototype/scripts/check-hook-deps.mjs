/**
 * Catch the one hook mistake this app keeps making. `npm run check:hooks`.
 *
 * Three times now, a `useCallback` has read a value that changes DURING a session and left it
 * out of its dependency array, so the callback kept the value it saw on the first render:
 *
 *   refreshSessions  omitted `viewer`          -> the list never refreshed when identity landed
 *   restoreSession   omitted `tokenMode`       -> clicking a conversation did nothing at all
 *   runLive          omitted `snapshotSession` -> every turn skipped the server save and then
 *                                                 redrew the history from the local cache
 *
 * All three share a shape: the value is false/null on the first render and becomes real once
 * `/agent/ui-config` and `/agent/whoami` answer. A callback memoised before that keeps the
 * pre-identity world forever, and the symptom never points at the closure — it looks like a
 * routing bug, a dead button, a wrong list.
 *
 * This is deliberately NOT a general exhaustive-deps implementation. It watches a named list of
 * session-lifecycle values and asks one question: if a callback reads it, is it declared? A
 * `*Ref` read is fine by construction, which is why the file uses refs for the rest.
 */
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const here = dirname(fileURLToPath(import.meta.url));
const FILES = [join(here, '..', 'src', 'App.tsx')];

// Values that are one thing before the deployment answers and another after.
const WATCHED = ['tokenMode', 'demoMode', 'viewer', 'snapshotSession', 'refreshSessions',
                 'restoreSession'];

let problems = 0;

for (const file of FILES) {
  const src = readFileSync(file, 'utf8');
  // Walk every `useCallback(` and find its argument list by matching parentheses, so a nested
  // callback or an object literal in the body cannot end the block early.
  for (let i = src.indexOf('useCallback('); i !== -1; i = src.indexOf('useCallback(', i + 1)) {
    const open = src.indexOf('(', i);
    let depth = 0, end = -1;
    for (let j = open; j < src.length; j++) {
      if (src[j] === '(') depth++;
      else if (src[j] === ')') { depth--; if (depth === 0) { end = j; break; } }
    }
    if (end === -1) continue;
    const call = src.slice(open + 1, end);
    // The dep array is the last top-level `[...]` in the call.
    const lastBracket = call.lastIndexOf('[');
    if (lastBracket === -1) continue;
    const deps = call.slice(lastBracket, call.lastIndexOf(']') + 1);
    // Comments are stripped before matching. They are prose ABOUT these values — this file
    // documents each of these bugs at the site it fixed them — and a checker that reads its own
    // explanation as a code reference reports every fix as the bug it describes.
    const body = call.slice(0, lastBracket)
      .replace(/\/\*[\s\S]*?\*\//g, ' ')
      .replace(/(^|[^:])\/\/[^\n]*/g, '$1 ');

    const name = (src.slice(Math.max(0, i - 80), i).match(/const (\w+)\s*=\s*$/) || [])[1]
      || `useCallback at char ${i}`;

    for (const value of WATCHED) {
      const reads = new RegExp(`(?<![.\\w])${value}(?![\\w])`).test(body);
      const declared = new RegExp(`(?<![.\\w])${value}(?![\\w])`).test(deps);
      if (reads && !declared) {
        const line = src.slice(0, i).split('\n').length;
        console.log(`FAIL ${file.split('/').pop()}:${line} ${name} reads \`${value}\` `
          + `but does not declare it.\n     deps = ${deps.replace(/\s+/g, ' ')}`);
        problems++;
      }
    }
  }
}

if (problems) {
  console.log(`\n${problems} stale-closure risk(s). Add the value to the deps, or read it `
    + `through a ref the way viewerRef does.`);
  process.exit(1);
}
console.log(`ok   no callback reads a session-lifecycle value it has not declared `
  + `(${WATCHED.length} watched)`);
