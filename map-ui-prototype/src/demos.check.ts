// The Demos catalogue is generated (scripts/build_demos.py); this checks what the tab relies on.
import { existsSync, statSync } from 'node:fs';
import { join } from 'node:path';

import { DEMOS, DEMO_CATEGORIES, demoBytes, demosByCategory, formatBytes, nextStep } from './demos';

let failed = 0;
const ok = (cond: unknown, msg: string) => { if (!cond) { failed++; console.error('FAIL', msg); } };

const ids = DEMOS.map((d) => d.id);
ok(DEMOS.length >= 5, `at least 5 demos, got ${DEMOS.length}`);
ok(new Set(ids).size === ids.length, 'demo ids are unique');
for (const d of DEMOS) {
  ok(/^[a-z0-9-]+$/.test(d.id), `${d.id}: id is a url-safe slug`);
  ok(d.steps.length >= 1, `${d.id}: has steps`);
  ok((DEMO_CATEGORIES as readonly string[]).includes(d.category), `${d.id}: category "${d.category}" is one of DEMO_CATEGORIES`);
  if (d.harnessTask) ok(/harness runs$/.test(d.reliability ?? ''), `${d.id}: says how often step 1 has worked`);
  // A live case fetches its own data: nothing to upload, and no harness figure to quote.
  else ok(d.files.length === 0 && d.reliability === undefined, `${d.id}: a live case uploads nothing`);
  for (const [k, s] of d.steps.entries()) {
    ok(s.label.trim() && s.prompt.trim().length > 20, `${d.id} step ${k + 1}: label and a real question`);
  }
  for (const f of d.files) {
    const p = join('public', f.path);
    ok(existsSync(p), `${d.id}: ${f.path} exists`);
    if (existsSync(p)) ok(statSync(p).size === f.bytes, `${d.id}: ${f.name} is ${f.bytes} bytes as catalogued`);
    // Step 1 states the task only (user, 2026-10-15): the agent reads the files itself, and
    // "Explain files" asks it to describe them. Short prompts: 11 of 12 demos still correct;
    // the one miss was a legend that existed only in the prompt, now in the data.
    ok(!d.steps[0].prompt.includes(f.name), `${d.id}: step 1 does not describe ${f.name}`);
  }
  ok(demoBytes(d) < 10 * 1024 * 1024, `${d.id}: under 10 MB (${formatBytes(demoBytes(d))})`);
}
const groups = demosByCategory();
ok(groups.reduce((n, g) => n + g.cases.length, 0) === DEMOS.length, 'every case is in exactly one group');
ok(groups.map((g) => g.category).join('|') === DEMO_CATEGORIES.filter((k) => DEMOS.some((d) => d.category === k)).join('|'),
   'groups follow DEMO_CATEGORIES order');
ok(demosByCategory([{ ...DEMOS[0], category: 'Something new' }])[0].category === 'Something new',
   'an unlisted category is shown, not dropped');
ok(groups[0].cases[0].id === 'chicago-crime', 'the Chicago case is listed first');
const c = DEMOS[0];
ok(nextStep(c, new Set()) === 0, 'next step starts at 1');
ok(nextStep(c, new Set([0])) === (c.steps.length > 1 ? 1 : null), 'next step follows the asked ones');
ok(nextStep(c, new Set(c.steps.map((_, k) => k))) === null, 'no next step when all are asked');
ok(formatBytes(512) === '512 B' && formatBytes(2048) === '2 KB' && formatBytes(5 * 1048576) === '5.0 MB', 'formatBytes');

if (failed) { console.error(`${failed} demo check(s) failed`); process.exit(1); }
console.log(`demos ok: ${DEMOS.length} cases, ${DEMOS.reduce((n, d) => n + d.steps.length, 0)} steps`);
