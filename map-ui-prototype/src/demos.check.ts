// The Demos catalogue is generated (scripts/build_demos.py); this checks what the tab relies on.
import { existsSync, statSync } from 'node:fs';
import { join } from 'node:path';

import { DEMOS, demoBytes, formatBytes, nextStep } from './demos';

let failed = 0;
const ok = (cond: unknown, msg: string) => { if (!cond) { failed++; console.error('FAIL', msg); } };

const ids = DEMOS.map((d) => d.id);
ok(DEMOS.length >= 5, `at least 5 demos, got ${DEMOS.length}`);
ok(new Set(ids).size === ids.length, 'demo ids are unique');
for (const d of DEMOS) {
  ok(/^[a-z0-9-]+$/.test(d.id), `${d.id}: id is a url-safe slug`);
  ok(d.steps.length >= 1, `${d.id}: has steps`);
  ok(/harness runs$/.test(d.reliability), `${d.id}: says how often step 1 has worked`);
  for (const [k, s] of d.steps.entries()) {
    ok(s.label.trim() && s.prompt.trim().length > 20, `${d.id} step ${k + 1}: label and a real question`);
  }
  for (const f of d.files) {
    const p = join('public', f.path);
    ok(existsSync(p), `${d.id}: ${f.path} exists`);
    if (existsSync(p)) ok(statSync(p).size === f.bytes, `${d.id}: ${f.name} is ${f.bytes} bytes as catalogued`);
    // The first question names its files, so the agent knows what it was given.
    ok(d.steps[0].prompt.includes(f.name), `${d.id}: step 1 names ${f.name}`);
  }
  ok(demoBytes(d) < 10 * 1024 * 1024, `${d.id}: under 10 MB (${formatBytes(demoBytes(d))})`);
}
const c = DEMOS[0];
ok(nextStep(c, new Set()) === 0, 'next step starts at 1');
ok(nextStep(c, new Set([0])) === (c.steps.length > 1 ? 1 : null), 'next step follows the asked ones');
ok(nextStep(c, new Set(c.steps.map((_, k) => k))) === null, 'no next step when all are asked');
ok(formatBytes(512) === '512 B' && formatBytes(2048) === '2 KB' && formatBytes(5 * 1048576) === '5.0 MB', 'formatBytes');

if (failed) { console.error(`${failed} demo check(s) failed`); process.exit(1); }
console.log(`demos ok: ${DEMOS.length} cases, ${DEMOS.reduce((n, d) => n + d.steps.length, 0)} steps`);
