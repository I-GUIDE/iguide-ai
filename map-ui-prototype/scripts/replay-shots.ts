// Screenshot recorded agent turns as the map UI shows them.
//   npm run replay:shots -- <absolute path>...     (npm runs it from the package root)
//   python -m gis_harness.screenshots <path>...    (the same, from the repo root)
//
// Every <task>[.t<n>].events.jsonl under the given paths is replayed through the real page (a
// VITE_REPLAY build, see src/replay.ts) in headless Chrome, and <task>[.t<n>].png is written next
// to it: the replay strip, the whole transcript (answer, banners, downloads), and the map with
// the turn's layers. Each directory also gets screenshots.jsonl: per turn, layers drawn of total,
// the reasons any were not, seconds and bytes.
//
//   --force        redo turns whose .png is newer than their events
//   --parallel N   pages at once (default 4)
//   --no-build     reuse dist-replay/ as it is
//   --keep-open    leave the server up after the run and print its url, to look at a turn by hand
//
// Chrome: the installed Google Chrome by default; REPLAY_CHROME=<path> for another binary.
import { spawnSync } from 'node:child_process';
import { existsSync, readFileSync, readdirSync, statSync, writeFileSync } from 'node:fs';
import { createServer } from 'node:http';
import type { AddressInfo } from 'node:net';
import { dirname, extname, join, relative, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { chromium, type BrowserContext } from 'playwright-core';
import { replayHandler, stemOf } from './replay-server';

// The bundle runs from node_modules/.cache, two levels below the package root.
const UI = resolve(dirname(fileURLToPath(import.meta.url)), '..', '..');
const ROOT = existsSync(join(UI, 'package.json')) ? UI : process.cwd();
const DIST = join(ROOT, 'dist-replay');
const SKIP_DIRS = new Set(['layers', 'server_files', '_data', 'node_modules', 'publish']);

const args = process.argv.slice(2);
const flag = (f: string) => { const i = args.indexOf(f); if (i === -1) return false; args.splice(i, 1); return true; };
const opt = (f: string, d: string) => { const i = args.indexOf(f); if (i === -1) return d; const v = args[i + 1]; args.splice(i, 2); return v; };
const force = flag('--force');
const noBuild = flag('--no-build');
const keepOpen = flag('--keep-open');
const parallel = Math.max(1, Number(opt('--parallel', '4')) || 4);
if (!args.length) { console.error('usage: replay:shots -- <dir|events.jsonl>... [--force] [--parallel N]'); process.exit(2); }

function walk(p: string, out: string[]) {
  const st = statSync(p);
  if (st.isFile()) { if (p.endsWith('.events.jsonl')) out.push(p); return; }
  for (const n of readdirSync(p).sort()) {
    if (SKIP_DIRS.has(n) || n.startsWith('.')) continue;
    walk(join(p, n), out);
  }
}
const all: string[] = [];
for (const a of args) walk(resolve(a), all);
const pngOf = (ev: string) => join(dirname(ev), `${stemOf(ev)}.png`);
const todo = all.filter((ev) => force || !existsSync(pngOf(ev)) || statSync(pngOf(ev)).mtimeMs < statSync(ev).mtimeMs);
console.log(`${all.length} recorded turns, ${todo.length} to render`);
if (!todo.length && !keepOpen) process.exit(0);

if (!noBuild || !existsSync(join(DIST, 'index.html'))) {
  const t = Date.now();
  const b = spawnSync('npx', ['vite', 'build', '--outDir', 'dist-replay', '--emptyOutDir', '--logLevel', 'error'],
                      { cwd: ROOT, env: { ...process.env, VITE_REPLAY: '1' }, stdio: 'inherit' });
  if (b.status !== 0) process.exit(b.status ?? 1);
  console.log(`replay build ${((Date.now() - t) / 1000).toFixed(1)} s`);
}

// Keys are positions in this run's list, so the page can name no file the run did not offer.
const keys = new Map(todo.map((ev, i) => [String(i), ev]));
const replay = replayHandler((k) => keys.get(k) ?? null);
const STATIC: Record<string, string> = { '.html': 'text/html', '.js': 'text/javascript', '.css': 'text/css',
  '.png': 'image/png', '.svg': 'image/svg+xml', '.json': 'application/json', '.woff2': 'font/woff2' };
const server = createServer((req, res) => {
  if (replay(req, res)) return;
  const path = new URL(req.url || '/', 'http://x').pathname;
  const file = join(DIST, path === '/' ? 'index.html' : path);
  if (!file.startsWith(DIST) || !existsSync(file) || statSync(file).isDirectory()) {
    res.writeHead(404); res.end(); return;
  }
  res.writeHead(200, { 'content-type': STATIC[extname(file)] || 'application/octet-stream' });
  res.end(readFileSync(file));
});
await new Promise<void>((r) => server.listen(0, '127.0.0.1', r));
const base = `http://127.0.0.1:${(server.address() as AddressInfo).port}`;

// One persistent profile, so basemap tiles come from Chrome's cache after the first turn in an
// area instead of from tile.openstreetmap.org again.
const ctx: BrowserContext = await chromium.launchPersistentContext(join(ROOT, 'node_modules', '.cache', 'replay-profile'), {
  headless: true,
  ...(process.env.REPLAY_CHROME ? { executablePath: process.env.REPLAY_CHROME } : { channel: 'chrome' }),
  viewport: { width: 1440, height: 900 },
  deviceScaleFactor: 1,
  args: ['--enable-unsafe-swiftshader', '--ignore-gpu-blocklist'],
});

interface Shot { turn: string; png: string; status: string; drawn?: number; total?: number;
  problems?: string[]; seconds: number; bytes: number; page_errors: string[] }
const results = new Map<string, Shot[]>();

async function shoot(key: string, ev: string): Promise<void> {
  const t0 = Date.now();
  const page = await ctx.newPage();
  const errors: string[] = [];
  page.on('pageerror', (e) => errors.push(String(e.message).slice(0, 300)));
  let status = 'done';
  try {
    await page.goto(`${base}/?replay=${key}`, { waitUntil: 'domcontentloaded' });
    // The turn ends when its layers have downloaded (or LAYER_WAIT_MS in App.tsx gave up on them).
    await page.waitForFunction(() => !!document.documentElement.dataset.replay, null, { timeout: 60_000 })
      .catch(() => { status = 'timeout'; });
    if (status === 'done') status = await page.evaluate(() => document.documentElement.dataset.replay || 'timeout');
    // The map settles: mounted, its style and tiles loaded, and the fit's ease finished. A map
    // that is on screen but not yet published counts as NOT settled: the replay delivers every
    // layer within milliseconds, so the map mounts after the turn ends and applies its pending
    // fit only then. Waiting for "no map handle" let the first backfill shoot mid-ease, a
    // 100-cell lattice as a speck at zoom 7. Settled = three quiet polls in a row. Bounded: a
    // tile server that does not answer leaves grey squares, not a hung run.
    let mapStatus = 'settled';
    await page.waitForFunction(() => {
      const w = window as any;
      if (!document.querySelector('.mapwrap')) return true;      // a chat-only turn
      const m = w.__map;
      const quiet = !!m && m.loaded() && m.areTilesLoaded() && !m.isMoving();
      w.__replayQuiet = quiet ? (w.__replayQuiet || 0) + 1 : 0;
      return w.__replayQuiet >= 3;
    }, null, { timeout: 20_000, polling: 250 }).catch(() => { mapStatus = 'unsettled'; });
    // The whole transcript: grow the window to the page instead of a full-page capture, which
    // tiles the first screen down a page whose root is 100% of the window. The map's height is
    // fixed in replay-capture mode, so its framing does not change with the window.
    const h = await page.evaluate(() => Math.ceil(document.querySelector('.app')?.scrollHeight || 900));
    if (h > 900) await page.setViewportSize({ width: 1440, height: Math.min(h, 16_000) });
    await page.waitForTimeout(600);
    const png = pngOf(ev);
    await page.screenshot({ path: png });
    const v = await page.evaluate(() => {
      const strip = document.querySelector('.replay-strip');
      const lines = [...document.querySelectorAll('.replay-strip .replay-line')].map((e) => e.textContent || '');
      const m = /map layers: (\d+) of (\d+) drawn/.exec(strip?.textContent || '');
      return { drawn: m ? +m[1] : undefined, total: m ? +m[2] : undefined, lines };
    });
    if (status === 'done' && mapStatus !== 'settled') status = 'map-unsettled';
    const shot: Shot = { turn: stemOf(ev), png: relative(dirname(ev), png), status,
      drawn: v.drawn, total: v.total, problems: v.lines,
      seconds: +((Date.now() - t0) / 1000).toFixed(2), bytes: statSync(png).size, page_errors: errors };
    const list = results.get(dirname(ev)) ?? [];
    list.push(shot); results.set(dirname(ev), list);
    console.log(`${status === 'done' ? 'ok  ' : status.toUpperCase().padEnd(4)} ${relative(process.cwd(), png)}`
      + `  layers ${v.drawn ?? '?'}/${v.total ?? '?'}  ${shot.seconds}s  ${(shot.bytes / 1e6).toFixed(2)} MB`
      + (v.lines.length ? `\n       ${v.lines.join('\n       ')}` : ''));
  } finally {
    await page.close();
  }
}

const t0 = Date.now();
const queue = [...keys.entries()];
await Promise.all(Array.from({ length: Math.min(parallel, queue.length) }, async () => {
  for (let job = queue.shift(); job; job = queue.shift()) {
    try { await shoot(job[0], job[1]); } catch (e: any) { console.error(`FAIL ${job[1]}: ${e.message}`); }
  }
}));

// screenshots.jsonl, not .json: `run.py --summarise`, `--compare` and the archive's tools read
// every .json under a run as a turn record.
for (const [dir, shots] of results) {
  const f = join(dir, 'screenshots.jsonl');
  const merged = new Map<string, Shot>();
  try {
    for (const l of readFileSync(f, 'utf8').split('\n')) if (l.trim()) { const s = JSON.parse(l); merged.set(s.turn, s); }
  } catch { /* first time */ }
  for (const s of shots) merged.set(s.turn, s);
  const turns = [...merged.values()].sort((a, b) => a.turn.localeCompare(b.turn));
  writeFileSync(f, turns.map((s) => JSON.stringify(s)).join('\n') + '\n');
}
const n = [...results.values()].reduce((a, s) => a + s.length, 0);
console.log(`${n} screenshots in ${((Date.now() - t0) / 1000).toFixed(1)} s, ${parallel} at a time`);

if (keepOpen) {
  console.log(`replay server: ${base}/?replay=<0..${keys.size - 1}>   (Ctrl-C to stop)`);
} else {
  await ctx.close();
  server.close();
}
